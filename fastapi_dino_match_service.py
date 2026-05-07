import io
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw

from dino import build_default_extractor
from find_similar_region_with_dino_tokens import (
    compute_similarity_map,
    extract_patch_tokens,   
    judge_match_confidence,
    resize_small_image_with_scale,
    select_top_candidates,
    token_box_to_pixel_box,
)


# 配置日志，便于独立排查匹配服务问题
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "dino_match_service.log"


class SafeFileHandler(logging.FileHandler):
    """
    更稳妥的文件日志处理器。

    设计目的：
    - 避免日志目录被清理后，下一次写日志时因为父目录不存在直接抛异常
    - 每次真正打开文件前都再次确认父目录存在
    """

    def _open(self):
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


def configure_logging() -> logging.Logger:
    """
    显式配置日志处理器，避免被其它模块提前调用 `basicConfig()` 后失效。

    设计要点：
    - 文件日志挂到 root logger，保证当前服务及其依赖模块日志都能落盘
    - 控制台 handler 尽量复用现有配置，避免重复打印
    - 多次 import / reload 时不重复添加同一个文件 handler
    """
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_path_str = str(LOG_FILE_PATH.resolve())
    has_same_file_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == Path(log_file_path_str):
                    has_same_file_handler = True
                    break
            except Exception:
                continue

    if not has_same_file_handler:
        file_handler = SafeFileHandler(LOG_FILE_PATH, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger_instance = logging.getLogger("dino_match_service")
    logger_instance.setLevel(logging.INFO)
    logger_instance.propagate = True
    return logger_instance


logger = configure_logging()


# 局部搜索缓存：
# - key: small_image_path
# - value: 上一次高置信匹配到的像素框与帧尺寸
MATCH_REGION_CACHE: dict[str, dict[str, Any]] = {}
MATCH_REGION_CACHE_LOCK = Lock()
SMALL_IMAGE_TOKEN_CACHE: dict[str, dict[str, Any]] = {}
SMALL_IMAGE_TOKEN_CACHE_LOCK = Lock()
DEFAULT_ROI_MIN_SIZE = 400
DEFAULT_ROI_EXPAND_RATIO = 2.5
DEFAULT_BEST_SCORE_Z_THRESHOLD = 1.0
DEFAULT_SCALE_CONSISTENCY_IOU_THRESHOLD = 0.3
DEFAULT_SCALE_SCORE_TOLERANCE = 0.03
DEFAULT_QUERY_CONSISTENCY_IOU_THRESHOLD = 0.3
DEFAULT_QUERY_SCORE_TOLERANCE = 0.08
DEFAULT_QUERY_BOX_CLUSTER_IOU_THRESHOLD = 0.3
DEFAULT_LOW_SCORE_CONSENSUS_THRESHOLD = 0.41
DEFAULT_LOW_SCORE_CONSENSUS_MAX_THRESHOLD = 0.43
DEFAULT_LOW_SCORE_CONSENSUS_BEST_SCORE_Z_THRESHOLD = 3.0
DEFAULT_STRONG_SCORE_MARGIN = 0.08
DEFAULT_TEMPORAL_CONSISTENCY_IOU_THRESHOLD = 0.3
DEFAULT_ROI_DIRECT_SCORE_MARGIN = 0.02
DEFAULT_ROI_DIRECT_TEMPORAL_IOU_THRESHOLD = 0.65

# 对外返回的核心置信字段：
# - 只保留业务上真正需要消费的判断结果与关键依据
# - 其余调试型字段统一归到 debug，避免接口返回过于臃肿
CORE_MATCH_CONFIDENCE_KEYS = {
    "is_match_confident",
    "reason",
    "top1_score",
    "top2_score",
    "top1_top2_margin",
    "score_threshold",
    "margin_threshold",
    "best_score_z",
    "is_peak_prominent",
    "is_scale_consistent",
    "scale_support_count",
    "required_scale_support",
    "is_query_consistent",
    "query_support_count",
    "required_query_support",
    "is_temporally_consistent",
    "temporal_iou_with_cached_box",
    "temporal_consistency_override_applied",
    "is_roi_directly_acceptable",
}


def format_scale_cache_key(scale: float) -> str:
    """
    规范化 scale 的缓存键，避免浮点数精度差异导致缓存失效。
    """
    return f"{float(scale):.6f}"


def build_small_image_token_cache_key(query_image_path: str, scale: float) -> str:
    """
    生成小图 token 缓存键。

    这里把绝对路径和 scale 一起作为缓存键，确保不同文件、不同尺度互不影响。
    """
    resolved_path = str(Path(query_image_path).expanduser().resolve())
    return f"{resolved_path}::{format_scale_cache_key(scale)}"


def get_small_image_file_signature(image_path: Path) -> dict[str, int]:
    """
    获取小图文件签名，用于在文件被替换后自动让缓存失效。
    """
    file_stat = image_path.stat()
    return {
        "mtime_ns": int(file_stat.st_mtime_ns),
        "file_size": int(file_stat.st_size),
    }


def get_or_extract_small_image_scale_info(
    extractor,
    query_image_path: str,
    scale: float,
) -> dict[str, Any]:
    """
    获取某张小图在指定 scale 下的 token 信息，优先走缓存。

    返回内容：
    - query_image_size: 原图尺寸
    - scaled_small_image_size: 缩放后尺寸
    - small_info: extract_patch_tokens 的结果
    - cache_hit: 是否命中缓存
    """
    image_path = Path(query_image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"小图文件不存在: {image_path}")

    cache_key = build_small_image_token_cache_key(str(image_path), scale)
    file_signature = get_small_image_file_signature(image_path)

    with SMALL_IMAGE_TOKEN_CACHE_LOCK:
        cached_value = SMALL_IMAGE_TOKEN_CACHE.get(cache_key)

    if cached_value is not None and cached_value.get("file_signature") == file_signature:
        return {
            "query_image_size": dict(cached_value["query_image_size"]),
            "scaled_small_image_size": dict(cached_value["scaled_small_image_size"]),
            "small_info": cached_value["small_info"],
            "cache_hit": True,
        }

    with Image.open(image_path) as image:
        query_image = image.convert("RGB")

    scaled_query_image = resize_small_image_with_scale(query_image, scale=scale)
    small_info = extract_patch_tokens(extractor, scaled_query_image)
    cache_value = {
        "file_signature": file_signature,
        "query_image_size": {
            "width": int(query_image.size[0]),
            "height": int(query_image.size[1]),
        },
        "scaled_small_image_size": {
            "width": int(scaled_query_image.size[0]),
            "height": int(scaled_query_image.size[1]),
        },
        "small_info": small_info,
    }

    with SMALL_IMAGE_TOKEN_CACHE_LOCK:
        SMALL_IMAGE_TOKEN_CACHE[cache_key] = cache_value

    return {
        "query_image_size": dict(cache_value["query_image_size"]),
        "scaled_small_image_size": dict(cache_value["scaled_small_image_size"]),
        "small_info": small_info,
        "cache_hit": False,
    }


def prepare_small_image_scale_infos(
    extractor,
    query_image_path: str,
    scales: list[float],
) -> dict[str, Any]:
    """
    预先准备某张小图在多尺度下的 token 信息，并统计缓存命中情况。

    这样做的目的，是把“读图 + resize + token 提取”从主匹配循环中抽出来，
    便于跨请求复用，也便于单独统计这块到底花了多少时间。
    """
    if not scales:
        raise ValueError("scales 不能为空")

    prepared_scale_infos: list[dict[str, Any]] = []
    query_image_size: tuple[int, int] | None = None
    cache_hit_count = 0
    cache_miss_count = 0
    total_prepare_elapsed = 0.0

    for scale in scales:
        prepare_start_time = perf_counter()
        scale_info = get_or_extract_small_image_scale_info(
            extractor=extractor,
            query_image_path=query_image_path,
            scale=scale,
        )
        prepare_elapsed = perf_counter() - prepare_start_time
        total_prepare_elapsed += prepare_elapsed

        if query_image_size is None:
            query_image_size = (
                int(scale_info["query_image_size"]["width"]),
                int(scale_info["query_image_size"]["height"]),
            )

        cache_hit = bool(scale_info["cache_hit"])
        if cache_hit:
            cache_hit_count += 1
        else:
            cache_miss_count += 1

        prepared_scale_infos.append({
            "scale": float(scale),
            "small_info": scale_info["small_info"],
            "scaled_small_image_size": dict(scale_info["scaled_small_image_size"]),
            "small_token_cache_hit": cache_hit,
            "small_token_prepare_elapsed_sec": float(prepare_elapsed),
        })

    if query_image_size is None:
        raise ValueError(f"未能成功准备小图 token: {query_image_path}")

    return {
        "query_image_size": query_image_size,
        "scale_infos": prepared_scale_infos,
        "small_token_cache_hit_count": int(cache_hit_count),
        "small_token_cache_miss_count": int(cache_miss_count),
        "total_prepare_elapsed_sec": float(total_prepare_elapsed),
    }


def parse_scales(scales_text: str) -> list[float]:
    """
    解析多尺度匹配配置。

    接口里使用逗号分隔字符串，便于通过 form-data 直接传参。
    """
    raw_values = [item.strip() for item in scales_text.split(",")]
    scales = [float(item) for item in raw_values if item]
    if not scales:
        raise ValueError("scales 不能为空")
    return scales


def is_image_file(path: Path) -> bool:
    """
    判断一个文件是否为常见图片格式。
    """
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_query_image_paths(
    small_image_path: str | None,
    small_image_dir: str | None,
) -> list[str]:
    """
    收集参与匹配的小图路径。

    规则：
    - `small_image_path` 与 `small_image_dir` 必须二选一
    - 如果传目录，则收集目录下全部常见图片文件并按文件名排序
    """
    if bool(small_image_path) == bool(small_image_dir):
        raise ValueError("必须且只能提供一个参数：small_image_path 或 small_image_dir")

    if small_image_path is not None:
        image_path = Path(small_image_path).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"小图文件不存在: {image_path}")
        return [str(image_path.resolve())]

    image_dir = Path(small_image_dir).expanduser()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"小图目录不存在: {image_dir}")

    image_paths = sorted(
        str(path.resolve())
        for path in image_dir.iterdir()
        if path.is_file() and is_image_file(path)
    )
    if not image_paths:
        raise ValueError(f"目录下未找到可用图片: {image_dir}")

    return image_paths


def load_rgb_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    从上传的二进制内容中读取 RGB 图片。
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def parse_bool_value(value: str) -> bool:
    """
    解析表单中的布尔值。

    兼容常见写法，避免调用方只能传严格的 true/false。
    """
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"无法解析布尔值: {value}")


def convert_cv2_frame_to_rgb_image(frame: np.ndarray) -> Image.Image:
    """
    将 cv2 读取到的 BGR 帧转换为 PIL RGB 图片。

    说明：
    - `cap.read()` 返回的 frame 通常是 HWC 格式的 BGR ndarray
    - 这里不强依赖 cv2.cvtColor，直接通过通道翻转转成 RGB
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame 必须是 numpy.ndarray，当前类型: {type(frame)}")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame 必须是 3 通道彩色图，当前 shape: {getattr(frame, 'shape', None)}"
        )

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    return Image.fromarray(rgb_frame, mode="RGB")


def save_annotated_frame_image(
    frame_image: Image.Image,
    best_box: list[int],
    is_match_confident: bool,
    annotated_boxes: list[dict[str, Any]] | None = None,
    output_path: str | None = None,
) -> str:
    """
    将匹配框画到帧图上并保存。

    说明：
    - 单模板场景下，默认只画 `best_box`
    - 多模板场景下，如果传入 `annotated_boxes`，则会把每张小图的最佳框都画出来
    - 如果调用方未指定输出路径，则默认保存在当前目录下
    """
    if output_path is None or not output_path.strip():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")
        confidence_suffix = "1" if is_match_confident else "0"
        output_path = f"out_image/{timestamp}_{confidence_suffix}.jpg"

    output_file = Path(output_path).expanduser()
    if not output_file.is_absolute():
        output_file = Path.cwd() / output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotated_image = frame_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # 为多模板结果准备一组容易区分的颜色，便于直接肉眼区分不同小图的最佳框。
    box_colors = [
        "red",
        "lime",
        "yellow",
        "cyan",
        "magenta",
        "orange",
        "white",
    ]

    normalized_annotated_boxes: list[dict[str, Any]] = []
    if annotated_boxes:
        for index, item in enumerate(annotated_boxes, start=1):
            pixel_box = item.get("pixel_box")
            if pixel_box is None:
                continue
            normalized_annotated_boxes.append({
                "rank": int(item.get("rank", index)),
                "query_image_name": str(item.get("query_image_name", f"query_{index}")),
                "best_score": item.get("best_score"),
                "pixel_box": [int(x) for x in pixel_box],
            })

    if not normalized_annotated_boxes:
        normalized_annotated_boxes = [{
            "rank": 1,
            "query_image_name": "best_match",
            "best_score": None,
            "pixel_box": [int(x) for x in best_box],
        }]

    for index, item in enumerate(normalized_annotated_boxes):
        box_color = box_colors[index % len(box_colors)]
        pixel_box = item["pixel_box"]
        draw.rectangle(pixel_box, outline=box_color, width=4)

        score_value = item.get("best_score")
        score_text = ""
        if score_value is not None:
            try:
                score_text = f" | {float(score_value):.3f}"
            except (TypeError, ValueError):
                score_text = f" | {score_value}"

        label_text = f"{item['rank']}. {item['query_image_name']}{score_text}"

        # 标签优先画在框上方；如果上方空间不足，则贴到框内顶部，避免超出图像边界。
        label_x = int(pixel_box[0])
        label_y = int(max(0, pixel_box[1] - 18))
        if label_y == 0:
            label_y = int(min(frame_image.size[1] - 16, pixel_box[1] + 2))

        draw.text(
            (label_x, label_y),
            label_text,
            fill=box_color,
        )

    annotated_image.save(output_file)

    logger.info(
        "已保存画框结果图: %s | image_name=%s | box_count=%s",
        output_file,
        output_file.name,
        len(normalized_annotated_boxes),
    )
    return str(output_file.resolve())


def crop_roi_around_box(
    frame_image: Image.Image,
    center_box: list[int],
    roi_expand_ratio: float = DEFAULT_ROI_EXPAND_RATIO,
    roi_min_size: int = DEFAULT_ROI_MIN_SIZE,
) -> tuple[Image.Image, dict[str, int]]:
    """
    以历史命中框中心为基准，按目标框大小自适应扩展 ROI。

    规则：
    - 先读取上一帧命中框的宽高
    - 再按 `roi_expand_ratio` 对宽高分别扩展
    - 同时用 `roi_min_size` 做下限兜底，避免目标框较小时 ROI 过小

    返回：
    - roi_image: 裁出的 ROI 图
    - roi_info: ROI 在原图中的偏移与尺寸信息
    """
    frame_width, frame_height = frame_image.size
    roi_expand_ratio = max(1.0, float(roi_expand_ratio))
    roi_min_size = max(32, int(roi_min_size))

    box_width = max(1, int(center_box[2] - center_box[0]))
    box_height = max(1, int(center_box[3] - center_box[1]))
    roi_width = max(roi_min_size, int(round(box_width * roi_expand_ratio)))
    roi_height = max(roi_min_size, int(round(box_height * roi_expand_ratio)))

    center_x = int(round((center_box[0] + center_box[2]) / 2))
    center_y = int(round((center_box[1] + center_box[3]) / 2))
    half_width = roi_width // 2
    half_height = roi_height // 2

    left = max(0, center_x - half_width)
    top = max(0, center_y - half_height)
    right = min(frame_width, left + roi_width)
    bottom = min(frame_height, top + roi_height)

    # 如果靠近边界，右/下裁回后，左/上再尽量补齐到目标 ROI 宽高。
    left = max(0, right - roi_width)
    top = max(0, bottom - roi_height)

    roi_image = frame_image.crop((left, top, right, bottom))
    roi_info = {
        "left": int(left),
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
        "width": int(right - left),
        "height": int(bottom - top),
        "source_box_width": int(box_width),
        "source_box_height": int(box_height),
        "expand_ratio": float(roi_expand_ratio),
        "roi_min_size": int(roi_min_size),
    }
    return roi_image, roi_info


def offset_pixel_box(pixel_box: list[int], offset_x: int, offset_y: int) -> list[int]:
    """
    将 ROI 内的像素框映射回原图坐标。
    """
    return [
        int(pixel_box[0] + offset_x),
        int(pixel_box[1] + offset_y),
        int(pixel_box[2] + offset_x),
        int(pixel_box[3] + offset_y),
    ]


def box_iou(box1: list[int], box2: list[int]) -> float:
    """
    计算两个像素框的 IoU。

    这里单独保留一份轻量实现，便于在服务层做跨尺度、跨模板的一致性判断。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def get_strong_score_threshold(
    score_threshold: float,
    strong_score_margin: float = DEFAULT_STRONG_SCORE_MARGIN,
) -> float:
    """
    计算“强分数”阈值。

    说明：
    - 基础阈值只说明“勉强过线”
    - 强分数阈值表示“明显高于过线水平”，可用于放松一部分一致性约束
    """
    return float(score_threshold + strong_score_margin)


def summarize_scale_candidates(
    raw_candidates: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    """
    提取某个 scale 下的核心候选摘要。

    说明：
    - `raw_candidates` 是该尺度下的全部滑窗候选
    - 这里复用 NMS 逻辑拿到低重叠 top2，便于衡量该尺度内部的区分度
    """
    if not raw_candidates:
        raise ValueError("raw_candidates 不能为空")

    scale_top_candidates = select_top_candidates(
        candidates=raw_candidates,
        topk=2,
        iou_threshold=iou_threshold,
    )
    best_candidate = scale_top_candidates[0]
    top1_score = float(best_candidate["score"])
    top2_score = (
        float(scale_top_candidates[1]["score"])
        if len(scale_top_candidates) >= 2
        else None
    )
    margin = None if top2_score is None else float(top1_score - top2_score)

    return {
        "scale": float(best_candidate["scale"]),
        "score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
        "pixel_box": list(best_candidate["pixel_box"]),
    }


def build_joint_confidence_for_single_query(
    *,
    base_confidence: dict[str, Any],
    best_match: dict[str, Any],
    all_candidates: list[dict[str, Any]],
    scale_best_matches: list[dict[str, Any]],
    best_score_z_threshold: float = DEFAULT_BEST_SCORE_Z_THRESHOLD,
    scale_consistency_iou_threshold: float = DEFAULT_SCALE_CONSISTENCY_IOU_THRESHOLD,
    scale_score_tolerance: float = DEFAULT_SCALE_SCORE_TOLERANCE,
    strong_score_margin: float = DEFAULT_STRONG_SCORE_MARGIN,
) -> dict[str, Any]:
    """
    基于“分数 + 分差 + 多尺度一致性 + 分数突出程度”构建单图联合置信度。

    背景：
    - 单看 `best_score`，容易把“无目标但局部很像”的区域误判成高置信
    - 因此这里增加两个补充条件：
      1. 最佳分是否明显高于整张图候选分布（z-score）
      2. 不同尺度下是否能在相近位置形成支持
    """
    score_values = np.asarray([float(item["score"]) for item in all_candidates], dtype=np.float32)
    score_mean = float(np.mean(score_values))
    score_std = float(np.std(score_values))
    best_score = float(best_match["score"])
    best_score_z = 0.0 if score_std < 1e-8 else float((best_score - score_mean) / score_std)
    is_peak_prominent = bool(best_score_z >= best_score_z_threshold)
    strong_score_threshold = get_strong_score_threshold(
        score_threshold=float(base_confidence["score_threshold"]),
        strong_score_margin=strong_score_margin,
    )
    is_strong_score = bool(best_score >= strong_score_threshold)

    scale_support_matches: list[dict[str, Any]] = []
    for scale_match in scale_best_matches:
        overlap = box_iou(best_match["pixel_box"], scale_match["pixel_box"])
        score_gap = float(best_score - scale_match["score"])
        if overlap >= scale_consistency_iou_threshold and score_gap <= scale_score_tolerance:
            scale_support_matches.append(
                {
                    "scale": float(scale_match["scale"]),
                    "score": float(scale_match["score"]),
                    "score_gap_to_best": score_gap,
                    "iou_with_best": float(overlap),
                    "pixel_box": list(scale_match["pixel_box"]),
                }
            )

    required_scale_support = 1 if len(scale_best_matches) <= 1 else min(2, len(scale_best_matches))
    scale_support_count = len(scale_support_matches)
    is_scale_consistent = bool(scale_support_count >= required_scale_support)
    scale_override_by_strong_score = bool((not is_scale_consistent) and is_strong_score)

    final_is_confident = bool(
        base_confidence["is_match_confident"]
        and is_peak_prominent
        and (is_scale_consistent or scale_override_by_strong_score)
    )

    reason_parts = []
    if not base_confidence["is_match_confident"]:
        reason_parts.append(f"base:{base_confidence['reason']}")
    if not is_peak_prominent:
        reason_parts.append("best_score_z_below_threshold")
    if not is_scale_consistent and not scale_override_by_strong_score:
        reason_parts.append("cross_scale_inconsistent")
    if not reason_parts:
        reason_parts.append("ok")

    merged_confidence = dict(base_confidence)
    merged_confidence.update({
        "is_match_confident": final_is_confident,
        "base_is_match_confident": bool(base_confidence["is_match_confident"]),
        "best_score_mean": score_mean,
        "best_score_std": score_std,
        "best_score_z": best_score_z,
        "best_score_z_threshold": float(best_score_z_threshold),
        "is_peak_prominent": is_peak_prominent,
        "strong_score_margin": float(strong_score_margin),
        "strong_score_threshold": float(strong_score_threshold),
        "is_strong_score": is_strong_score,
        "scale_support_count": int(scale_support_count),
        "required_scale_support": int(required_scale_support),
        "scale_consistency_iou_threshold": float(scale_consistency_iou_threshold),
        "scale_score_tolerance": float(scale_score_tolerance),
        "is_scale_consistent": is_scale_consistent,
        "scale_override_by_strong_score": scale_override_by_strong_score,
        "scale_support_matches": scale_support_matches,
        "scale_best_matches": scale_best_matches,
        "reason": ",".join(reason_parts),
    })
    return merged_confidence


def apply_query_consensus_to_result(
    result: dict[str, Any],
    ranked_results: list[dict[str, Any]],
    query_consistency_iou_threshold: float = DEFAULT_QUERY_CONSISTENCY_IOU_THRESHOLD,
    query_score_tolerance: float = DEFAULT_QUERY_SCORE_TOLERANCE,
    low_score_consensus_threshold: float = DEFAULT_LOW_SCORE_CONSENSUS_THRESHOLD,
    low_score_consensus_max_threshold: float = DEFAULT_LOW_SCORE_CONSENSUS_MAX_THRESHOLD,
    low_score_consensus_best_score_z_threshold: float = DEFAULT_LOW_SCORE_CONSENSUS_BEST_SCORE_Z_THRESHOLD,
    strong_score_margin: float = DEFAULT_STRONG_SCORE_MARGIN,
) -> dict[str, Any]:
    """
    在多模板模式下追加“跨模板共识”判断。

    目标：
    - 压低“只有某一张模板误报很高、其他模板并不支持”的情况
    - 保留“多张模板都指向同一片区域”的真目标结果
    """
    if not ranked_results:
        raise ValueError("ranked_results 不能为空")

    best_match = result["best_match"]
    best_score = float(best_match["score"])
    best_box = list(best_match["pixel_box"])

    query_support_matches: list[dict[str, Any]] = []
    for item in ranked_results:
        candidate_best_match = item["best_match"]
        overlap = box_iou(best_box, candidate_best_match["pixel_box"])
        score_gap = float(best_score - float(candidate_best_match["score"]))
        if overlap >= query_consistency_iou_threshold and score_gap <= query_score_tolerance:
            query_support_matches.append(
                {
                    "query_image_name": item["query_image_name"],
                    "score": float(candidate_best_match["score"]),
                    "score_gap_to_best": score_gap,
                    "iou_with_best": float(overlap),
                    "pixel_box": list(candidate_best_match["pixel_box"]),
                }
            )

    required_query_support = 1 if len(ranked_results) <= 1 else min(2, len(ranked_results))
    query_support_count = len(query_support_matches)
    is_query_consistent = bool(query_support_count >= required_query_support)
    selected_cluster_summary = next(
        (
            item
            for item in result.get("cluster_summary", []) or []
            if item.get("is_selected")
        ),
        None,
    )
    selected_cluster_support_count = (
        int(selected_cluster_summary.get("support_count", 0))
        if selected_cluster_summary is not None
        else 0
    )

    original_confidence = dict(result["match_confidence"])
    temporal_iou_with_cached_box = original_confidence.get("temporal_iou_with_cached_box")
    has_temporal_reference = bool(
        original_confidence.get("temporal_reference_box") is not None
        or temporal_iou_with_cached_box is not None
    )
    strong_score_threshold = get_strong_score_threshold(
        score_threshold=float(original_confidence["score_threshold"]),
        strong_score_margin=strong_score_margin,
    )
    is_strong_score = bool(best_score >= strong_score_threshold)
    has_extra_query_support = bool(query_support_count > required_query_support)
    query_consensus_override_applied = bool(
        (not original_confidence["is_match_confident"])
        and bool(original_confidence.get("base_is_match_confident"))
        and bool(original_confidence.get("is_peak_prominent"))
        and is_query_consistent
        and (is_strong_score or has_extra_query_support)
    )
    low_score_consensus_override_applied = bool(
        (not original_confidence["is_match_confident"])
        and has_temporal_reference
        and bool(original_confidence.get("is_scale_consistent"))
        and is_query_consistent
        and selected_cluster_support_count >= required_query_support
        and _safe_float(original_confidence.get("best_score_z"), default=-1.0)
        >= float(low_score_consensus_best_score_z_threshold)
        and best_score >= float(low_score_consensus_threshold)
        and best_score <= float(low_score_consensus_max_threshold)
    )
    final_is_confident = bool(
        (original_confidence["is_match_confident"] and is_query_consistent)
        or query_consensus_override_applied
        or low_score_consensus_override_applied
    )

    reason_parts = []
    if (
        original_confidence["reason"] != "ok"
        and not query_consensus_override_applied
        and not low_score_consensus_override_applied
    ):
        reason_parts.append(original_confidence["reason"])
    if not is_query_consistent:
        reason_parts.append("cross_query_inconsistent")
    if not reason_parts:
        reason_parts.append("ok")

    original_confidence.update({
        "is_match_confident": final_is_confident,
        "base_is_match_confident_before_query_consistency": bool(result["match_confidence"]["is_match_confident"]),
        "query_support_count": int(query_support_count),
        "required_query_support": int(required_query_support),
        "query_consistency_iou_threshold": float(query_consistency_iou_threshold),
        "query_score_tolerance": float(query_score_tolerance),
        "is_query_consistent": is_query_consistent,
        "selected_cluster_support_count": int(selected_cluster_support_count),
        "has_temporal_reference": has_temporal_reference,
        "has_extra_query_support": has_extra_query_support,
        "query_consensus_override_applied": query_consensus_override_applied,
        "low_score_consensus_threshold": float(low_score_consensus_threshold),
        "low_score_consensus_max_threshold": float(low_score_consensus_max_threshold),
        "low_score_consensus_best_score_z_threshold": float(low_score_consensus_best_score_z_threshold),
        "low_score_consensus_override_applied": low_score_consensus_override_applied,
        "strong_score_threshold": float(strong_score_threshold),
        "strong_score_margin": float(strong_score_margin),
        "query_support_matches": query_support_matches,
        "reason": ",".join(reason_parts),
    })
    result["match_confidence"] = original_confidence
    return result


def apply_temporal_consistency_to_result(
    result: dict[str, Any],
    reference_box: list[int],
    temporal_consistency_iou_threshold: float = DEFAULT_TEMPORAL_CONSISTENCY_IOU_THRESHOLD,
) -> dict[str, Any]:
    """
    把“与上一帧高置信框的位置连续性”纳入最终判定。

    设计原则：
    - 只在基础分数已过线、峰值突出、且当前结果主要卡在一致性规则时做放宽
    - 不用它去拯救明显低分的结果，避免把无目标帧重新放大成误报
    """
    best_match = result.get("best_match")
    if best_match is None or best_match.get("pixel_box") is None:
        return result

    best_box = list(best_match["pixel_box"])
    temporal_iou = box_iou(best_box, reference_box)
    original_confidence = dict(result["match_confidence"])
    is_temporally_consistent = bool(temporal_iou >= temporal_consistency_iou_threshold)
    temporal_consistency_override_applied = bool(
        (not original_confidence["is_match_confident"])
        and bool(original_confidence.get("base_is_match_confident"))
        and bool(original_confidence.get("is_peak_prominent"))
        and bool(original_confidence.get("is_query_consistent", True))
        and is_temporally_consistent
    )
    final_is_confident = bool(
        original_confidence["is_match_confident"] or temporal_consistency_override_applied
    )

    reason_parts = []
    if original_confidence["reason"] != "ok" and not temporal_consistency_override_applied:
        reason_parts.append(original_confidence["reason"])
    if not reason_parts:
        reason_parts.append("ok")

    original_confidence.update({
        "is_match_confident": final_is_confident,
        "temporal_reference_box": [int(x) for x in reference_box],
        "temporal_iou_with_cached_box": float(temporal_iou),
        "temporal_consistency_iou_threshold": float(temporal_consistency_iou_threshold),
        "is_temporally_consistent": is_temporally_consistent,
        "temporal_consistency_override_applied": temporal_consistency_override_applied,
        "reason": ",".join(reason_parts),
    })
    result["match_confidence"] = original_confidence
    return result


def evaluate_roi_direct_acceptance(
    result: dict[str, Any],
    roi_direct_score_margin: float = DEFAULT_ROI_DIRECT_SCORE_MARGIN,
    roi_direct_temporal_iou_threshold: float = DEFAULT_ROI_DIRECT_TEMPORAL_IOU_THRESHOLD,
) -> dict[str, Any]:
    """
    判断 ROI 结果是否可以直接作为最终结果返回。

    设计目的：
    - ROI 搜索空间更小，局部背景更容易形成“勉强过线”的假阳性
    - 因此 ROI 即使已经满足基础置信规则，也要再过一层更严格的门
    - 如果这层不过，则回退全图搜索；这样主要影响速度，不会直接损伤召回
    """
    confidence_info = dict(result["match_confidence"])
    top1_score = confidence_info.get("top1_score")
    score_threshold = float(confidence_info.get("score_threshold", 0.0))
    temporal_iou = confidence_info.get("temporal_iou_with_cached_box")

    roi_direct_score_threshold = get_strong_score_threshold(
        score_threshold=score_threshold,
        strong_score_margin=roi_direct_score_margin,
    )
    passes_roi_direct_score = bool(
        top1_score is not None and float(top1_score) >= roi_direct_score_threshold
    )
    passes_roi_direct_temporal = bool(
        temporal_iou is not None
        and float(temporal_iou) >= roi_direct_temporal_iou_threshold
    )
    is_roi_directly_acceptable = bool(
        confidence_info["is_match_confident"]
        and (passes_roi_direct_score or passes_roi_direct_temporal)
    )

    confidence_info.update({
        "roi_direct_score_margin": float(roi_direct_score_margin),
        "roi_direct_score_threshold": float(roi_direct_score_threshold),
        "passes_roi_direct_score": passes_roi_direct_score,
        "roi_direct_temporal_iou_threshold": float(roi_direct_temporal_iou_threshold),
        "passes_roi_direct_temporal": passes_roi_direct_temporal,
        "is_roi_directly_acceptable": is_roi_directly_acceptable,
    })
    result["match_confidence"] = confidence_info
    return result


def apply_roi_offset_to_result(
    result: dict[str, Any],
    offset_x: int,
    offset_y: int,
) -> dict[str, Any]:
    """
    将 ROI 内的匹配结果统一映射回原图坐标。

    说明：
    - 之前只偏移了 `best_match / top_matches / confidence_candidates`
    - 多图模式下还有 `ranking_summary` 等结构也会携带局部坐标
    - 这里统一收口，避免某些字段仍然残留 ROI 局部坐标
    """
    # 某些结构之间会共享同一个候选对象（例如 best_match 可能同时出现在
    # top_matches / confidence_candidates 中）。如果直接逐处偏移，同一框会被
    # 重复加偏移，出现 1292 被加两三次这类明显错误。
    #
    # 这里按对象 id 去重，确保同一个候选对象只偏移一次。
    offset_applied_object_ids: set[int] = set()

    def offset_match_item_once(match_item: dict[str, Any] | None) -> None:
        if match_item is None or match_item.get("pixel_box") is None:
            return

        object_id = id(match_item)
        if object_id in offset_applied_object_ids:
            return

        match_item["pixel_box"] = offset_pixel_box(
            pixel_box=match_item["pixel_box"],
            offset_x=offset_x,
            offset_y=offset_y,
        )
        offset_applied_object_ids.add(object_id)

    offset_match_item_once(result.get("best_match"))

    for match_item in result.get("top_matches", []):
        offset_match_item_once(match_item)

    for match_item in result.get("confidence_candidates", []):
        offset_match_item_once(match_item)

    # ranking_summary 里的每一项通常是新字典，但 pixel_box 需要同样回写到全图坐标。
    for ranking_item in result.get("ranking_summary", []) or []:
        if ranking_item.get("pixel_box") is not None:
            ranking_item["pixel_box"] = offset_pixel_box(
                pixel_box=ranking_item["pixel_box"],
                offset_x=offset_x,
                offset_y=offset_y,
            )

    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    将任意值安全转成 float，避免排序或日志阶段被 None 打断。
    """
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def split_match_confidence_info(
    confidence_info: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    将完整置信信息拆成“核心字段”和“调试字段”两部分。

    设计目的：
    - `match_confidence` 面向业务调用方，只保留最关键的判断结论
    - `match_confidence_debug` 面向排障与调参，保留全部细节
    """
    if not confidence_info:
        return {}, {}

    core_confidence: dict[str, Any] = {}
    debug_confidence: dict[str, Any] = {}

    for key, value in confidence_info.items():
        if key in CORE_MATCH_CONFIDENCE_KEYS:
            core_confidence[key] = value
        else:
            debug_confidence[key] = value

    return core_confidence, debug_confidence


def get_query_result_priority(item: dict[str, Any]) -> tuple[float, ...]:
    """
    生成单个 query 结果的优先级排序键。

    排序思想：
    - 先优先已经通过本模板置信判断的结果
    - 再优先峰值更突出的结果（best_score_z）
    - 再参考原始 best_score 和 top1-top2 分差

    这样做的目的，是减少“某张模板天然分高，但并不是更可靠”的影响。
    """
    confidence_info = item.get("match_confidence", {})
    return (
        float(bool(confidence_info.get("is_match_confident", False))),
        float(bool(confidence_info.get("is_peak_prominent", False))),
        _safe_float(confidence_info.get("best_score_z"), default=-1.0),
        _safe_float(item.get("best_match", {}).get("score"), default=-1.0),
        _safe_float(confidence_info.get("top1_top2_margin"), default=-1.0),
    )


def select_best_result_by_query_box_consensus(
    query_results: list[dict[str, Any]],
    cluster_iou_threshold: float = DEFAULT_QUERY_BOX_CLUSTER_IOU_THRESHOLD,
) -> dict[str, Any]:
    """
    在多模板场景下，先按框位置聚类，再从支持最强的框簇中选择最终赢家。

    设计动机：
    - 之前直接按 raw score 比大小，容易出现“框错了但分更高”的模板压过正确框
    - 这里改成“先看有多少模板支持同一块区域，再在该区域内选最强模板”
    - 对只有 3 张模板的小规模场景，这种方式更贴近实际业务目标
    """
    if not query_results:
        raise ValueError("query_results 不能为空")

    if len(query_results) == 1:
        only_result = query_results[0]
        return {
            "best_result": only_result,
            "ranked_results": [only_result],
            "cluster_summary": [
                {
                    "cluster_id": 1,
                    "support_count": 1,
                    "confident_count": int(bool(only_result["match_confidence"]["is_match_confident"])),
                    "max_best_score_z": _safe_float(only_result["match_confidence"].get("best_score_z"), default=-1.0),
                    "max_best_score": _safe_float(only_result["best_match"].get("score"), default=-1.0),
                    "query_image_names": [only_result["query_image_name"]],
                    "representative_box": list(only_result["best_match"]["pixel_box"]),
                    "is_selected": True,
                }
            ],
        }

    sorted_query_results = sorted(
        query_results,
        key=get_query_result_priority,
        reverse=True,
    )

    clusters: list[dict[str, Any]] = []
    for item in sorted_query_results:
        current_box = list(item["best_match"]["pixel_box"])
        matched_cluster = None
        matched_overlap = -1.0

        for cluster in clusters:
            cluster_overlap = max(
                box_iou(current_box, list(member["best_match"]["pixel_box"]))
                for member in cluster["members"]
            )
            if cluster_overlap >= cluster_iou_threshold and cluster_overlap > matched_overlap:
                matched_cluster = cluster
                matched_overlap = cluster_overlap

        if matched_cluster is None:
            clusters.append({"members": [item]})
        else:
            matched_cluster["members"].append(item)

    cluster_summary: list[dict[str, Any]] = []
    member_cluster_index_map: dict[int, int] = {}

    for cluster_index, cluster in enumerate(clusters, start=1):
        members = cluster["members"]
        sorted_members = sorted(
            members,
            key=get_query_result_priority,
            reverse=True,
        )
        best_member = sorted_members[0]
        support_count = len(members)
        confident_count = sum(
            int(bool(member["match_confidence"].get("is_match_confident", False)))
            for member in members
        )
        max_best_score_z = max(
            _safe_float(member["match_confidence"].get("best_score_z"), default=-1.0)
            for member in members
        )
        sum_best_score_z = sum(
            _safe_float(member["match_confidence"].get("best_score_z"), default=0.0)
            for member in members
        )
        max_best_score = max(
            _safe_float(member["best_match"].get("score"), default=-1.0)
            for member in members
        )
        sum_best_score = sum(
            _safe_float(member["best_match"].get("score"), default=0.0)
            for member in members
        )

        cluster_sort_key = (
            float(support_count),
            float(confident_count),
            max_best_score_z,
            sum_best_score_z,
            max_best_score,
            sum_best_score,
        )

        cluster_summary.append({
            "cluster_id": int(cluster_index),
            "support_count": int(support_count),
            "confident_count": int(confident_count),
            "max_best_score_z": float(max_best_score_z),
            "sum_best_score_z": float(sum_best_score_z),
            "max_best_score": float(max_best_score),
            "sum_best_score": float(sum_best_score),
            "query_image_names": [member["query_image_name"] for member in sorted_members],
            "representative_box": list(best_member["best_match"]["pixel_box"]),
            "best_member": best_member,
            "sorted_members": sorted_members,
            "cluster_sort_key": cluster_sort_key,
        })

        for member in members:
            member_cluster_index_map[id(member)] = int(cluster_index)

    selected_cluster = max(
        cluster_summary,
        key=lambda item: item["cluster_sort_key"],
    )
    selected_cluster_id = int(selected_cluster["cluster_id"])
    selected_members = list(selected_cluster["sorted_members"])
    unselected_members = [
        item
        for item in sorted_query_results
        if member_cluster_index_map[id(item)] != selected_cluster_id
    ]
    unselected_members = sorted(
        unselected_members,
        key=lambda item: (
            float(cluster_summary[member_cluster_index_map[id(item)] - 1]["support_count"]),
            *get_query_result_priority(item),
        ),
        reverse=True,
    )

    ranked_results = selected_members + unselected_members
    best_result = selected_members[0]

    normalized_cluster_summary = []
    for item in sorted(
        cluster_summary,
        key=lambda cluster_item: cluster_item["cluster_sort_key"],
        reverse=True,
    ):
        normalized_cluster_summary.append({
            "cluster_id": int(item["cluster_id"]),
            "support_count": int(item["support_count"]),
            "confident_count": int(item["confident_count"]),
            "max_best_score_z": float(item["max_best_score_z"]),
            "sum_best_score_z": float(item["sum_best_score_z"]),
            "max_best_score": float(item["max_best_score"]),
            "sum_best_score": float(item["sum_best_score"]),
            "query_image_names": list(item["query_image_names"]),
            "representative_box": list(item["representative_box"]),
            "is_selected": bool(item["cluster_id"] == selected_cluster_id),
        })

    return {
        "best_result": best_result,
        "ranked_results": ranked_results,
        "cluster_summary": normalized_cluster_summary,
    }


def get_cached_region(small_image_path: str, frame_size: tuple[int, int]) -> dict[str, Any] | None:
    """
    获取某个 query 图对应的历史高置信命中区域。

    只有当当前帧尺寸与缓存帧尺寸一致时才复用，避免跨尺寸误用。
    """
    with MATCH_REGION_CACHE_LOCK:
        cached = MATCH_REGION_CACHE.get(small_image_path)

    if cached is None:
        return None

    cached_frame_size = cached.get("frame_size")
    if cached_frame_size != {"width": frame_size[0], "height": frame_size[1]}:
        return None

    return cached


def update_cached_region(
    small_image_path: str,
    frame_size: tuple[int, int],
    pixel_box: list[int],
) -> None:
    """
    更新某个 query 图的历史高置信命中框。
    """
    with MATCH_REGION_CACHE_LOCK:
        MATCH_REGION_CACHE[small_image_path] = {
            "frame_size": {"width": frame_size[0], "height": frame_size[1]},
            "pixel_box": [int(x) for x in pixel_box],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }


@lru_cache(maxsize=1)
def get_feature_extractor():
    """
    缓存 DINO 提取器，避免重复加载模型。
    """
    logger.info("开始初始化 DINO 匹配服务提取器")
    extractor = build_default_extractor()
    logger.info("DINO 匹配服务提取器初始化完成")
    return extractor


def find_similar_region_with_precomputed_big_info(
    extractor,
    frame_image: Image.Image,
    query_image: Image.Image | None,
    big_info: dict[str, Any],
    topk: int = 5,
    iou_threshold: float = 0.6,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    confidence_iou_threshold: float = 0.3,
    small_scale_infos: list[dict[str, Any]] | None = None,
    query_image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    在大图 token 已预计算的前提下，对单张小图执行多尺度匹配。

    这样在“一个大图对多张小图”的场景中，可以避免重复提取大图 token。
    """
    total_start_time = perf_counter()

    if scales is None:
        scales = [0.8, 1.0, 1.2]

    if query_image is None and small_scale_infos is None:
        raise ValueError("query_image 和 small_scale_infos 不能同时为空")

    if query_image is not None:
        resolved_query_image_size = (
            int(query_image.size[0]),
            int(query_image.size[1]),
        )
    elif query_image_size is not None:
        resolved_query_image_size = (
            int(query_image_size[0]),
            int(query_image_size[1]),
        )
    else:
        raise ValueError("使用 small_scale_infos 时必须提供 query_image_size")

    original_big_size = frame_image.size
    padded_big_size = (
        big_info["padded_size"]["width"],
        big_info["padded_size"]["height"],
    )
    precomputed_small_scale_info_map = {}
    if small_scale_infos is not None:
        precomputed_small_scale_info_map = {
            format_scale_cache_key(item["scale"]): item
            for item in small_scale_infos
        }

    all_candidates: list[dict[str, Any]] = []
    scale_timing_details: list[dict[str, Any]] = []
    scale_best_matches: list[dict[str, Any]] = []

    for scale in scales:
        scale_start_time = perf_counter()
        precomputed_scale_info = precomputed_small_scale_info_map.get(
            format_scale_cache_key(scale)
        )

        if precomputed_scale_info is not None:
            small_info = precomputed_scale_info["small_info"]
            scaled_small_image_size = dict(precomputed_scale_info["scaled_small_image_size"])
            # 命中预计算信息后，这里不再重复提 token，因此当前阶段耗时记为 0。
            # 真正的小图准备耗时会在 query 级别的 `prepare_small_image_elapsed_sec` 中体现。
            small_token_elapsed = 0.0
            small_token_cache_hit = bool(
                precomputed_scale_info.get("small_token_cache_hit", False)
            )
        else:
            if query_image is None:
                raise ValueError("缺少 query_image，无法在未命中预计算信息时继续处理")
            scaled_query_image = resize_small_image_with_scale(query_image, scale=scale)
            scaled_small_image_size = {
                "width": int(scaled_query_image.size[0]),
                "height": int(scaled_query_image.size[1]),
            }
            small_token_start_time = perf_counter()
            small_info = extract_patch_tokens(extractor, scaled_query_image)
            small_token_elapsed = perf_counter() - small_token_start_time
            small_token_cache_hit = False

        if (
            small_info["grid_size"]["height"] > big_info["grid_size"]["height"]
            or small_info["grid_size"]["width"] > big_info["grid_size"]["width"]
        ):
            logger.warning(
                "跳过尺度 %.3f：小图 token 网格大于帧图 token 网格，small=%s, big=%s",
                scale,
                small_info["grid_size"],
                big_info["grid_size"],
            )
            scale_timing_details.append({
                "scale": float(scale),
                "small_token_elapsed_sec": float(small_token_elapsed),
                "match_elapsed_sec": None,
                "total_elapsed_sec": float(perf_counter() - scale_start_time),
                "skipped": True,
                "candidate_count": 0,
                "small_token_cache_hit": small_token_cache_hit,
            })
            continue

        match_start_time = perf_counter()
        raw_candidates, _ = compute_similarity_map(
            big_tokens=big_info["tokens"],
            query_tokens=small_info["tokens"],
            topk_ratio=topk_ratio,
        )
        match_elapsed = perf_counter() - match_start_time

        # 这里先把当前 scale 的原始候选补全 pixel_box，再继续做后续摘要和总候选合并。
        # 原始 compute_similarity_map 返回的结构里只有 score/token_box，没有 pixel_box，
        # 直接拿去做 select_top_candidates 会触发 KeyError。
        scale_candidates: list[dict[str, Any]] = []
        for item in raw_candidates:
            pixel_box = token_box_to_pixel_box(
                token_box=item["token_box"],
                patch_size=big_info["patch_size"],
                padded_big_size=padded_big_size,
                original_big_size=original_big_size,
            )
            scale_candidates.append({
                "score": item["score"],
                "token_box": item["token_box"],
                "pixel_box": pixel_box,
                "scale": scale,
                "scaled_small_image_size": dict(scaled_small_image_size),
                "small_token_grid": small_info["grid_size"],
            })

        scale_best_matches.append(
            summarize_scale_candidates(
                raw_candidates=scale_candidates,
                iou_threshold=confidence_iou_threshold,
            )
        )
        all_candidates.extend(scale_candidates)

        scale_total_elapsed = perf_counter() - scale_start_time
        scale_timing_details.append({
            "scale": float(scale),
            "small_token_elapsed_sec": float(small_token_elapsed),
            "match_elapsed_sec": float(match_elapsed),
            "total_elapsed_sec": float(scale_total_elapsed),
            "skipped": False,
            "candidate_count": len(raw_candidates),
            "small_token_cache_hit": small_token_cache_hit,
        })
        logger.info(
            "阶段耗时 | scale=%.3f | small_size=(%s,%s) | small_grid=(%s,%s) | token=%.3fs | cache_hit=%s | match=%.3fs | total=%.3fs | candidates=%s",
            scale,
            scaled_small_image_size["width"],
            scaled_small_image_size["height"],
            small_info["grid_size"]["height"],
            small_info["grid_size"]["width"],
            small_token_elapsed,
            small_token_cache_hit,
            match_elapsed,
            scale_total_elapsed,
            len(scale_candidates),
        )

    if not all_candidates:
        raise ValueError("所有尺度都无法生成有效候选，请检查帧图和小图尺寸。")

    postprocess_start_time = perf_counter()
    top_candidates = select_top_candidates(
        candidates=all_candidates,
        topk=topk,
        iou_threshold=iou_threshold,
    )
    confidence_candidates = select_top_candidates(
        candidates=all_candidates,
        topk=max(topk, 2),
        iou_threshold=confidence_iou_threshold,
    )
    best_match = top_candidates[0]
    confidence_info = judge_match_confidence(
        ranked_candidates=confidence_candidates,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
    )
    confidence_info = build_joint_confidence_for_single_query(
        base_confidence=confidence_info,
        best_match=best_match,
        all_candidates=all_candidates,
        scale_best_matches=scale_best_matches,
    )
    postprocess_elapsed = perf_counter() - postprocess_start_time
    total_elapsed = perf_counter() - total_start_time

    logger.info(
        "阶段耗时 | 候选后处理完成 | all_candidates=%s | top_candidates=%s | confidence_candidates=%s | postprocess=%.3fs | total=%.3fs",
        len(all_candidates),
        len(top_candidates),
        len(confidence_candidates),
        postprocess_elapsed,
        total_elapsed,
    )

    return {
        "frame_size": {
            "width": original_big_size[0],
            "height": original_big_size[1],
        },
        "small_image_size": {
            "width": resolved_query_image_size[0],
            "height": resolved_query_image_size[1],
        },
        "patch_size": big_info["patch_size"],
        "frame_token_grid": big_info["grid_size"],
        "searched_scales": scales,
        "topk_ratio": float(topk_ratio),
        "best_match": best_match,
        "top_matches": top_candidates,
        "confidence_candidates": confidence_candidates,
        "match_confidence": confidence_info,
        "timing": {
            "postprocess_elapsed_sec": float(postprocess_elapsed),
            "total_elapsed_sec": float(total_elapsed),
            "scale_details": scale_timing_details,
            "scale_best_matches": scale_best_matches,
        },
    }


def find_similar_region_in_memory(
    extractor,
    frame_image: Image.Image,
    query_image: Image.Image | None = None,
    query_image_path: str | None = None,
    topk: int = 5,
    iou_threshold: float = 0.6,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    confidence_iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    在内存中的视频帧里查找小图最相似的位置。

    这里复用单图 token 匹配逻辑，但避免把视频帧先写入磁盘。
    如果传入 `query_image_path`，则会优先复用小图 token 缓存。
    """
    if bool(query_image is not None) == bool(query_image_path is not None):
        raise ValueError("必须且只能提供一个参数：query_image 或 query_image_path")

    big_token_start_time = perf_counter()
    big_info = extract_patch_tokens(extractor, frame_image)
    big_token_elapsed = perf_counter() - big_token_start_time
    original_big_size = frame_image.size

    logger.info(
        "阶段耗时 | 大图 token 提取完成 | frame_size=(%s,%s) | grid=(%s,%s) | elapsed=%.3fs",
        original_big_size[0],
        original_big_size[1],
        big_info["grid_size"]["height"],
        big_info["grid_size"]["width"],
        big_token_elapsed,
    )

    if query_image_path is not None:
        if scales is None:
            scales = [0.8, 1.0, 1.2]
        prepared_small_info = prepare_small_image_scale_infos(
            extractor=extractor,
            query_image_path=query_image_path,
            scales=scales,
        )
        result = find_similar_region_with_precomputed_big_info(
            extractor=extractor,
            frame_image=frame_image,
            query_image=None,
            big_info=big_info,
            topk=topk,
            iou_threshold=iou_threshold,
            scales=scales,
            topk_ratio=topk_ratio,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            confidence_iou_threshold=confidence_iou_threshold,
            small_scale_infos=prepared_small_info["scale_infos"],
            query_image_size=prepared_small_info["query_image_size"],
        )
        result["timing"]["small_token_cache_hit_count"] = int(
            prepared_small_info["small_token_cache_hit_count"]
        )
        result["timing"]["small_token_cache_miss_count"] = int(
            prepared_small_info["small_token_cache_miss_count"]
        )
        result["timing"]["prepare_small_image_elapsed_sec"] = float(
            prepared_small_info["total_prepare_elapsed_sec"]
        )
    else:
        result = find_similar_region_with_precomputed_big_info(
            extractor=extractor,
            frame_image=frame_image,
            query_image=query_image,
            big_info=big_info,
            topk=topk,
            iou_threshold=iou_threshold,
            scales=scales,
            topk_ratio=topk_ratio,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            confidence_iou_threshold=confidence_iou_threshold,
        )

    result["timing"]["big_token_elapsed_sec"] = float(big_token_elapsed)
    result["timing"]["total_elapsed_sec"] = float(
        big_token_elapsed + result["timing"]["total_elapsed_sec"]
    )
    return result


def find_best_similar_region_in_memory(
    extractor,
    frame_image: Image.Image,
    query_image_paths: list[str],
    topk: int = 5,
    iou_threshold: float = 0.6,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    confidence_iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    在一张大图中同时对比多张小图，并返回分数最高的结果。

    关键优化：
    - 大图 token 只提取一次
    - 目录里的多张 query 图共享同一份大图特征
    """
    if not query_image_paths:
        raise ValueError("query_image_paths 不能为空")

    total_start_time = perf_counter()

    big_token_start_time = perf_counter()
    big_info = extract_patch_tokens(extractor, frame_image)
    big_token_elapsed = perf_counter() - big_token_start_time
    original_big_size = frame_image.size

    logger.info(
        "阶段耗时 | 大图 token 提取完成 | frame_size=(%s,%s) | grid=(%s,%s) | elapsed=%.3fs",
        original_big_size[0],
        original_big_size[1],
        big_info["grid_size"]["height"],
        big_info["grid_size"]["width"],
        big_token_elapsed,
    )

    query_results: list[dict[str, Any]] = []
    query_timing_details: list[dict[str, Any]] = []

    for index, query_image_path in enumerate(query_image_paths, start=1):
        query_start_time = perf_counter()
        query_name = Path(query_image_path).name
        logger.info(
            "开始处理 query 图 %s / %s: %s",
            index,
            len(query_image_paths),
            query_image_path,
        )

        try:
            prepare_query_start_time = perf_counter()
            prepared_small_info = prepare_small_image_scale_infos(
                extractor=extractor,
                query_image_path=query_image_path,
                scales=scales if scales is not None else [0.8, 1.0, 1.2],
            )
            load_query_elapsed = perf_counter() - prepare_query_start_time

            result = find_similar_region_with_precomputed_big_info(
                extractor=extractor,
                frame_image=frame_image,
                query_image=None,
                big_info=big_info,
                topk=topk,
                iou_threshold=iou_threshold,
                scales=scales,
                topk_ratio=topk_ratio,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
                confidence_iou_threshold=confidence_iou_threshold,
                small_scale_infos=prepared_small_info["scale_infos"],
                query_image_size=prepared_small_info["query_image_size"],
            )

            result["small_image_path"] = str(Path(query_image_path).resolve())
            result["query_image_name"] = query_name
            result["timing"]["load_query_image_elapsed_sec"] = float(load_query_elapsed)
            result["timing"]["small_token_cache_hit_count"] = int(
                prepared_small_info["small_token_cache_hit_count"]
            )
            result["timing"]["small_token_cache_miss_count"] = int(
                prepared_small_info["small_token_cache_miss_count"]
            )
            result["timing"]["prepare_small_image_elapsed_sec"] = float(
                prepared_small_info["total_prepare_elapsed_sec"]
            )
            result["timing"]["total_elapsed_sec"] = float(
                load_query_elapsed + result["timing"]["total_elapsed_sec"]
            )
            query_results.append(result)

            query_timing_details.append({
                "query_image_name": query_name,
                "small_image_path": result["small_image_path"],
                "best_score": float(result["best_match"]["score"]),
                "is_match_confident": bool(result["match_confidence"]["is_match_confident"]),
                "load_query_image_elapsed_sec": float(load_query_elapsed),
                "match_elapsed_sec": float(result["timing"]["total_elapsed_sec"]),
                "small_token_cache_hit_count": int(prepared_small_info["small_token_cache_hit_count"]),
                "small_token_cache_miss_count": int(prepared_small_info["small_token_cache_miss_count"]),
                "error": None,
            })
            logger.info(
                "query 图处理完成 | query=%s | best_score=%.6f | is_match_confident=%s | prepare_small_image=%.3fs | small_token_cache_hit=%s | small_token_cache_miss=%s | elapsed=%.3fs",
                query_name,
                result["best_match"]["score"],
                result["match_confidence"]["is_match_confident"],
                prepared_small_info["total_prepare_elapsed_sec"],
                prepared_small_info["small_token_cache_hit_count"],
                prepared_small_info["small_token_cache_miss_count"],
                perf_counter() - query_start_time,
            )
        except Exception as exc:
            logger.exception("处理 query 图失败: %s", query_image_path)
            query_timing_details.append({
                "query_image_name": query_name,
                "small_image_path": str(Path(query_image_path).resolve()),
                "best_score": None,
                "is_match_confident": False,
                "load_query_image_elapsed_sec": None,
                "match_elapsed_sec": None,
                "error": str(exc),
            })

    if not query_results:
        raise ValueError("目录中的小图均处理失败，无法生成有效匹配结果。")

    selection_result = select_best_result_by_query_box_consensus(
        query_results=query_results,
    )
    ranked_results = selection_result["ranked_results"]
    best_result = selection_result["best_result"]
    overall_elapsed = perf_counter() - total_start_time

    merged_result = dict(best_result)
    merged_result["query_image_count"] = len(query_image_paths)
    merged_result["selection_strategy"] = "query_box_cluster_consensus"
    merged_result["cluster_summary"] = selection_result["cluster_summary"]
    merged_result["ranking_summary"] = [
        {
            "rank": rank,
            "query_image_name": item["query_image_name"],
            "small_image_path": item["small_image_path"],
            "best_score": float(item["best_match"]["score"]),
            "best_score_z": _safe_float(item["match_confidence"].get("best_score_z"), default=-1.0),
            "is_match_confident": bool(item["match_confidence"]["is_match_confident"]),
            "cluster_id": next(
                (
                    cluster_item["cluster_id"]
                    for cluster_item in selection_result["cluster_summary"]
                    if item["query_image_name"] in cluster_item["query_image_names"]
                ),
                None,
            ),
            "pixel_box": item["best_match"]["pixel_box"],
        }
        for rank, item in enumerate(ranked_results, start=1)
    ]
    merged_result = apply_query_consensus_to_result(
        result=merged_result,
        ranked_results=ranked_results,
    )

    # 统一打印全部小图的成绩，便于直接在日志里查看完整排行榜。
    ranking_log_lines = [
        (
            f"rank={item['rank']}, "
            f"query={item['query_image_name']}, "
            f"score={item['best_score']:.6f}, "
            f"z={item['best_score_z']:.6f}, "
            f"confident={item['is_match_confident']}, "
            f"cluster={item['cluster_id']}, "
            f"box={item['pixel_box']}"
        )
        for item in merged_result["ranking_summary"]
    ]
    cluster_log_lines = [
        (
            f"cluster={item['cluster_id']}, "
            f"selected={item['is_selected']}, "
            f"support={item['support_count']}, "
            f"confident={item['confident_count']}, "
            f"max_z={item['max_best_score_z']:.6f}, "
            f"sum_z={item['sum_best_score_z']:.6f}, "
            f"queries={item['query_image_names']}, "
            f"box={item['representative_box']}"
        )
        for item in merged_result["cluster_summary"]
    ]
    failed_query_lines = [
        (
            f"query={item['query_image_name']}, "
            f"error={item['error']}"
        )
        for item in query_timing_details
        if item["error"] is not None
    ]
    if ranking_log_lines:
        logger.info(
            "全部小图成绩汇总如下:\n%s",
            "\n".join(ranking_log_lines),
        )
    if cluster_log_lines:
        logger.info(
            "多模板框簇汇总如下:\n%s",
            "\n".join(cluster_log_lines),
        )
    logger.info(
        "多模板共识判断 | selection_strategy=%s | selected_query=%s | query_support_count=%s | required_query_support=%s | selected_cluster_support=%s | is_query_consistent=%s | query_consensus_override=%s | low_score_consensus_override=%s | final_is_match_confident=%s",
        merged_result.get("selection_strategy"),
        merged_result.get("query_image_name"),
        merged_result["match_confidence"].get("query_support_count"),
        merged_result["match_confidence"].get("required_query_support"),
        merged_result["match_confidence"].get("selected_cluster_support_count"),
        merged_result["match_confidence"].get("is_query_consistent"),
        merged_result["match_confidence"].get("query_consensus_override_applied"),
        merged_result["match_confidence"].get("low_score_consensus_override_applied"),
        merged_result["match_confidence"].get("is_match_confident"),
    )
    if failed_query_lines:
        logger.warning(
            "以下小图处理失败:\n%s",
            "\n".join(failed_query_lines),
        )

    merged_timing = dict(best_result["timing"])
    merged_timing["big_token_elapsed_sec"] = float(big_token_elapsed)
    merged_timing["total_elapsed_sec"] = float(overall_elapsed)
    merged_timing["query_details"] = query_timing_details
    merged_result["timing"] = merged_timing
    return merged_result


def find_similar_region_from_cv2_frame(
    frame: np.ndarray,
    small_image_path: str,
    topk: int = 5,
    iou_threshold: float = 0.6,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    confidence_iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    直接接收 `cv2.VideoCapture(...).read()` 返回的 frame，并返回最佳匹配框。

    适用场景：
    - 调用方和当前代码在同一个 Python 进程里
    - 不走 HTTP，只想直接传 numpy.ndarray
    """
    logger.info("收到内存帧匹配请求: small_image_path=%s", small_image_path)

    extractor = get_feature_extractor()
    frame_image = convert_cv2_frame_to_rgb_image(frame)

    result = find_similar_region_in_memory(
        extractor=extractor,
        frame_image=frame_image,
        query_image_path=small_image_path,
        topk=topk,
        iou_threshold=iou_threshold,
        scales=scales,
        topk_ratio=topk_ratio,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        confidence_iou_threshold=confidence_iou_threshold,
    )

    best_box = result["best_match"]["pixel_box"]
    core_match_confidence, debug_match_confidence = split_match_confidence_info(
        result.get("match_confidence")
    )
    return {
        "small_image_path": small_image_path,
        "frame_size": result["frame_size"],
        "match_box": {
            "x1": int(best_box[0]),
            "y1": int(best_box[1]),
            "x2": int(best_box[2]),
            "y2": int(best_box[3]),
        },
        "best_score": result["best_match"]["score"],
        "match_confidence": core_match_confidence,
        "match_confidence_debug": debug_match_confidence,
        "timing": result.get("timing"),
    }


def summarize_load_small_image_elapsed(result_timing: dict[str, Any] | None) -> float:
    """
    汇总小图准备耗时，避免顶层 timing 字段与真实含义不一致。

    规则：
    - 多模板模式下，优先累加 `query_details[*].load_query_image_elapsed_sec`
    - 单模板模式下，回退到 `prepare_small_image_elapsed_sec`
    - 都没有时返回 0
    """
    if not result_timing:
        return 0.0

    query_details = result_timing.get("query_details") or []
    if query_details:
        return float(sum(
            float(item.get("load_query_image_elapsed_sec") or 0.0)
            for item in query_details
        ))

    return float(result_timing.get("prepare_small_image_elapsed_sec") or 0.0)


app = FastAPI(
    title="DINO 相似区域匹配服务",
    description="只负责在视频帧或图片中定位与小图最相似的区域，不包含裁框特征提取或 FAISS 检索逻辑。",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/dino/find-similar-region")
async def find_similar_region(
    frame_file: UploadFile = File(..., description="视频帧图片文件，建议由 cv2.imencode 后上传"),
    small_image_path: str | None = Form(None, description="待查找的小图本地路径，与 small_image_dir 二选一"),
    small_image_dir: str | None = Form('./chen', description="待查找的小图目录路径，与 small_image_path 二选一"),
    scales: str = Form("1.0,1.2", description="多尺度列表，逗号分隔"),
    topk: int = Form(5, description="候选区域数量"),
    iou_threshold: float = Form(0.6, description="候选去重使用的 IoU 阈值"),
    topk_ratio: float = Form(0.6, description="每个候选区域保留的 top-k patch 比例"),
    score_threshold: float = Form(0.5, description="最佳候选分数阈值"),
    margin_threshold: float = Form(0.001, description="top1 与 top2 分差阈值"),
    confidence_iou_threshold: float = Form(0.3, description="可信度判断时用于过滤重复框的 IoU 阈值"),
    roi_expand_ratio: float = Form(DEFAULT_ROI_EXPAND_RATIO, description="命中历史高置信框后，ROI 相对目标框宽高的扩展倍数"),
    roi_min_size: int = Form(DEFAULT_ROI_MIN_SIZE, description="ROI 的最小宽高，单位像素"),
    save_annotated_image: str = Form("false", description="是否保存画框后的结果图，支持 true/false"),
    annotated_output_path: str | None = Form(None, description="画框结果图输出路径；未传时使用默认文件名"),
) -> dict[str, Any]:
    """
    在视频帧中查找指定小图的位置。

    必填参数有两类：
    - frame_file: 视频帧图片内容
    - small_image_path / small_image_dir: 单图路径或小图目录，二选一
    """
    logger.info(
        "收到相似区域查找请求: frame_file=%s, small_image_path=%s, small_image_dir=%s, roi_expand_ratio=%.3f, roi_min_size=%s",
        frame_file.filename,
        small_image_path,
        small_image_dir,
        roi_expand_ratio,
        roi_min_size,
    )

    extractor = get_feature_extractor()

    try:
        request_start_time = perf_counter()

        read_frame_start_time = perf_counter()
        frame_bytes = await frame_file.read()
        if not frame_bytes:
            raise ValueError("上传的视频帧内容为空")
        read_frame_elapsed = perf_counter() - read_frame_start_time

        decode_frame_start_time = perf_counter()
        frame_image = load_rgb_image_from_bytes(frame_bytes)
        decode_frame_elapsed = perf_counter() - decode_frame_start_time

        collect_query_start_time = perf_counter()
        query_image_paths = collect_query_image_paths(
            small_image_path=small_image_path,
            small_image_dir=small_image_dir,
        )
        collect_query_elapsed = perf_counter() - collect_query_start_time
        cache_key = (
            str(Path(small_image_path).expanduser().resolve())
            if small_image_path is not None
            else str(Path(small_image_dir).expanduser().resolve())
        )

        logger.info(
            "阶段耗时 | 请求预处理完成 | read_frame=%.3fs | decode_frame=%.3fs | collect_query=%.3fs | query_count=%s",
            read_frame_elapsed,
            decode_frame_elapsed,
            collect_query_elapsed,
            len(query_image_paths),
        )

        parsed_scales = parse_scales(scales)
        frame_size = frame_image.size
        cached_region = get_cached_region(
            small_image_path=cache_key,
            frame_size=frame_size,
        )

        match_start_time = perf_counter()
        search_strategy = "full_frame"
        roi_fallback_triggered = False

        if cached_region is not None:
            roi_search_start_time = perf_counter()
            roi_image, roi_info = crop_roi_around_box(
                frame_image=frame_image,
                center_box=cached_region["pixel_box"],
                roi_expand_ratio=roi_expand_ratio,
                roi_min_size=roi_min_size,
            )
            logger.info(
                "命中历史高置信缓存，先执行 ROI 搜索 | source_box=(%s,%s,%s,%s) | source_box_size=(%s,%s) | expand_ratio=%.3f | roi_min_size=%s | roi=(%s,%s,%s,%s) | roi_size=(%s,%s)",
                cached_region["pixel_box"][0],
                cached_region["pixel_box"][1],
                cached_region["pixel_box"][2],
                cached_region["pixel_box"][3],
                roi_info["source_box_width"],
                roi_info["source_box_height"],
                roi_info["expand_ratio"],
                roi_info["roi_min_size"],
                roi_info["left"],
                roi_info["top"],
                roi_info["right"],
                roi_info["bottom"],
                roi_info["width"],
                roi_info["height"],
            )

            roi_result = find_best_similar_region_in_memory(
                extractor=extractor,
                frame_image=roi_image,
                query_image_paths=query_image_paths,
                topk=topk,
                iou_threshold=iou_threshold,
                scales=parsed_scales,
                topk_ratio=topk_ratio,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
                confidence_iou_threshold=confidence_iou_threshold,
            )

            roi_local_best_box = None
            if roi_result.get("best_match") is not None:
                roi_local_best_box = list(roi_result["best_match"]["pixel_box"])

            roi_result = apply_roi_offset_to_result(
                result=roi_result,
                offset_x=roi_info["left"],
                offset_y=roi_info["top"],
            )

            roi_result["frame_size"] = {"width": frame_size[0], "height": frame_size[1]}
            roi_result = apply_temporal_consistency_to_result(
                result=roi_result,
                reference_box=cached_region["pixel_box"],
            )
            roi_result = evaluate_roi_direct_acceptance(
                result=roi_result,
            )
            roi_elapsed = perf_counter() - roi_search_start_time
            logger.info(
                "ROI 搜索完成 | elapsed=%.3fs | is_match_confident=%s | is_roi_directly_acceptable=%s | top1_score=%s | reason=%s | temporal_iou=%s | query_override=%s | temporal_override=%s | roi_direct_score_threshold=%s | roi_direct_temporal_threshold=%s | local_best_box=%s | global_best_box=%s",
                roi_elapsed,
                roi_result["match_confidence"]["is_match_confident"],
                roi_result["match_confidence"].get("is_roi_directly_acceptable"),
                roi_result["match_confidence"]["top1_score"],
                roi_result["match_confidence"]["reason"],
                roi_result["match_confidence"].get("temporal_iou_with_cached_box"),
                roi_result["match_confidence"].get("query_consensus_override_applied"),
                roi_result["match_confidence"].get("temporal_consistency_override_applied"),
                roi_result["match_confidence"].get("roi_direct_score_threshold"),
                roi_result["match_confidence"].get("roi_direct_temporal_iou_threshold"),
                roi_local_best_box,
                None if roi_result.get("best_match") is None else roi_result["best_match"]["pixel_box"],
            )

            if roi_result["match_confidence"].get("is_roi_directly_acceptable"):
                result = roi_result
                search_strategy = "roi_only"
            else:
                roi_fallback_triggered = True
                logger.info(
                    "ROI 搜索未达到直接放行条件，回退到全图搜索。 | is_match_confident=%s | is_roi_directly_acceptable=%s | top1_score=%s | roi_direct_score_threshold=%s | temporal_iou=%s | roi_direct_temporal_threshold=%s",
                    roi_result["match_confidence"]["is_match_confident"],
                    roi_result["match_confidence"].get("is_roi_directly_acceptable"),
                    roi_result["match_confidence"].get("top1_score"),
                    roi_result["match_confidence"].get("roi_direct_score_threshold"),
                    roi_result["match_confidence"].get("temporal_iou_with_cached_box"),
                    roi_result["match_confidence"].get("roi_direct_temporal_iou_threshold"),
                )
                result = find_best_similar_region_in_memory(
                    extractor=extractor,
                    frame_image=frame_image,
                    query_image_paths=query_image_paths,
                    topk=topk,
                    iou_threshold=iou_threshold,
                    scales=parsed_scales,
                    topk_ratio=topk_ratio,
                    score_threshold=score_threshold,
                    margin_threshold=margin_threshold,
                    confidence_iou_threshold=confidence_iou_threshold,
                )
                result = apply_temporal_consistency_to_result(
                    result=result,
                    reference_box=cached_region["pixel_box"],
                )
                search_strategy = "roi_fallback_full_frame"
        else:
            logger.info("未命中历史高置信缓存，直接执行全图搜索。")
            result = find_best_similar_region_in_memory(
                extractor=extractor,
                frame_image=frame_image,
                query_image_paths=query_image_paths,
                topk=topk,
                iou_threshold=iou_threshold,
                scales=parsed_scales,
                topk_ratio=topk_ratio,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
                confidence_iou_threshold=confidence_iou_threshold,
            )

        match_elapsed = perf_counter() - match_start_time

        best_box = result["best_match"]["pixel_box"]
        should_save_annotated_image = parse_bool_value(save_annotated_image)
        annotated_image_path = None
        annotated_image_name = None
        if should_save_annotated_image:
            save_image_start_time = perf_counter()
            annotated_boxes = result.get("ranking_summary")
            annotated_image_path = save_annotated_frame_image(
                frame_image=frame_image,
                best_box=best_box,
                is_match_confident=result["match_confidence"]["is_match_confident"],
                annotated_boxes=annotated_boxes,
                output_path=annotated_output_path,
            )
            annotated_image_name = Path(annotated_image_path).name
            save_image_elapsed = perf_counter() - save_image_start_time
            logger.info(
                "标注图片文件名: %s | annotated_image_path=%s",
                annotated_image_name,
                annotated_image_path,
            )
        else:
            save_image_elapsed = 0.0

        confidence_info = result["match_confidence"]
        result_timing = result.get("timing")
        load_small_image_elapsed = summarize_load_small_image_elapsed(result_timing)
        core_match_confidence, debug_match_confidence = split_match_confidence_info(
            confidence_info
        )
        top1_score = confidence_info["top1_score"]
        score_gap = None if top1_score is None else float(top1_score - score_threshold)

        if confidence_info["is_match_confident"]:
            update_cached_region(
                small_image_path=cache_key,
                frame_size=frame_size,
                pixel_box=best_box,
            )

        logger.info(
            "匹配结果摘要 | strategy=%s | roi_fallback_triggered=%s | top1_score=%s | score_threshold=%.6f | score_gap=%s | top2_score=%s | margin=%s | margin_threshold=%.6f | best_score_z=%s | scale_support=%s/%s | scale_override=%s | query_support=%s/%s | selected_cluster_support=%s | query_override=%s | low_score_consensus_threshold=%s | low_score_consensus_max_threshold=%s | low_score_consensus_z_threshold=%s | low_score_consensus_override=%s | temporal_iou=%s | temporal_override=%s | roi_direct_ok=%s | roi_direct_score_threshold=%s | roi_direct_temporal_threshold=%s | is_match_confident=%s | reason=%s | annotated_image_name=%s",
            search_strategy,
            roi_fallback_triggered,
            None if top1_score is None else f"{top1_score:.6f}",
            score_threshold,
            None if score_gap is None else f"{score_gap:.6f}",
            None if confidence_info["top2_score"] is None else f"{confidence_info['top2_score']:.6f}",
            None if confidence_info["top1_top2_margin"] is None else f"{confidence_info['top1_top2_margin']:.6f}",
            margin_threshold,
            None if confidence_info.get("best_score_z") is None else f"{confidence_info['best_score_z']:.6f}",
            confidence_info.get("scale_support_count"),
            confidence_info.get("required_scale_support"),
            confidence_info.get("scale_override_by_strong_score"),
            confidence_info.get("query_support_count"),
            confidence_info.get("required_query_support"),
            confidence_info.get("selected_cluster_support_count"),
            confidence_info.get("query_consensus_override_applied"),
            None if confidence_info.get("low_score_consensus_threshold") is None else f"{confidence_info['low_score_consensus_threshold']:.6f}",
            None if confidence_info.get("low_score_consensus_max_threshold") is None else f"{confidence_info['low_score_consensus_max_threshold']:.6f}",
            None if confidence_info.get("low_score_consensus_best_score_z_threshold") is None else f"{confidence_info['low_score_consensus_best_score_z_threshold']:.6f}",
            confidence_info.get("low_score_consensus_override_applied"),
            None if confidence_info.get("temporal_iou_with_cached_box") is None else f"{confidence_info['temporal_iou_with_cached_box']:.6f}",
            confidence_info.get("temporal_consistency_override_applied"),
            confidence_info.get("is_roi_directly_acceptable"),
            None if confidence_info.get("roi_direct_score_threshold") is None else f"{confidence_info['roi_direct_score_threshold']:.6f}",
            None if confidence_info.get("roi_direct_temporal_iou_threshold") is None else f"{confidence_info['roi_direct_temporal_iou_threshold']:.6f}",
            confidence_info["is_match_confident"],
            confidence_info["reason"],
            annotated_image_name,
        )

        request_total_elapsed = perf_counter() - request_start_time
        logger.info(
            "阶段耗时 | 接口请求完成 | match=%.3fs | save_image=%.3fs | total=%.3fs",
            match_elapsed,
            save_image_elapsed,
            request_total_elapsed,
        )

        return {
            "small_image_path": result.get("small_image_path"),
            "input_small_image_path": small_image_path,
            "input_small_image_dir": small_image_dir,
            "query_image_count": len(query_image_paths),
            "best_query_image_name": result.get("query_image_name"),
            "frame_size": result["frame_size"],
            "match_box": {
                "x1": int(best_box[0]),
                "y1": int(best_box[1]),
                "x2": int(best_box[2]),
                "y2": int(best_box[3]),
            },
            "best_score": result["best_match"]["score"],
            "match_confidence": core_match_confidence,
            "match_confidence_debug": debug_match_confidence,
            "annotated_image_saved": should_save_annotated_image,
            "annotated_image_path": annotated_image_path,
            "search_strategy": search_strategy,
            "roi_fallback_triggered": roi_fallback_triggered,
            "roi_config": {
                "roi_expand_ratio": float(roi_expand_ratio),
                "roi_min_size": int(roi_min_size),
            },
            "ranking_summary": result.get("ranking_summary"),
            "timing": {
                "read_frame_elapsed_sec": float(read_frame_elapsed),
                "decode_frame_elapsed_sec": float(decode_frame_elapsed),
                "collect_query_elapsed_sec": float(collect_query_elapsed),
                "load_small_image_elapsed_sec": float(load_small_image_elapsed),
                "match_elapsed_sec": float(match_elapsed),
                "save_image_elapsed_sec": float(save_image_elapsed),
                "total_elapsed_sec": float(request_total_elapsed),
                "match_detail": result_timing,
            },
        }
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.exception("小图文件或目录不存在")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("相似区域查找参数无效")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("相似区域查找失败")
        raise HTTPException(status_code=500, detail=f"相似区域查找失败: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_dino_match_service:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
    )
