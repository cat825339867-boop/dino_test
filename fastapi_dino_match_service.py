import io
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("dino_match_service")


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
    output_path: str | None = None,
) -> str:
    """
    将匹配框画到帧图上并保存。

    如果调用方未指定输出路径，则默认保存在当前目录下。
    """
    if output_path is None or not output_path.strip():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")
        output_path = f"out_image/{timestamp}.jpg"

    output_file = Path(output_path).expanduser()
    if not output_file.is_absolute():
        output_file = Path.cwd() / output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotated_image = frame_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    draw.rectangle(best_box, outline="red", width=4)
    annotated_image.save(output_file)

    logger.info("已保存画框结果图: %s", output_file)
    return str(output_file.resolve())


@lru_cache(maxsize=1)
def get_feature_extractor():
    """
    缓存 DINO 提取器，避免重复加载模型。
    """
    logger.info("开始初始化 DINO 匹配服务提取器")
    extractor = build_default_extractor()
    logger.info("DINO 匹配服务提取器初始化完成")
    return extractor


def find_similar_region_in_memory(
    extractor,
    frame_image: Image.Image,
    query_image: Image.Image,
    topk: int = 5,
    iou_threshold: float = 0.5,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    confidence_iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    在内存中的视频帧里查找小图最相似的位置。

    这里复用单图 token 匹配逻辑，但避免把视频帧先写入磁盘。
    """
    total_start_time = perf_counter()

    if scales is None:
        # scales = [0.75, 0.9, 1.0, 1.1, 1.25]
        scales = [0.8,1.0,1.2]

    big_token_start_time = perf_counter()
    big_info = extract_patch_tokens(extractor, frame_image)
    big_token_elapsed = perf_counter() - big_token_start_time
    original_big_size = frame_image.size
    padded_big_size = (
        big_info["padded_size"]["width"],
        big_info["padded_size"]["height"],
    )

    logger.info(
        "阶段耗时 | 大图 token 提取完成 | frame_size=(%s,%s) | grid=(%s,%s) | elapsed=%.3fs",
        original_big_size[0],
        original_big_size[1],
        big_info["grid_size"]["height"],
        big_info["grid_size"]["width"],
        big_token_elapsed,
    )

    all_candidates: list[dict[str, Any]] = []
    scale_timing_details: list[dict[str, Any]] = []

    for scale in scales:
        scale_start_time = perf_counter()
        scaled_query_image = resize_small_image_with_scale(query_image, scale=scale)

        small_token_start_time = perf_counter()
        small_info = extract_patch_tokens(extractor, scaled_query_image)
        small_token_elapsed = perf_counter() - small_token_start_time

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
            })
            continue

        match_start_time = perf_counter()
        raw_candidates, _ = compute_similarity_map(
            big_tokens=big_info["tokens"],
            query_tokens=small_info["tokens"],
            topk_ratio=topk_ratio,
        )
        match_elapsed = perf_counter() - match_start_time

        for item in raw_candidates:
            pixel_box = token_box_to_pixel_box(
                token_box=item["token_box"],
                patch_size=big_info["patch_size"],
                padded_big_size=padded_big_size,
                original_big_size=original_big_size,
            )
            all_candidates.append({
                "score": item["score"],
                "token_box": item["token_box"],
                "pixel_box": pixel_box,
                "scale": scale,
                "scaled_small_image_size": {
                    "width": scaled_query_image.size[0],
                    "height": scaled_query_image.size[1],
                },
                "small_token_grid": small_info["grid_size"],
            })

        scale_total_elapsed = perf_counter() - scale_start_time
        scale_timing_details.append({
            "scale": float(scale),
            "small_token_elapsed_sec": float(small_token_elapsed),
            "match_elapsed_sec": float(match_elapsed),
            "total_elapsed_sec": float(scale_total_elapsed),
            "skipped": False,
            "candidate_count": len(raw_candidates),
        })
        logger.info(
            "阶段耗时 | scale=%.3f | small_size=(%s,%s) | small_grid=(%s,%s) | token=%.3fs | match=%.3fs | total=%.3fs | candidates=%s",
            scale,
            scaled_query_image.size[0],
            scaled_query_image.size[1],
            small_info["grid_size"]["height"],
            small_info["grid_size"]["width"],
            small_token_elapsed,
            match_elapsed,
            scale_total_elapsed,
            len(raw_candidates),
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
            "width": query_image.size[0],
            "height": query_image.size[1],
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
            "big_token_elapsed_sec": float(big_token_elapsed),
            "postprocess_elapsed_sec": float(postprocess_elapsed),
            "total_elapsed_sec": float(total_elapsed),
            "scale_details": scale_timing_details,
        },
    }


def find_similar_region_from_cv2_frame(
    frame: np.ndarray,
    small_image_path: str,
    topk: int = 5,
    iou_threshold: float = 0.5,
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

    with Image.open(small_image_path) as image:
        query_image = image.convert("RGB")

    result = find_similar_region_in_memory(
        extractor=extractor,
        frame_image=frame_image,
        query_image=query_image,
        topk=topk,
        iou_threshold=iou_threshold,
        scales=scales,
        topk_ratio=topk_ratio,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        confidence_iou_threshold=confidence_iou_threshold,
    )

    best_box = result["best_match"]["pixel_box"]
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
        "match_confidence": result["match_confidence"],
        "timing": result.get("timing"),
    }


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
    small_image_path: str = Form('chen.jpg', description="待查找的小图本地路径"),
    scales: str = Form("1.0,1.2", description="多尺度列表，逗号分隔"),
    topk: int = Form(5, description="候选区域数量"),
    iou_threshold: float = Form(0.5, description="候选去重使用的 IoU 阈值"),
    topk_ratio: float = Form(0.6, description="每个候选区域保留的 top-k patch 比例"),
    score_threshold: float = Form(0.35, description="最佳候选分数阈值"),
    margin_threshold: float = Form(0.001, description="top1 与 top2 分差阈值"),
    confidence_iou_threshold: float = Form(0.3, description="可信度判断时用于过滤重复框的 IoU 阈值"),
    save_annotated_image: str = Form("false", description="是否保存画框后的结果图，支持 true/false"),
    annotated_output_path: str | None = Form(None, description="画框结果图输出路径；未传时使用默认文件名"),
) -> dict[str, Any]:
    """
    在视频帧中查找指定小图的位置。

    必填参数只有两个：
    - frame_file: 视频帧图片内容
    - small_image_path: 本地小图路径
    """
    logger.info(
        "收到相似区域查找请求: frame_file=%s, small_image_path=%s",
        frame_file.filename,
        small_image_path,
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

        load_small_image_start_time = perf_counter()
        with Image.open(small_image_path) as image:
            query_image = image.convert("RGB")
        load_small_image_elapsed = perf_counter() - load_small_image_start_time

        logger.info(
            "阶段耗时 | 请求预处理完成 | read_frame=%.3fs | decode_frame=%.3fs | load_small_image=%.3fs",
            read_frame_elapsed,
            decode_frame_elapsed,
            load_small_image_elapsed,
        )

        match_start_time = perf_counter()
        result = find_similar_region_in_memory(
            extractor=extractor,
            frame_image=frame_image,
            query_image=query_image,
            topk=topk,
            iou_threshold=iou_threshold,
            scales=parse_scales(scales),
            topk_ratio=topk_ratio,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            confidence_iou_threshold=confidence_iou_threshold,
        )
        match_elapsed = perf_counter() - match_start_time

        best_box = result["best_match"]["pixel_box"]
        should_save_annotated_image = parse_bool_value(save_annotated_image)
        annotated_image_path = None
        if should_save_annotated_image:
            save_image_start_time = perf_counter()
            annotated_image_path = save_annotated_frame_image(
                frame_image=frame_image,
                best_box=best_box,
                output_path=annotated_output_path,
            )
            save_image_elapsed = perf_counter() - save_image_start_time
        else:
            save_image_elapsed = 0.0

        request_total_elapsed = perf_counter() - request_start_time
        logger.info(
            "阶段耗时 | 接口请求完成 | match=%.3fs | save_image=%.3fs | total=%.3fs",
            match_elapsed,
            save_image_elapsed,
            request_total_elapsed,
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
            "match_confidence": result["match_confidence"],
            "annotated_image_saved": should_save_annotated_image,
            "annotated_image_path": annotated_image_path,
            "timing": {
                "read_frame_elapsed_sec": float(read_frame_elapsed),
                "decode_frame_elapsed_sec": float(decode_frame_elapsed),
                "load_small_image_elapsed_sec": float(load_small_image_elapsed),
                "match_elapsed_sec": float(match_elapsed),
                "save_image_elapsed_sec": float(save_image_elapsed),
                "total_elapsed_sec": float(request_total_elapsed),
                "match_detail": result.get("timing"),
            },
        }
    except FileNotFoundError as exc:
        logger.exception("小图文件不存在")
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
