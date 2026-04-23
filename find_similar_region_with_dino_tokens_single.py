import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dino import build_default_extractor


# 基础日志，便于观察 token 网格尺寸和匹配过程
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("find_similar_region_with_dino_tokens")


def compute_region_score(
    region: torch.Tensor,
    query_tokens: torch.Tensor,
    topk_ratio: float = 0.6,
) -> float:
    """
    计算一个候选区域和小图 token 网格的匹配分数。

    做法：
    - 先逐 patch 计算余弦相似度（前面已经做过 L2 normalize，所以点积就是余弦）
    - 再只取 top-k 的 patch 分数做平均
    """
    patch_scores = torch.sum(region * query_tokens, dim=-1).reshape(-1)

    k = max(1, int(round(patch_scores.numel() * topk_ratio)))
    topk_scores = torch.topk(patch_scores, k=k).values
    return float(topk_scores.mean().item())

def load_rgb_image(image_path: str) -> Image.Image:
    """读取图片并统一转成 RGB。"""
    image = Image.open(image_path).convert("RGB")
    return image


def pad_to_patch_multiple(
    image: Image.Image,
    patch_size: int,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, dict[str, int]]:
    """
    将图片右侧和下侧 pad 到 patch_size 的整数倍，不改变原图内容比例。

    返回：
    - padded_image: pad 后图片
    - pad_info:
        {
            "original_width": 原始宽,
            "original_height": 原始高,
            "padded_width": pad后宽,
            "padded_height": pad后高,
        }
    """
    width, height = image.size

    target_width = max(patch_size, math.ceil(width / patch_size) * patch_size)
    target_height = max(patch_size, math.ceil(height / patch_size) * patch_size)

    if target_width == width and target_height == height:
        return image, {
            "original_width": width,
            "original_height": height,
            "padded_width": width,
            "padded_height": height,
        }

    logger.info(
        "图片 pad 到 patch 整数倍: (%s, %s) -> (%s, %s)",
        width,
        height,
        target_width,
        target_height,
    )

    padded = Image.new("RGB", (target_width, target_height), fill_color)
    padded.paste(image, (0, 0))

    return padded, {
        "original_width": width,
        "original_height": height,
        "padded_width": target_width,
        "padded_height": target_height,
    }


def get_patch_size(extractor) -> int:
    """
    尝试从模型配置中读取 patch size。
    如果读取不到，则回退到 16。
    """
    patch_size = getattr(extractor.model.config, "patch_size", None)
    if patch_size is None:
        patch_size = 16
    return int(patch_size)


def get_prefix_token_count(
    extractor,
    total_token_count: int,
    expected_patch_token_count: int,
) -> int:
    """
    获取模型输出前缀 token 数量。

    DINOv3 某些模型输出中除了 1 个 CLS token 外，
    还可能存在若干 register tokens。

    优先从模型配置中读取 num_register_tokens；
    如果配置里没有，再根据总 token 数和 patch token 数的差值兜底推断。
    """
    num_register_tokens = getattr(extractor.model.config, "num_register_tokens", None)
    if num_register_tokens is not None:
        prefix_token_count = 1 + int(num_register_tokens)
        logger.info(
            "从模型配置读取到 register tokens 数量: %s，前缀 token 总数: %s",
            num_register_tokens,
            prefix_token_count,
        )
        return prefix_token_count

    inferred_prefix_count = total_token_count - expected_patch_token_count
    if inferred_prefix_count <= 0:
        raise ValueError(
            "无法推断前缀 token 数量: "
            f"total_token_count={total_token_count}, expected_patch_token_count={expected_patch_token_count}"
        )

    logger.warning(
        "模型配置中未找到 num_register_tokens，按差值兜底推断前缀 token 数量: %s",
        inferred_prefix_count,
    )
    return inferred_prefix_count


@torch.no_grad()
def extract_patch_tokens(extractor, image: Image.Image) -> dict[str, Any]:
    """
    提取图片的 patch token，并还原成二维网格。

    返回：
    - padded_image: pad 到 patch 整数倍后的图片
    - tokens: shape=(grid_h, grid_w, dim) 的归一化 token
    - grid_size: token 网格尺寸
    - image_size: 原始图片尺寸
    - padded_size: pad 后图片尺寸
    """
    patch_size = get_patch_size(extractor)

    padded_image, pad_info = pad_to_patch_multiple(image, patch_size)
    padded_width = pad_info["padded_width"]
    padded_height = pad_info["padded_height"]

    inputs = extractor.processor(
        images=padded_image,
        return_tensors="pt",
        do_resize=False,
    ).to(extractor.device)
    outputs = extractor.model(**inputs)

    grid_w = padded_width // patch_size
    grid_h = padded_height // patch_size
    expected_token_count = grid_w * grid_h
    total_token_count = outputs.last_hidden_state.shape[1]

    prefix_token_count = get_prefix_token_count(
        extractor=extractor,
        total_token_count=total_token_count,
        expected_patch_token_count=expected_token_count,
    )

    if total_token_count < prefix_token_count:
        raise ValueError(
            "模型输出 token 数量小于前缀 token 数量: "
            f"total_token_count={total_token_count}, prefix_token_count={prefix_token_count}"
        )

    # 去掉 CLS token 和可能存在的 register tokens，只保留 patch token
    patch_tokens = outputs.last_hidden_state[:, prefix_token_count:, :]
    patch_tokens = F.normalize(patch_tokens, dim=-1)

    token_count = patch_tokens.shape[1]
    if token_count != expected_token_count:
        raise ValueError(
            "patch token 数量和网格尺寸不匹配: "
            f"token_count={token_count}, expected={expected_token_count}, "
            f"grid=({grid_h}, {grid_w}), padded_image=({padded_width}, {padded_height}), "
            f"patch_size={patch_size}, total_token_count={total_token_count}, "
            f"prefix_token_count={prefix_token_count}"
        )

    token_grid = patch_tokens.reshape(1, grid_h, grid_w, -1).squeeze(0).cpu()

    logger.info(
        "patch token 提取完成: original=(%s,%s), padded=(%s,%s), patch=%s, grid=(%s,%s), dim=%s",
        image.size[0],
        image.size[1],
        padded_width,
        padded_height,
        patch_size,
        grid_h,
        grid_w,
        token_grid.shape[-1],
    )

    return {
        "padded_image": padded_image,
        "tokens": token_grid,
        "grid_size": {"height": grid_h, "width": grid_w},
        "image_size": {"width": image.size[0], "height": image.size[1]},
        "padded_size": {"width": padded_width, "height": padded_height},
        "patch_size": patch_size,
        "pad_info": pad_info,
    }

def compute_similarity_map(
    big_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    topk_ratio: float = 0.6,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    在大图 token 网格上滑动小图 token 网格，计算每个位置的鲁棒匹配分数。
    """
    big_h, big_w, dim = big_tokens.shape
    query_h, query_w, query_dim = query_tokens.shape

    if dim != query_dim:
        raise ValueError(f"大图 token 维度和小图 token 维度不一致: {dim} vs {query_dim}")
    if query_h > big_h or query_w > big_w:
        raise ValueError(
            f"小图 token 网格大于大图 token 网格: query=({query_h},{query_w}), big=({big_h},{big_w})"
        )

    candidates: list[dict[str, Any]] = []
    total_positions = (big_h - query_h + 1) * (big_w - query_w + 1)
    score_map = np.zeros((big_h - query_h + 1, big_w - query_w + 1), dtype=np.float32)
    logger.info("开始密集匹配，总 token 位置数量: %s", total_positions)

    processed = 0
    for top in range(big_h - query_h + 1):
        for left in range(big_w - query_w + 1):
            region = big_tokens[top:top + query_h, left:left + query_w, :]
            score = compute_region_score(
                region=region,
                query_tokens=query_tokens,
                topk_ratio=topk_ratio,
            )
            score_map[top, left] = score

            candidates.append({
                "score": score,
                "token_box": [left, top, left + query_w, top + query_h],
            })

            processed += 1
            if processed % 200 == 0:
                logger.info("密集匹配进度: %s / %s", processed, total_positions)

    return candidates, score_map

def resize_small_image_with_scale(
    image: Image.Image,
    scale: float,
    min_size: int = 16,
) -> Image.Image:
    """
    按比例缩放小图，用于多尺度匹配。
    """
    width, height = image.size
    new_width = max(min_size, int(round(width * scale)))
    new_height = max(min_size, int(round(height * scale)))

    if new_width == width and new_height == height:
        return image

    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)

def token_box_to_pixel_box(
    token_box: list[int],
    patch_size: int,
    padded_big_size: tuple[int, int],
    original_big_size: tuple[int, int],
) -> list[int]:
    """
    把 token 网格坐标恢复成原图像素坐标。

    注意：
    - 现在前处理是 pad，不是 resize
    - 所以 token 对应到的像素坐标在左上区域与原图一致
    - 只需把越界部分裁回原图边界
    """
    left_token, top_token, right_token, bottom_token = token_box
    original_w, original_h = original_big_size

    x1 = left_token * patch_size
    y1 = top_token * patch_size
    x2 = right_token * patch_size
    y2 = bottom_token * patch_size

    # pad 只发生在右侧和下侧，所以左上坐标不需要缩放
    x1 = max(0, min(x1, original_w))
    y1 = max(0, min(y1, original_h))
    x2 = max(0, min(x2, original_w))
    y2 = max(0, min(y2, original_h))

    return [int(x1), int(y1), int(x2), int(y2)]


def iou(box1: list[int], box2: list[int]) -> float:
    """计算两个像素框的 IoU，用于简单去重。"""
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


def select_top_candidates(
    candidates: list[dict[str, Any]],
    topk: int,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """
    对密集匹配的结果做简单 NMS 去重，避免返回一堆重叠框。
    """
    sorted_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []

    for candidate in sorted_candidates:
        current_box = candidate["pixel_box"]
        if any(iou(current_box, item["pixel_box"]) > iou_threshold for item in selected):
            continue

        selected.append(candidate)
        if len(selected) >= topk:
            break

    return selected

def judge_match_confidence(
    top_candidates: list[dict[str, Any]],
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    """
    判断当前最佳匹配是否足够可信。

    规则：
    - top1 分数必须 >= score_threshold
    - top1 和 top2 的分差必须 >= margin_threshold
    """
    if not top_candidates:
        return {
            "is_match_confident": False,
            "top1_score": None,
            "top2_score": None,
            "top1_top2_margin": None,
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            "reason": "no_candidates",
        }

    top1_score = float(top_candidates[0]["score"])
    top2_score = float(top_candidates[1]["score"]) if len(top_candidates) >= 2 else None
    margin = top1_score - top2_score if top2_score is not None else float("inf")

    is_confident = (top1_score >= score_threshold) and (margin >= margin_threshold)

    reason_parts = []
    if top1_score < score_threshold:
        reason_parts.append("top1_score_below_threshold")
    if margin < margin_threshold:
        reason_parts.append("top1_top2_margin_too_small")
    if not reason_parts:
        reason_parts.append("ok")

    return {
        "is_match_confident": bool(is_confident),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "top1_top2_margin": None if margin == float("inf") else float(margin),
        "score_threshold": float(score_threshold),
        "margin_threshold": float(margin_threshold),
        "reason": ",".join(reason_parts),
    }


def draw_best_box(big_image_path: str, best_box: list[int], output_path: str):
    """把最佳匹配框画到大图上。"""
    with Image.open(big_image_path) as image:
        rgb_image = image.convert("RGB")
        draw = ImageDraw.Draw(rgb_image)
        draw.rectangle(best_box, outline="red", width=4)
        rgb_image.save(output_path)

    logger.info("最佳框可视化已保存: %s", output_path)


def normalize_score_map(score_map: np.ndarray) -> np.ndarray:
    """
    将相似度矩阵归一化到 0~1，便于生成热力图。
    """
    min_value = float(np.min(score_map))
    max_value = float(np.max(score_map))
    if max_value - min_value < 1e-8:
        return np.zeros_like(score_map, dtype=np.float32)
    return (score_map - min_value) / (max_value - min_value)


def build_heatmap_image(score_map: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    """
    将二维相似度矩阵转换为伪彩色热力图。
    颜色规则：
    - 低分：蓝色
    - 中分：黄色
    - 高分：红色
    """
    normalized = normalize_score_map(score_map)
    height, width = normalized.shape

    red = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    green = np.clip((1.0 - np.abs(normalized - 0.5) * 2.0) * 255.0, 0, 255).astype(np.uint8)
    blue = np.clip((1.0 - normalized) * 255.0, 0, 255).astype(np.uint8)

    heatmap_array = np.stack([red, green, blue], axis=-1).reshape(height, width, 3)
    heatmap_image = Image.fromarray(heatmap_array, mode="RGB")
    return heatmap_image.resize(output_size, Image.Resampling.BICUBIC)


def save_heatmap_images(
    big_image_path: str,
    score_map: np.ndarray,
    heatmap_output_path: str,
    overlay_output_path: str,
):
    """
    保存两张图：
    - 纯热力图
    - 热力图叠加到原图上的 overlay
    """
    with Image.open(big_image_path) as image:
        rgb_image = image.convert("RGB")
        heatmap_image = build_heatmap_image(score_map, rgb_image.size)
        heatmap_image.save(heatmap_output_path)

        overlay_image = Image.blend(rgb_image, heatmap_image, alpha=0.45)
        overlay_image.save(overlay_output_path)

    logger.info("热力图已保存: %s", Path(heatmap_output_path).resolve())
    logger.info("热力图叠加图已保存: %s", Path(overlay_output_path).resolve())


def find_similar_region_with_tokens(
    big_image_path: str,
    small_image_path: str,
    topk: int,
    iou_threshold: float,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    使用 DINO patch/token 特征做多尺度密集匹配。
    """
    if scales is None:
        scales = [0.75, 0.9, 1.0, 1.1, 1.25]

    extractor = build_default_extractor()
    original_big_image = load_rgb_image(big_image_path)
    original_small_image = load_rgb_image(small_image_path)

    # 大图只提一次 token，避免重复算
    big_info = extract_patch_tokens(extractor, original_big_image)

    original_big_size = original_big_image.size
    padded_big_size = (
        big_info["padded_size"]["width"],
        big_info["padded_size"]["height"],
    )

    all_candidates: list[dict[str, Any]] = []
    best_score_map = None
    best_scale = None
    score_map_by_scale: dict[float, np.ndarray] = {}

    for scale in scales:
        scaled_small_image = resize_small_image_with_scale(original_small_image, scale=scale)

        logger.info(
            "开始处理小图尺度: scale=%.3f, resized_small=(%s,%s)",
            scale,
            scaled_small_image.size[0],
            scaled_small_image.size[1],
        )

        small_info = extract_patch_tokens(extractor, scaled_small_image)

        # 如果小图 token 网格比大图还大，跳过这个尺度
        if (
            small_info["grid_size"]["height"] > big_info["grid_size"]["height"]
            or small_info["grid_size"]["width"] > big_info["grid_size"]["width"]
        ):
            logger.warning(
                "跳过尺度 %.3f：小图 token 网格大于大图 token 网格，small=%s, big=%s",
                scale,
                small_info["grid_size"],
                big_info["grid_size"],
            )
            continue

        raw_candidates, score_map = compute_similarity_map(
            big_tokens=big_info["tokens"],
            query_tokens=small_info["tokens"],
            topk_ratio=topk_ratio,
        )
        score_map_by_scale[scale] = score_map

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
                    "width": scaled_small_image.size[0],
                    "height": scaled_small_image.size[1],
                },
                "small_token_grid": small_info["grid_size"],
            })

    if not all_candidates:
        raise ValueError("所有尺度都无法生成有效候选，请检查输入图片尺寸和 patch size。")

    top_candidates = select_top_candidates(
        candidates=all_candidates,
        topk=topk,
        iou_threshold=iou_threshold,
    )

    best_match = top_candidates[0]
    best_scale = best_match["scale"]
    best_score_map = score_map_by_scale[best_scale]

    confidence_info = judge_match_confidence(
        top_candidates=top_candidates,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
    )

    result = {
        "big_image_path": str(Path(big_image_path).resolve()),
        "small_image_path": str(Path(small_image_path).resolve()),
        "big_image_size": {
            "width": original_big_size[0],
            "height": original_big_size[1],
        },
        "small_image_size": {
            "width": original_small_image.size[0],
            "height": original_small_image.size[1],
        },
        "patch_size": big_info["patch_size"],
        "big_token_grid": big_info["grid_size"],
        "searched_scales": scales,
        "topk_ratio": float(topk_ratio),
        "best_scale_for_heatmap": best_scale,
        "total_positions": len(all_candidates),
        "best_match": best_match,
        "top_matches": top_candidates,
        "match_confidence": confidence_info,
    }

    return result, best_score_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 DINO patch/token 特征做密集匹配，在大图中定位与小图最相似的位置"
    )
    parser.add_argument(
        "--big-image",
        required=True,
        help="大图路径",
    )
    parser.add_argument(
        "--small-image",
        required=True,
        help="小图路径",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="返回前 topk 个候选区域",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="候选去重使用的 IoU 阈值",
    )
    parser.add_argument(
        "--result-json",
        default="similar_region_token_result.json",
        help="结果 JSON 输出路径",
    )
    parser.add_argument(
        "--vis-output",
        default="similar_region_token_best_box.jpg",
        help="最佳框可视化输出路径",
    )
    parser.add_argument(
        "--heatmap-output",
        default="similar_region_token_heatmap.jpg",
        help="纯热力图输出路径",
    )
    parser.add_argument(
        "--overlay-output",
        default="similar_region_token_heatmap_overlay.jpg",
        help="热力图叠加到原图上的输出路径",
    )
    parser.add_argument(
    "--scales",
    nargs="+",
    type=float,
    default=[0.75, 0.9, 1.0, 1.1, 1.25],
    help="多尺度匹配使用的小图缩放比例列表",
    )
    parser.add_argument(
    "--topk-ratio",
    type=float,
    default=0.6,
    help="每个候选区域只保留 top-k patch 分数做平均，范围建议 0.3~1.0",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="最佳候选分数低于该阈值时，认为匹配不可靠",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.03,
        help="top1 和 top2 分差低于该阈值时，认为匹配不可靠",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result, score_map = find_similar_region_with_tokens(
        big_image_path=args.big_image,
        small_image_path=args.small_image,
        topk=args.topk,
        iou_threshold=args.iou_threshold,
        scales=args.scales,
        topk_ratio=args.topk_ratio,
        score_threshold=args.score_threshold,
        margin_threshold=args.margin_threshold,
    )

    result_json_path = Path(args.result_json)
    with open(result_json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    logger.info("结果 JSON 已保存: %s", result_json_path.resolve())

    draw_best_box(
        big_image_path=args.big_image,
        best_box=result["best_match"]["pixel_box"],
        output_path=args.vis_output,
    )
    save_heatmap_images(
        big_image_path=args.big_image,
        score_map=score_map,
        heatmap_output_path=args.heatmap_output,
        overlay_output_path=args.overlay_output,
    )

    print("\n===== 最佳匹配结果 =====")
    print(f"最佳 token 框: {result['best_match']['token_box']}")
    print(f"最佳像素框: {result['best_match']['pixel_box']}")
    print(f"最佳相似度: {result['best_match']['score']:.6f}")
    print(f"大图 token 网格: {result['big_token_grid']}")
    print(f"最佳尺度下的小图尺寸: {result['best_match']['scaled_small_image_size']}")
    print(f"最佳尺度下的小图 token 网格: {result['best_match']['small_token_grid']}")
    print(f"总匹配位置数: {result['total_positions']}")
    print(f"topk_ratio: {result['topk_ratio']}")
    print(f"热力图输出: {Path(args.heatmap_output).resolve()}")
    print(f"热力图叠加输出: {Path(args.overlay_output).resolve()}")

    confidence = result["match_confidence"]
    print("\n===== 匹配可信度判断 =====")
    print(f"是否可信匹配: {confidence['is_match_confident']}")
    print(f"top1 分数: {confidence['top1_score']}")
    print(f"top2 分数: {confidence['top2_score']}")
    print(f"top1-top2 分差: {confidence['top1_top2_margin']}")
    print(f"score_threshold: {confidence['score_threshold']}")
    print(f"margin_threshold: {confidence['margin_threshold']}")
    print(f"判定原因: {confidence['reason']}")


if __name__ == "__main__":
    main()


"""
python find_similar_region_with_dino_tokens.py \
  --big-image /home/ubuntu/yzm_workspace/compare_embedding/frames_456/test01.jpg \
  --small-image /home/ubuntu/yzm_workspace/compare_embedding/cut.png \
  --topk 5 \
  --iou-threshold 0.3 \
  --scales 0.75 0.9 1.0 1.1 1.25 \
  --topk-ratio 0.6 \
  --score-threshold 0.35 \
  --margin-threshold 0.03
"""