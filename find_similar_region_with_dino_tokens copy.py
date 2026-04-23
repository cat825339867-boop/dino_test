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


def is_image_file(path: Path) -> bool:
    """
    判断一个文件是否是常见图片格式。
    """
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rgb_image(image_path: str) -> Image.Image:
    """读取图片并统一转成 RGB。"""
    image = Image.open(image_path).convert("RGB")
    return image


def collect_small_image_paths(
    small_image: str | None,
    small_image_dir: str | None,
) -> list[str]:
    """
    收集待匹配的小图路径。

    规则：
    - --small-image 和 --small-image-dir 二选一
    - 如果给的是目录，则收集目录下所有常见图片文件
    """
    if bool(small_image) == bool(small_image_dir):
        raise ValueError("必须且只能提供一个参数：--small-image 或 --small-image-dir")

    if small_image is not None:
        image_path = Path(small_image)
        if not image_path.is_file():
            raise FileNotFoundError(f"小图文件不存在: {image_path}")
        return [str(image_path)]

    image_dir = Path(small_image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"小图目录不存在: {image_dir}")

    image_paths = sorted(
        [str(p) for p in image_dir.iterdir() if p.is_file() and is_image_file(p)]
    )
    if not image_paths:
        raise ValueError(f"目录下未找到可用图片: {image_dir}")

    return image_paths


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

            candidates.append(
                {
                    "score": score,
                    "token_box": [left, top, left + query_w, top + query_h],
                }
            )

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
    ranked_candidates: list[dict[str, Any]],
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    """
    判断当前最佳匹配是否足够可信。

    规则：
    - top1 分数必须 >= score_threshold
    - top1 和 top2 的分差必须 >= margin_threshold
    """
    if not ranked_candidates:
        return {
            "is_match_confident": False,
            "top1_score": None,
            "top2_score": None,
            "top1_top2_margin": None,
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            "reason": "no_candidates",
        }

    top1_score = float(ranked_candidates[0]["score"])
    top2_score = float(ranked_candidates[1]["score"]) if len(ranked_candidates) >= 2 else None
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


def get_match_token_count(match: dict[str, Any]) -> int:
    """
    读取一个候选框对应的小图 token 数量。

    说明：
    - token 数量越大，模板越“宽泛”，更容易形成大框和大簇
    - 多图竞争时，我们希望对这类大模板做轻微惩罚，让结果更偏向定位精确的 query
    """
    small_grid = match.get("small_token_grid", {})
    height = int(small_grid.get("height", 0) or 0)
    width = int(small_grid.get("width", 0) or 0)
    token_count = height * width
    return max(1, token_count)


def compute_query_quality_score(item: dict[str, Any]) -> float:
    """
    计算跨 query 可比的质量分。

    目标：
    - 避免 `is_match_confident` 这种布尔量一票否决
    - 兼顾分数高度、热力图尖锐程度、top1-top2 分差
    - 对特别大的模板做更明显的惩罚，降低“大框更容易抱团”的偏置

    经验上，多图定位更怕“大而泛”的模板把结果拉偏，
    所以这里把原始分数权重调高，同时也提高大模板惩罚系数，
    让更小、更精确的 query 更容易在多图竞争里胜出。
    """
    confidence = item["match_confidence"]
    best_match = item["best_match"]
    score_stats = item.get("score_stats", {})

    raw_score = float(best_match["score"])
    best_score_z = float(score_stats.get("best_score_z", 0.0))
    if not np.isfinite(best_score_z):
        best_score_z = 0.0

    margin = confidence.get("top1_top2_margin")
    if margin is None or not np.isfinite(margin):
        margin = 0.0

    token_count = get_match_token_count(best_match)
    precision_penalty = 0.40 * math.log(token_count)
    confidence_bonus = 0.15 if confidence["is_match_confident"] else 0.0

    return (
        best_score_z
        + raw_score * 5.0
        + max(0.0, float(margin)) * 4.0
        + confidence_bonus
        - precision_penalty
    )


def get_cluster_candidate_sort_key(candidate: dict[str, Any]) -> tuple:
    """
    给参与多图聚类的候选打一个排序键。

    目标：
    - 先让更靠谱的 query 候选参与聚类
    - 同一个 query 的 top1 候选优先于 top2/top3
    """
    query_result = candidate["query_result"]
    query_quality_score = compute_query_quality_score(query_result)
    selected_match = candidate["selected_match"]
    candidate_rank = int(candidate["candidate_rank"])

    return (
        query_quality_score,
        float(selected_match["score"]),
        -candidate_rank,
    )


def compute_cluster_member_quality(member: dict[str, Any]) -> float:
    """
    计算某个聚类成员的质量分。

    这里不再只看“支持数”，而是把 query 自身质量和当前候选质量一起考虑：
    - query 自身在全局竞争里的质量分
    - 当前候选自己的 raw score
    - 如果当前候选只是该 query 的 top2/top3，做惩罚
    """
    query_result = member["query_result"]
    query_quality_score = compute_query_quality_score(query_result)
    selected_match = member["selected_match"]
    candidate_rank = int(member["candidate_rank"])

    return (
        query_quality_score
        + float(selected_match["score"]) * 1.5
        - 0.45 * max(0, candidate_rank - 1)
    )


def get_cluster_member_ranking_key(member: dict[str, Any]) -> tuple:
    """
    给簇内成员排序，用于选择最终代表 query。

    优先顺序：
    - 聚类成员质量分
    - 当前候选是不是该 query 的 top1
    - 当前候选 raw score
    """
    selected_match = member["selected_match"]
    candidate_rank = int(member["candidate_rank"])
    return (
        compute_cluster_member_quality(member),
        int(candidate_rank == 1),
        float(selected_match["score"]),
    )


def get_global_ranking_key(item: dict[str, Any]) -> tuple:
    """
    给多张小图的匹配结果生成排序键。

    排序优先级：
    1. query 综合质量分
    2. 当前 query 内部 best 是否足够突出（best_score_z）
    3. 原始 best_match score
    4. top1-top2 margin
    """
    confidence = item["match_confidence"]
    best_match = item["best_match"]
    score_stats = item.get("score_stats", {})
    query_quality_score = compute_query_quality_score(item)
    best_score_z = float(score_stats.get("best_score_z", -1e9))
    margin = confidence["top1_top2_margin"]
    if margin is None:
        margin = -1e9
    raw_score = float(best_match["score"])

    return (
        query_quality_score,
        best_score_z,
        raw_score,
        float(margin),
    )


def average_boxes(boxes: list[list[int]]) -> list[int]:
    """
    对一组框做简单平均，得到最终共识框。
    """
    if not boxes:
        raise ValueError("boxes 不能为空")

    arr = np.asarray(boxes, dtype=np.float32)
    mean_box = np.mean(arr, axis=0)
    return [int(round(x)) for x in mean_box.tolist()]


def cluster_query_results_by_iou(
    query_results: list[dict[str, Any]],
    iou_threshold: float = 0.3,
    per_query_topk: int = 1,
) -> list[dict[str, Any]]:
    """
    把多张 query 的 top_matches 按 pixel_box 的 IoU 做简单聚类。

    规则：
    - 每个 query 拿自己的前 top-k 候选参与聚类
    - 如果和某个已有簇的代表框 IoU >= 阈值，就归入该簇
    - 否则新建一个簇
    - 同一张 query 在同一簇里只保留一个最强候选，避免重复投票
    """
    if per_query_topk <= 0:
        raise ValueError(f"per_query_topk 必须大于 0，当前值: {per_query_topk}")

    valid_items = [item for item in query_results if item.get("best_match") is not None]

    candidate_entries: list[dict[str, Any]] = []
    for item in valid_items:
        top_matches = item.get("top_matches") or []
        query_key = item.get("small_image_path") or item.get("query_image_name")

        for rank, match in enumerate(top_matches[:per_query_topk], start=1):
            candidate_entries.append(
                {
                    "query_key": query_key,
                    "query_result": item,
                    "query_image_name": item["query_image_name"],
                    "selected_match": dict(match),
                    "candidate_rank": rank,
                }
            )

    candidate_entries.sort(key=get_cluster_candidate_sort_key, reverse=True)
    clusters: list[dict[str, Any]] = []

    for candidate in candidate_entries:
        box = candidate["selected_match"]["pixel_box"]

        matched_cluster = None
        matched_iou = -1.0
        for cluster in clusters:
            rep_box = cluster["representative_box"]
            overlap = iou(box, rep_box)
            if overlap >= iou_threshold and overlap > matched_iou:
                matched_cluster = cluster
                matched_iou = overlap

        if matched_cluster is None:
            clusters.append(
                {
                    "representative_box": box,
                    "members_by_query": {
                        candidate["query_key"]: candidate,
                    },
                }
            )
        else:
            query_key = candidate["query_key"]
            existing = matched_cluster["members_by_query"].get(query_key)
            if existing is None or get_cluster_candidate_sort_key(candidate) > get_cluster_candidate_sort_key(existing):
                matched_cluster["members_by_query"][query_key] = candidate

            matched_cluster["representative_box"] = average_boxes(
                [
                    m["selected_match"]["pixel_box"]
                    for m in matched_cluster["members_by_query"].values()
                ]
            )

    normalized_clusters = []
    for cluster in clusters:
        members = list(cluster["members_by_query"].values())
        normalized_clusters.append(
            {
                "representative_box": cluster["representative_box"],
                "members": members,
            }
        )

    logger.info(
        "多图聚类完成: 候选数=%s, 簇数=%s, per_query_topk=%s, iou_threshold=%.3f",
        len(candidate_entries),
        len(normalized_clusters),
        per_query_topk,
        iou_threshold,
    )

    return normalized_clusters


def summarize_clusters(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    统计每个簇的支持数、平均分、平均 z、平均 margin，并排序。

    和旧逻辑不同，这里改成“质量优先、支持数次之”：
    - 先看簇内 query 整体质量是否高
    - 再看是不是有更多 query 支持
    """
    summaries = []

    for idx, cluster in enumerate(clusters, start=1):
        members = cluster["members"]

        support_count = len(members)
        avg_score = float(np.mean([m["selected_match"]["score"] for m in members]))
        avg_z = float(
            np.mean(
                [
                    m["query_result"].get("score_stats", {}).get("best_score_z", 0.0)
                    for m in members
                ]
            )
        )
        avg_margin = float(
            np.mean(
                [
                    m["query_result"]["match_confidence"]["top1_top2_margin"]
                    if m["query_result"]["match_confidence"]["top1_top2_margin"] is not None
                    else 0.0
                    for m in members
                ]
            )
        )
        avg_candidate_rank = float(np.mean([m["candidate_rank"] for m in members]))
        member_quality_scores = [compute_cluster_member_quality(m) for m in members]
        # 质量分优先，支持数只作为温和加成，避免“低质量 query 抱团”压过高质量少量证据。
        quality_score = float(
            np.mean(member_quality_scores) + 0.2 * max(0, support_count - 1)
        )

        summaries.append(
            {
                "cluster_id": idx,
                "support_count": support_count,
                "avg_score": avg_score,
                "avg_best_score_z": avg_z,
                "avg_margin": avg_margin,
                "avg_candidate_rank": avg_candidate_rank,
                "quality_score": quality_score,
                "consensus_box": average_boxes(
                    [m["selected_match"]["pixel_box"] for m in members]
                ),
                "member_query_names": [m["query_image_name"] for m in members],
                "member_details": [
                    {
                        "query_image_name": m["query_image_name"],
                        "candidate_rank": m["candidate_rank"],
                        "score": m["selected_match"]["score"],
                        "pixel_box": m["selected_match"]["pixel_box"],
                    }
                    for m in members
                ],
                "members": members,
            }
        )

    summaries.sort(
        key=lambda c: (
            c["quality_score"],
            c["avg_best_score_z"],
            c["support_count"],
            c["avg_margin"],
            c["avg_score"],
        ),
        reverse=True,
    )

    return summaries


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
    green = np.clip((1.0 - np.abs(normalized - 0.5) * 2.0) * 255.0, 0, 255).astype(
        np.uint8
    )
    blue = np.clip((1.0 - normalized) * 255.0, 0, 255).astype(np.uint8)

    heatmap_array = np.stack([red, green, blue], axis=-1).reshape(height, width, 3)
    heatmap_image = Image.fromarray(heatmap_array, mode="RGB")
    return heatmap_image.resize(output_size, Image.Resampling.BICUBIC)


def save_heatmap_images(
    big_image_path: str,
    score_map: np.ndarray,
    heatmap_output_path: str,
    overlay_output_path: str,
    best_box: list[int] | None = None,
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

        if best_box is not None:
            draw = ImageDraw.Draw(overlay_image)
            draw.rectangle(best_box, outline="lime", width=4)

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
    confidence_iou_threshold: float = 0.3,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    单图模式包装函数：
    内部先提取大图特征，再调用 match_one_query_with_precomputed_big。
    """
    extractor = build_default_extractor()
    original_big_image = load_rgb_image(big_image_path)
    big_info = extract_patch_tokens(extractor, original_big_image)

    return match_one_query_with_precomputed_big(
        extractor=extractor,
        big_image_path=big_image_path,
        original_big_image=original_big_image,
        big_info=big_info,
        small_image_path=small_image_path,
        topk=topk,
        iou_threshold=iou_threshold,
        scales=scales,
        topk_ratio=topk_ratio,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        confidence_iou_threshold=confidence_iou_threshold,
    )


def match_one_query_with_precomputed_big(
    extractor,
    big_image_path: str,
    original_big_image: Image.Image,
    big_info: dict[str, Any],
    small_image_path: str,
    topk: int,
    iou_threshold: float,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    confidence_iou_threshold: float = 0.3,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    在大图特征已预计算的前提下，对单张小图做多尺度密集匹配。
    """
    if scales is None:
        scales = [0.75, 0.9, 1.0, 1.1, 1.25]

    original_small_image = load_rgb_image(small_image_path)
    original_big_size = original_big_image.size

    all_candidates: list[dict[str, Any]] = []
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
                padded_big_size=(
                    big_info["padded_size"]["width"],
                    big_info["padded_size"]["height"],
                ),
                original_big_size=original_big_size,
            )
            all_candidates.append(
                {
                    "score": item["score"],
                    "token_box": item["token_box"],
                    "pixel_box": pixel_box,
                    "scale": scale,
                    "scaled_small_image_size": {
                        "width": scaled_small_image.size[0],
                        "height": scaled_small_image.size[1],
                    },
                    "small_token_grid": small_info["grid_size"],
                }
            )

    if not all_candidates:
        raise ValueError("所有尺度都无法生成有效候选，请检查输入图片尺寸和 patch size。")

    ranked_candidates = sorted(all_candidates, key=lambda item: item["score"], reverse=True)
    top_candidates = select_top_candidates(
        candidates=all_candidates,
        topk=topk,
        iou_threshold=iou_threshold,
    )
    # 可信度判断不再直接看原始 top2，而是和低重叠候选比较，
    # 避免同一目标附近的重复框把好 query 误判成“不可信”。
    confidence_candidates = select_top_candidates(
        candidates=all_candidates,
        topk=max(topk, 2),
        iou_threshold=confidence_iou_threshold,
    )

    best_match = top_candidates[0]
    best_scale = best_match["scale"]
    best_score_map = score_map_by_scale[best_scale]

    # 统计当前 query 自己的热力图分布，用于后续多 query 排名。
    score_mean = float(np.mean(best_score_map))
    score_std = float(np.std(best_score_map))
    best_score = float(best_match["score"])
    best_score_z = 0.0 if score_std < 1e-8 else (best_score - score_mean) / score_std

    confidence_info = judge_match_confidence(
        ranked_candidates=confidence_candidates,
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
        "confidence_iou_threshold": float(confidence_iou_threshold),
        "best_scale_for_heatmap": best_scale,
        "total_positions": len(all_candidates),
        "best_match": best_match,
        "top_matches": top_candidates,
        "confidence_candidates": confidence_candidates,
        "match_confidence": confidence_info,
        "query_image_name": Path(small_image_path).name,
        "score_stats": {
            "mean": score_mean,
            "std": score_std,
            "best_score_z": best_score_z,
        },
    }

    return result, best_score_map


def find_best_match_from_multiple_queries(
    big_image_path: str,
    small_image_paths: list[str],
    topk: int,
    iou_threshold: float,
    scales: list[float] | None = None,
    topk_ratio: float = 0.6,
    score_threshold: float = 0.35,
    margin_threshold: float = 0.03,
    confidence_iou_threshold: float = 0.3,
    cluster_query_topk: int = 1,
    cluster_iou_threshold: float = 0.3,
) -> tuple[dict[str, Any], np.ndarray]:
    """
    用多张小图在同一张大图中匹配，选出全局最优结果。

    返回：
    - global_result: 包含全局最优结果 + 所有 query 的排行榜
    - best_score_map: 全局最优 query 对应的热力图
    """
    # 关键优化：模型和大图 token 只初始化一次。
    extractor = build_default_extractor()
    original_big_image = load_rgb_image(big_image_path)
    big_info = extract_patch_tokens(extractor, original_big_image)

    all_query_results: list[dict[str, Any]] = []

    for idx, small_image_path in enumerate(small_image_paths, start=1):
        logger.info(
            "开始处理 query 图 %s / %s: %s",
            idx,
            len(small_image_paths),
            small_image_path,
        )

        try:
            result, score_map = match_one_query_with_precomputed_big(
                extractor=extractor,
                big_image_path=big_image_path,
                original_big_image=original_big_image,
                big_info=big_info,
                small_image_path=small_image_path,
                topk=topk,
                iou_threshold=iou_threshold,
                scales=scales,
                topk_ratio=topk_ratio,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
                confidence_iou_threshold=confidence_iou_threshold,
            )

            all_query_results.append(result)

        except Exception as exc:
            logger.exception("处理 query 图失败: %s", small_image_path)
            all_query_results.append(
                {
                    "small_image_path": str(Path(small_image_path).resolve()),
                    "query_image_name": Path(small_image_path).name,
                    "best_match": None,
                    "top_matches": [],
                    "error": str(exc),
                    "match_confidence": {
                        "is_match_confident": False,
                        "top1_score": None,
                        "top2_score": None,
                        "top1_top2_margin": None,
                        "score_threshold": score_threshold,
                        "margin_threshold": margin_threshold,
                        "reason": "exception",
                    },
                }
            )

    # 多图模式不再先做“可信 query 硬过滤”。
    # 低置信 query 仍然可以参与，但会在簇质量分里被自然降权，
    # 这样能避免把本来正确、但被重复框压低 margin 的好 query 直接丢掉。
    candidate_pool = [item for item in all_query_results if item.get("best_match") is not None]

    if not candidate_pool:
        raise ValueError("没有可用于聚类的有效 query 结果。")

    ranked_results = sorted(
        candidate_pool,
        key=get_global_ranking_key,
        reverse=True,
    )
    best_single_result = ranked_results[0]

    # 多图模式的最终选择策略改成“多张 query 竞争，单 query 最优优先”。
    # 聚类仍然保留，但只作为辅助说明，不再反过来覆盖最终 best_result。
    clusters = cluster_query_results_by_iou(
        query_results=candidate_pool,
        iou_threshold=cluster_iou_threshold,
        per_query_topk=cluster_query_topk,
    )
    cluster_summaries = summarize_clusters(clusters)
    leading_cluster = cluster_summaries[0] if cluster_summaries else None

    best_query_path = best_single_result["small_image_path"]
    best_query_box = best_single_result["best_match"]["pixel_box"]
    support_cluster = None
    for cluster in cluster_summaries:
        if any(
            member["query_result"]["small_image_path"] == best_query_path
            and member["selected_match"]["pixel_box"] == best_query_box
            for member in cluster["members"]
        ):
            support_cluster = cluster
            break

    global_best_result = dict(best_single_result)
    global_best_result["best_match"] = dict(best_single_result["best_match"])
    global_best_result["selection_strategy"] = "best_query"
    global_best_result["selected_from_query_top_rank"] = 1
    global_best_result["query_quality_score"] = compute_query_quality_score(best_single_result)
    global_best_result["best_scale_for_heatmap"] = global_best_result["best_match"]["scale"]
    if support_cluster is not None:
        global_best_result["consensus_info"] = {
            "cluster_id": support_cluster["cluster_id"],
            "support_count": support_cluster["support_count"],
            "quality_score": support_cluster["quality_score"],
            "avg_score": support_cluster["avg_score"],
            "avg_best_score_z": support_cluster["avg_best_score_z"],
            "avg_margin": support_cluster["avg_margin"],
            "avg_candidate_rank": support_cluster["avg_candidate_rank"],
            "member_query_names": support_cluster["member_query_names"],
            "consensus_box": support_cluster["consensus_box"],
            "cluster_best_pixel_box": global_best_result["best_match"]["pixel_box"],
            "cluster_best_query_top_rank": 1,
        }

    # 重新拿一次最终代表 query 的热力图，保证和最终 best_result 保持一致。
    _, global_best_score_map = match_one_query_with_precomputed_big(
        extractor=extractor,
        big_image_path=big_image_path,
        original_big_image=original_big_image,
        big_info=big_info,
        small_image_path=best_query_path,
        topk=topk,
        iou_threshold=iou_threshold,
        scales=scales,
        topk_ratio=topk_ratio,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        confidence_iou_threshold=confidence_iou_threshold,
    )

    ranking_summary = []
    for rank, item in enumerate(ranked_results, start=1):
        confidence = item["match_confidence"]
        best_match = item.get("best_match")
        score_stats = item.get("score_stats", {})

        ranking_summary.append(
            {
                "rank": rank,
                "query_image_name": item.get("query_image_name"),
                "small_image_path": item.get("small_image_path"),
                "is_match_confident": confidence["is_match_confident"],
                "score": None if best_match is None else best_match["score"],
                "best_score_z": score_stats.get("best_score_z"),
                "query_quality_score": compute_query_quality_score(item),
                "margin": confidence["top1_top2_margin"],
                "pixel_box": None if best_match is None else best_match["pixel_box"],
                "error": item.get("error"),
            }
        )

    global_result = {
        "big_image_path": str(Path(big_image_path).resolve()),
        "query_image_count": len(small_image_paths),
        "best_query_image": global_best_result["small_image_path"],
        "best_query_image_name": global_best_result["query_image_name"],
        "best_result": global_best_result,
        "selection_strategy": "best_query",
        "all_query_results": ranked_results,
        "ranking_summary": ranking_summary,
        "leading_cluster": None if leading_cluster is None else {
            "cluster_id": leading_cluster["cluster_id"],
            "support_count": leading_cluster["support_count"],
            "quality_score": leading_cluster["quality_score"],
            "avg_score": leading_cluster["avg_score"],
            "avg_best_score_z": leading_cluster["avg_best_score_z"],
            "avg_margin": leading_cluster["avg_margin"],
            "avg_candidate_rank": leading_cluster["avg_candidate_rank"],
            "consensus_box": leading_cluster["consensus_box"],
            "member_query_names": leading_cluster["member_query_names"],
        },
        "cluster_summary": [
            {
                "cluster_id": c["cluster_id"],
                "support_count": c["support_count"],
                "quality_score": c["quality_score"],
                "avg_score": c["avg_score"],
                "avg_best_score_z": c["avg_best_score_z"],
                "avg_margin": c["avg_margin"],
                "avg_candidate_rank": c["avg_candidate_rank"],
                "consensus_box": c["consensus_box"],
                "member_query_names": c["member_query_names"],
                "member_details": c["member_details"],
            }
            for c in cluster_summaries
        ],
    }

    return global_result, global_best_score_map


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
        default=None,
        help="单张小图路径，与 --small-image-dir 二选一",
    )
    parser.add_argument(
        "--small-image-dir",
        default=None,
        help="小图目录路径，与 --small-image 二选一",
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
    parser.add_argument(
        "--confidence-iou-threshold",
        type=float,
        default=0.3,
        help="可信度判断时，top1 与次优候选之间允许的最大重叠 IoU；越小越能过滤重复框",
    )
    parser.add_argument(
        "--cluster-query-topk",
        type=int,
        default=1,
        help="多图聚类时，每张 query 参与聚类的 top 候选数量；默认只用 top1，避免次优候选把结果拉偏",
    )
    parser.add_argument(
        "--cluster-iou-threshold",
        type=float,
        default=0.3,
        help="多图聚类时的 IoU 阈值",
    )
    parser.add_argument(
        "--ranking-json",
        default="query_ranking_result.json",
        help="多 query 图排行榜结果 JSON 输出路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    small_image_paths = collect_small_image_paths(
        small_image=args.small_image,
        small_image_dir=args.small_image_dir,
    )

    logger.info("待匹配的小图数量: %s", len(small_image_paths))

    if len(small_image_paths) == 1:
        # 单图模式：兼容原有逻辑
        result, score_map = find_similar_region_with_tokens(
            big_image_path=args.big_image,
            small_image_path=small_image_paths[0],
            topk=args.topk,
            iou_threshold=args.iou_threshold,
            scales=args.scales,
            topk_ratio=args.topk_ratio,
            score_threshold=args.score_threshold,
            margin_threshold=args.margin_threshold,
            confidence_iou_threshold=args.confidence_iou_threshold,
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
            best_box=result["best_match"]["pixel_box"],
        )

        print("\n===== 最佳匹配结果（单图模式） =====")
        print(f"query 图: {result['small_image_path']}")
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

    else:
        # 多图模式：文件夹里多张小图竞争
        global_result, best_score_map = find_best_match_from_multiple_queries(
            big_image_path=args.big_image,
            small_image_paths=small_image_paths,
            topk=args.topk,
            iou_threshold=args.iou_threshold,
            scales=args.scales,
            topk_ratio=args.topk_ratio,
            score_threshold=args.score_threshold,
            margin_threshold=args.margin_threshold,
            confidence_iou_threshold=args.confidence_iou_threshold,
            cluster_query_topk=args.cluster_query_topk,
            cluster_iou_threshold=args.cluster_iou_threshold,
        )

        # 保存总结果
        ranking_json_path = Path(args.ranking_json)
        with open(ranking_json_path, "w", encoding="utf-8") as file:
            json.dump(global_result, file, ensure_ascii=False, indent=2)
        logger.info("多 query 排行榜 JSON 已保存: %s", ranking_json_path.resolve())

        # 同时把全局最优单独保存一份，便于兼容之前的消费逻辑
        best_result_json_path = Path(args.result_json)
        with open(best_result_json_path, "w", encoding="utf-8") as file:
            json.dump(global_result["best_result"], file, ensure_ascii=False, indent=2)
        logger.info("全局最优结果 JSON 已保存: %s", best_result_json_path.resolve())

        best_result = global_result["best_result"]

        draw_best_box(
            big_image_path=args.big_image,
            best_box=best_result["best_match"]["pixel_box"],
            output_path=args.vis_output,
        )
        save_heatmap_images(
            big_image_path=args.big_image,
            score_map=best_score_map,
            heatmap_output_path=args.heatmap_output,
            overlay_output_path=args.overlay_output,
            best_box=best_result["best_match"]["pixel_box"],
        )

        print("\n===== 最佳匹配结果（多图模式） =====")
        print(f"参与匹配的小图数量: {global_result['query_image_count']}")
        print(f"选择策略: {global_result['selection_strategy']}")
        print(f"全局最优 query 图: {global_result['best_query_image']}")
        print(f"最佳 token 框: {best_result['best_match']['token_box']}")
        print(f"最佳像素框: {best_result['best_match']['pixel_box']}")
        print(f"最佳相似度: {best_result['best_match']['score']:.6f}")
        print(f"query_quality_score: {best_result.get('query_quality_score')}")
        print(f"最佳尺度下的小图尺寸: {best_result['best_match']['scaled_small_image_size']}")
        print(f"最佳尺度下的小图 token 网格: {best_result['best_match']['small_token_grid']}")
        print(f"热力图输出: {Path(args.heatmap_output).resolve()}")
        print(f"热力图叠加输出: {Path(args.overlay_output).resolve()}")
        print(f"排行榜 JSON: {ranking_json_path.resolve()}")

        confidence = best_result["match_confidence"]
        consensus = best_result.get("consensus_info")
        if consensus is not None:
            print("\n===== 共识簇信息 =====")
            print(f"cluster_id: {consensus['cluster_id']}")
            print(f"support_count: {consensus['support_count']}")
            print(f"quality_score: {consensus['quality_score']}")
            print(f"avg_score: {consensus['avg_score']}")
            print(f"avg_best_score_z: {consensus['avg_best_score_z']}")
            print(f"avg_margin: {consensus['avg_margin']}")
            print(f"avg_candidate_rank: {consensus['avg_candidate_rank']}")
            print(f"consensus_box: {consensus['consensus_box']}")
            print(f"cluster_best_pixel_box: {consensus['cluster_best_pixel_box']}")
            print(f"cluster_best_query_top_rank: {consensus['cluster_best_query_top_rank']}")
            print(f"member_query_names: {consensus['member_query_names']}")
        print("\n===== 共识簇 Top 3 =====")
        for item in global_result["cluster_summary"][:3]:
            print(
                f"cluster_id={item['cluster_id']}, "
                f"support_count={item['support_count']}, "
                f"quality={item['quality_score']}, "
                f"avg_z={item['avg_best_score_z']}, "
                f"avg_margin={item['avg_margin']}, "
                f"avg_score={item['avg_score']}, "
                f"avg_rank={item['avg_candidate_rank']}, "
                f"box={item['consensus_box']}, "
                f"members={item['member_query_names']}"
            )
        print("\n===== 全局最优匹配可信度判断 =====")
        print(f"是否可信匹配: {confidence['is_match_confident']}")
        print(f"top1 分数: {confidence['top1_score']}")
        print(f"top2 分数: {confidence['top2_score']}")
        print(f"top1-top2 分差: {confidence['top1_top2_margin']}")
        print(f"score_threshold: {confidence['score_threshold']}")
        print(f"margin_threshold: {confidence['margin_threshold']}")
        print(f"判定原因: {confidence['reason']}")
        print("\n===== Query 排行榜 Top 3 =====")
        for item in global_result["ranking_summary"][:3]:
            print(
                f"rank={item['rank']}, "
                f"query={item['query_image_name']}, "
                f"confident={item['is_match_confident']}, "
                f"score={item['score']}, "
                f"z={item['best_score_z']}, "
                f"quality={item['query_quality_score']}, "
                f"margin={item['margin']}, "
                f"box={item['pixel_box']}, "
                f"error={item['error']}"
            )


if __name__ == "__main__":
    main()


"""
单图模式
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
"""
多图模式
python find_similar_region_with_dino_tokens.py \
  --big-image /home/ubuntu/yzm_workspace/compare_embedding/frames_456/test01.jpg \
  --small-image-dir /home/ubuntu/yzm_workspace/compare_embedding/images \
  --topk 5 \
  --iou-threshold 0.3 \
  --scales 0.75 0.9 1.0 1.1 1.25 \
  --topk-ratio 0.6 \
  --score-threshold 0.35 \
  --margin-threshold 0.03 \
  --ranking-json query_ranking_result.json
"""
