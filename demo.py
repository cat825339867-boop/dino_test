import argparse
import logging
from pathlib import Path

from PIL import Image

from dino import build_default_extractor
from find_similar_region_with_dino_tokens import (
    compute_region_score,
    compute_similarity_map,
    extract_patch_tokens,
    resize_small_image_with_scale,
)


# 脚本日志尽量保持简洁，既能看到处理进度，也便于排查图片读取或模型初始化问题。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("small_image_similarity_demo")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    """
    判断文件是否为常见图片格式。
    """
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def validate_image_path(image_path: str, arg_name: str) -> Path:
    """
    校验单张图片路径，避免后续模型提特征时才暴露输入错误。
    """
    resolved_path = Path(image_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{arg_name} 指定的图片不存在: {resolved_path}")
    if resolved_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"{arg_name} 不是受支持的图片格式: {resolved_path}")
    return resolved_path


def collect_image_paths(image_dir: str) -> list[Path]:
    """
    收集目录下全部图片并按文件名排序，方便结果稳定复现。
    """
    resolved_dir = Path(image_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"图片目录不存在: {resolved_dir}")

    image_paths = sorted(
        path for path in resolved_dir.iterdir()
        if is_image_file(path)
    )
    if not image_paths:
        raise ValueError(f"目录下没有可用图片: {resolved_dir}")
    return image_paths


def load_rgb_image(image_path: Path) -> Image.Image:
    """
    读取图片并统一转换为 RGB。
    """
    with Image.open(image_path) as image:
        return image.convert("RGB")


def compute_patch_similarity_from_token_infos(
    image_a_token_info: dict,
    image_b_token_info: dict,
    topk_ratio: float,
) -> dict[str, float | str | dict]:
    """
    使用 patch token 比较两张图片的相似度。

    规则：
    - 如果两张图的 token 网格完全一致，则逐 patch 对齐比较
    - 如果网格不一致，则让较小网格在较大网格上滑窗，取最佳候选分数
    - 输出里会额外标明本次比较采用的是哪种模式，便于后续分析结果
    """
    image_a_tokens = image_a_token_info["tokens"]
    image_b_tokens = image_b_token_info["tokens"]
    grid_a = image_a_token_info["grid_size"]
    grid_b = image_b_token_info["grid_size"]

    if image_a_tokens.shape == image_b_tokens.shape:
        similarity = compute_region_score(
            region=image_a_tokens,
            query_tokens=image_b_tokens,
            topk_ratio=topk_ratio,
        )
        return {
            "similarity": float(similarity),
            "comparison_mode": "aligned_patch_grid",
            "big_grid": dict(grid_a),
            "query_grid": dict(grid_b),
        }

    area_a = int(grid_a["height"]) * int(grid_a["width"])
    area_b = int(grid_b["height"]) * int(grid_b["width"])

    if area_a >= area_b:
        big_info = image_a_token_info
        query_info = image_b_token_info
        big_side = "image_a"
        query_side = "image_b"
    else:
        big_info = image_b_token_info
        query_info = image_a_token_info
        big_side = "image_b"
        query_side = "image_a"

    candidates, _ = compute_similarity_map(
        big_tokens=big_info["tokens"],
        query_tokens=query_info["tokens"],
        topk_ratio=topk_ratio,
    )
    if not candidates:
        raise ValueError("patch 滑窗比较未生成任何候选结果")

    best_candidate = max(candidates, key=lambda item: item["score"])
    return {
        "similarity": float(best_candidate["score"]),
        "comparison_mode": "sliding_patch_window",
        "big_image_side": big_side,
        "query_image_side": query_side,
        "big_grid": dict(big_info["grid_size"]),
        "query_grid": dict(query_info["grid_size"]),
        "best_token_box": list(best_candidate["token_box"]),
    }


def compare_two_images(
    image_a_path: Path,
    image_b_path: Path,
    scales: list[float],
    topk_ratio: float,
) -> dict[str, float | str | dict]:
    """
    比较两张图片的 patch feature 相似度并返回最佳结果。

    说明：
    - 固定 `image_a` 作为参考图
    - 对 `image_b` 按多尺度缩放后逐个比较
    - 最终返回分数最高的那一个尺度结果
    """
    logger.info("开始初始化 DINO 特征提取器")
    extractor = build_default_extractor()
    logger.info("DINO 特征提取器初始化完成，开始提取 patch token")

    image_a = load_rgb_image(image_a_path)
    image_b = load_rgb_image(image_b_path)
    image_a_token_info = extract_patch_tokens(extractor, image_a)

    best_result: dict[str, float | str | dict] | None = None
    for scale in scales:
        scaled_image_b = resize_small_image_with_scale(image_b, scale=scale)
        image_b_token_info = extract_patch_tokens(extractor, scaled_image_b)
        current_result = compute_patch_similarity_from_token_infos(
            image_a_token_info=image_a_token_info,
            image_b_token_info=image_b_token_info,
            topk_ratio=topk_ratio,
        )
        current_result.update({
            "image_a": str(image_a_path),
            "image_b": str(image_b_path),
            "image_a_size": dict(image_a_token_info["image_size"]),
            "image_b_original_size": {
                "width": int(image_b.size[0]),
                "height": int(image_b.size[1]),
            },
            "image_b_scaled_size": dict(image_b_token_info["image_size"]),
            "image_b_scale": float(scale),
        })

        logger.info(
            "patch 对比完成 | image_a=%s | image_b=%s | scale=%.3f | mode=%s | similarity=%.6f",
            image_a_path.name,
            image_b_path.name,
            scale,
            current_result["comparison_mode"],
            current_result["similarity"],
        )

        if best_result is None or float(current_result["similarity"]) > float(best_result["similarity"]):
            best_result = current_result

    if best_result is None:
        raise RuntimeError("两张图片 patch 对比失败，未生成有效结果")

    return best_result


def compare_image_with_directory(
    query_image_path: Path,
    candidate_image_paths: list[Path],
    topk: int,
    scales: list[float],
    topk_ratio: float,
) -> list[dict[str, float | str | dict]]:
    """
    比较一张查询图和目录中所有候选图的 patch 相似度，并按分数倒序输出。
    """
    ranking_results: list[dict[str, float | str | dict]] = []
    for candidate_image_path in candidate_image_paths:
        result = compare_two_images(
            image_a_path=query_image_path,
            image_b_path=candidate_image_path,
            scales=scales,
            topk_ratio=topk_ratio,
        )
        ranking_results.append({
            "query_image": str(query_image_path),
            "candidate_image": str(candidate_image_path.resolve()),
            "similarity": result["similarity"],
            "comparison_mode": result["comparison_mode"],
            "image_b_scale": result["image_b_scale"],
            "best_token_box": result.get("best_token_box"),
            "big_image_side": result.get("big_image_side"),
            "query_image_side": result.get("query_image_side"),
        })

    ranking_results.sort(key=lambda item: item["similarity"], reverse=True)
    return ranking_results[:topk]


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    使用规则：
    - `--image-a` 必填
    - `--image-b` 和 `--image-dir` 二选一
    """
    parser = argparse.ArgumentParser(
        description="使用当前仓库的 DINO 特征提取能力比较两张小图相似度"
    )
    parser.add_argument(
        "--image-a",
        required=True,
        help="第一张图片路径；如果配合 --image-dir 使用，则它表示查询图",
    )
    parser.add_argument(
        "--image-b",
        default=None,
        help="第二张图片路径，与 --image-dir 二选一",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="候选图片目录，与 --image-b 二选一",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="目录比对模式下返回前 topk 条结果，默认 5",
    )
    parser.add_argument(
        "--scales",
        default="0.75,0.9,1.0,1.1,1.25",
        help="对第二张图或候选图使用的多尺度列表，逗号分隔",
    )
    parser.add_argument(
        "--topk-ratio",
        type=float,
        default=0.6,
        help="patch 比较时仅保留 top-k patch 分数做平均，建议 0.3~1.0",
    )
    return parser.parse_args()


def main() -> None:
    """
    脚本入口。
    """
    args = parse_args()

    if bool(args.image_b) == bool(args.image_dir):
        raise ValueError("必须且只能提供一个参数：--image-b 或 --image-dir")
    if not (0 < float(args.topk_ratio) <= 1.0):
        raise ValueError("--topk-ratio 必须在 (0, 1] 范围内")

    image_a_path = validate_image_path(args.image_a, "--image-a")
    scales = [
        float(item.strip())
        for item in str(args.scales).split(",")
        if item.strip()
    ]
    if not scales:
        raise ValueError("--scales 不能为空")

    if args.image_b:
        image_b_path = validate_image_path(args.image_b, "--image-b")
        result = compare_two_images(
            image_a_path=image_a_path,
            image_b_path=image_b_path,
            scales=scales,
            topk_ratio=float(args.topk_ratio),
        )

        print("\n===== 两张图片 Patch 相似度结果 =====")
        print(f"image_a: {result['image_a']}")
        print(f"image_b: {result['image_b']}")
        print(f"comparison_mode: {result['comparison_mode']}")
        print(f"image_b_scale: {result['image_b_scale']:.3f}")
        print(f"image_a_size: {result['image_a_size']}")
        print(f"image_b_original_size: {result['image_b_original_size']}")
        print(f"image_b_scaled_size: {result['image_b_scaled_size']}")
        print(f"big_grid: {result['big_grid']}")
        print(f"query_grid: {result['query_grid']}")
        if result.get("best_token_box") is not None:
            print(f"best_token_box: {result['best_token_box']}")
        print(f"similarity: {result['similarity']:.6f}")
        return

    candidate_image_paths = collect_image_paths(args.image_dir)
    ranking_results = compare_image_with_directory(
        query_image_path=image_a_path,
        candidate_image_paths=candidate_image_paths,
        topk=max(1, int(args.topk)),
        scales=scales,
        topk_ratio=float(args.topk_ratio),
    )

    print("\n===== 图片目录 Patch 相似度 TopK 结果 =====")
    print(f"query_image: {image_a_path}")
    print(f"candidate_count: {len(candidate_image_paths)}")
    for index, item in enumerate(ranking_results, start=1):
        print(
            f"{index}. candidate={item['candidate_image']}, "
            f"similarity={item['similarity']:.6f}, "
            f"mode={item['comparison_mode']}, "
            f"scale={float(item['image_b_scale']):.3f}, "
            f"best_token_box={item['best_token_box']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("小图相似度脚本执行失败")
        raise SystemExit(f"执行失败: {exc}") from exc


# python demo.py --image-a need_compare/002.png --image-dir need_compare --topk 7
# python demo.py --image-a /home/ubuntu/yzm_workspace/compare_embedding/need_compare/002.png --image-b /home/ubuntu/yzm_workspace/compare_embedding/need_compare/001.png --topk 2