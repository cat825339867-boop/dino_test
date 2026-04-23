import argparse
import json
from pathlib import Path

import numpy as np

from dino import build_default_extractor


# 支持的图片后缀
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_files(root_dir: str) -> list[str]:
    """
    递归收集目录下全部图片文件。
    这是纯 DINO 环境使用的入口，不依赖任何 FAISS 逻辑。
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {root_dir}")

    image_files: list[str] = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS:
            image_files.append(str(file_path.resolve()))

    image_files.sort()
    return image_files


def save_feature_bundle(output_dir: str, features: np.ndarray, image_paths: list[str]):
    """
    保存 DINO 环境导出的标准产物，供 FAISS 环境继续建库：
    - features.npy
    - id_to_path.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_path = output_path / "features.npy"
    mapping_path = output_path / "id_to_path.json"

    np.save(str(feature_path), features)
    with open(mapping_path, "w", encoding="utf-8") as file:
        json.dump(
            {str(idx): path for idx, path in enumerate(image_paths)},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"特征矩阵已保存: {feature_path}")
    print(f"路径映射已保存: {mapping_path}")


def export_features_for_val(val_dir: str, output_dir: str, batch_size: int):
    """
    纯 DINO 导出主流程：
    1. 扫描图片目录
    2. 提取 embedding
    3. 保存 features.npy 和 id_to_path.json
    """
    print(f"开始扫描图片目录: {val_dir}")
    image_paths = collect_image_files(val_dir)
    if not image_paths:
        raise ValueError(f"目录中未找到任何图片: {val_dir}")

    print(f"共找到 {len(image_paths)} 张图片")
    extractor = build_default_extractor()
    features, valid_paths = extractor.extract_batch(image_paths, batch_size=batch_size)

    if features is None or not valid_paths:
        raise RuntimeError("没有任何图片成功提取特征，无法导出特征文件")

    if len(valid_paths) != features.shape[0]:
        raise RuntimeError(
            f"有效图片数量和特征数量不一致: paths={len(valid_paths)}, features={features.shape[0]}"
        )

    print(f"特征提取完成，特征数量: {features.shape[0]}，特征维度: {features.shape[1]}")
    save_feature_bundle(
        output_dir=output_dir,
        features=features.astype("float32"),
        image_paths=valid_paths,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="纯 DINO 环境特征导出脚本：从图片目录导出 features.npy 和 id_to_path.json"
    )
    parser.add_argument(
        "--val-dir",
        default="./val",
        help="待导出特征的图片目录",
    )
    parser.add_argument(
        "--output-dir",
        default="./feature_export_store",
        help="特征矩阵和路径映射输出目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批量提取特征时的 batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_features_for_val(
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
