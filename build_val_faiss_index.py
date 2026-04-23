import argparse
import json
from pathlib import Path

import faiss
import numpy as np


def load_feature_bundle(feature_path: str, mapping_path: str) -> tuple[np.ndarray, dict[int, str]]:
    """
    从磁盘加载特征矩阵和路径映射。
    这是纯 FAISS 环境使用的入口，不依赖任何 DINO 逻辑。
    """
    feature_file = Path(feature_path)
    mapping_file = Path(mapping_path)

    if not feature_file.exists():
        raise FileNotFoundError(f"特征文件不存在: {feature_file}")
    if not mapping_file.exists():
        raise FileNotFoundError(f"路径映射文件不存在: {mapping_file}")

    features = np.load(str(feature_file)).astype("float32")
    if features.ndim != 2:
        raise ValueError(f"特征矩阵必须是二维，实际 shape={features.shape}")

    with open(mapping_file, "r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    id_to_path = {int(key): str(value) for key, value in raw_mapping.items()}
    if len(id_to_path) != features.shape[0]:
        raise ValueError(
            "路径映射数量和特征数量不一致: "
            f"mapping={len(id_to_path)}, features={features.shape[0]}"
        )

    print(f"特征文件加载完成: {feature_file}")
    print(f"路径映射加载完成: {mapping_file}")
    print(f"特征数量: {features.shape[0]}，特征维度: {features.shape[1]}")
    return features, id_to_path


def create_faiss_index(features: np.ndarray, use_gpu: bool = True):
    """
    根据特征矩阵建立 FAISS 索引。
    这里默认使用内积索引，并在检索前后统一做 L2 归一化。
    """
    feature_matrix = features.astype("float32", copy=True)
    faiss.normalize_L2(feature_matrix)

    dim = feature_matrix.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)

    gpu_available = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    if use_gpu and gpu_available:
        print(f"检测到 {faiss.get_num_gpus()} 张 GPU，使用 GPU 建立索引")
        resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
        gpu_index.add(feature_matrix)
        return gpu_index, cpu_index, resources

    print("未使用 GPU 索引，回退到 CPU 建立索引")
    cpu_index.add(feature_matrix)
    return cpu_index, cpu_index, None


def save_outputs(
    output_dir: str,
    cpu_index: faiss.Index,
    id_to_path: dict[int, str],
    features: np.ndarray,
):
    """
    保存 FAISS 索引、路径映射和特征矩阵。
    保持与当前 FAISS 服务约定一致：
    - val.faiss
    - id_to_path.json
    - features.npy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_path = output_path / "val.faiss"
    mapping_path = output_path / "id_to_path.json"
    features_path = output_path / "features.npy"

    faiss.write_index(cpu_index, str(index_path))
    with open(mapping_path, "w", encoding="utf-8") as file:
        json.dump(
            {str(idx): path for idx, path in sorted(id_to_path.items())},
            file,
            ensure_ascii=False,
            indent=2,
        )
    np.save(str(features_path), features)

    print(f"索引已保存: {index_path}")
    print(f"路径映射已保存: {mapping_path}")
    print(f"特征矩阵已保存: {features_path}")


def build_faiss_index_from_features(
    feature_path: str,
    mapping_path: str,
    output_dir: str,
    use_gpu: bool,
):
    """
    纯 FAISS 建库主流程：
    1. 加载特征矩阵
    2. 加载路径映射
    3. 建立索引
    4. 保存索引产物
    """
    print(f"FAISS GPU 数量: {faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 0}")
    features, id_to_path = load_feature_bundle(feature_path, mapping_path)
    _, cpu_index, _ = create_faiss_index(features, use_gpu=use_gpu)

    print(f"索引建立完成，向量数: {cpu_index.ntotal}，特征维度: {features.shape[1]}")
    save_outputs(
        output_dir=output_dir,
        cpu_index=cpu_index,
        id_to_path=id_to_path,
        features=features,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="纯 FAISS 环境建库脚本：从 features.npy 和 id_to_path.json 建立 val.faiss"
    )
    parser.add_argument(
        "--feature-path",
        default="./features.npy",
        help="DINO 环境导出的特征矩阵文件路径",
    )
    parser.add_argument(
        "--mapping-path",
        default="./id_to_path.json",
        help="DINO 环境导出的路径映射文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default="./faiss_val_store",
        help="索引和映射文件输出目录",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="强制只使用 CPU 建索引",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_faiss_index_from_features(
        feature_path=args.feature_path,
        mapping_path=args.mapping_path,
        output_dir=args.output_dir,
        use_gpu=not args.cpu_only,
    )


if __name__ == "__main__":
    main()
