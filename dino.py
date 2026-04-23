import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


logger = logging.getLogger(__name__)
DEFAULT_DINO_MODEL_DIR = "/home/ubuntu/.cache/modelscope/hub/models/facebook/dinov3-vitl16-pretrain-lvd1689m"


class DinoV3Extractor:
    """
    使用 HuggingFace 格式的 DINOv3 提取图像 embedding
    """

    def __init__(self, model_path):
        # 选择设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 检查模型路径
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        print("加载 DINOv3 模型（HuggingFace 格式）...")

        # 加载预处理器（负责 resize / normalize 等）
        self.processor = AutoImageProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )

        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        ).to(self.device)

        self.model.eval()
        print("模型加载完成")

    @torch.no_grad()
    def extract_from_image(self, image: Image.Image) -> np.ndarray:
        """
        直接从 PIL Image 提取 embedding。
        这样接口服务在内存中裁剪后可以直接复用，不需要先落盘。
        """
        rgb_image = image.convert("RGB")

        # 预处理
        inputs = self.processor(
            images=rgb_image,
            return_tensors="pt"
        ).to(self.device)

        # 前向推理
        outputs = self.model(**inputs)

        # 取 CLS token（全局特征）
        feat = outputs.last_hidden_state[:, 0, :]

        # L2 归一化（用于相似度计算）
        feat = F.normalize(feat, dim=-1)

        return feat.squeeze(0).cpu().numpy().astype("float32")

    @torch.no_grad()
    def extract_single(self, image_path: str) -> np.ndarray:
        """
        提取单张图片 embedding
        """
        with Image.open(image_path) as image:
            return self.extract_from_image(image)

    @torch.no_grad()
    def extract_batch(self, image_paths, batch_size=8):
        """
        批量提取 embedding（推荐）
        """
        all_features = []
        valid_paths = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"读取失败: {path}, 错误: {e}")

            if len(images) == 0:
                continue

            # 预处理
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)

            # 推理
            outputs = self.model(**inputs)

            # CLS token
            feat = outputs.last_hidden_state[:, 0, :]

            # 归一化
            feat = F.normalize(feat, dim=-1)

            all_features.append(feat.cpu().numpy().astype("float32"))

        if len(all_features) == 0:
            return None, []

        features = np.concatenate(all_features, axis=0)
        return features, valid_paths


def resolve_local_dino_model_dir() -> str:
    """
    解析本地 DINO 模型目录。

    规则：
    - 优先读取环境变量 DINO_MODEL_DIR
    - 如果未设置，则使用默认目录 DEFAULT_DINO_MODEL_DIR
    - 不再自动搜索多个缓存目录，避免路径来源不明确
    """
    model_dir = os.getenv("DINO_MODEL_DIR", DEFAULT_DINO_MODEL_DIR).strip()
    if not model_dir:
        raise ValueError(
            "DINO_MODEL_DIR 为空，且默认模型目录也为空，请检查配置。"
        )

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            "DINO 模型目录不存在。\n"
            f"当前使用目录: {model_dir}\n"
            "请设置环境变量 DINO_MODEL_DIR，或确认默认目录是否正确。"
        )

    has_config = os.path.exists(os.path.join(model_dir, "config.json"))
    has_weight = (
        os.path.exists(os.path.join(model_dir, "model.safetensors"))
        or os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
    )
    if not has_config or not has_weight:
        raise FileNotFoundError(
            "DINO 模型目录缺少必要文件。\n"
            f"当前使用目录: {model_dir}\n"
            f"config.json 存在: {has_config}\n"
            f"权重文件存在: {has_weight}"
        )

    logger.info("使用本地 DINO 模型目录: %s", model_dir)
    return model_dir


def build_default_extractor() -> DinoV3Extractor:
    """
    使用默认本地模型目录创建提取器。
    """
    model_dir = resolve_local_dino_model_dir()
    return DinoV3Extractor(model_dir)


# =========================
# 运行示例
# =========================
if __name__ == "__main__":

    # 默认读取环境变量 DINO_MODEL_DIR，未设置时回落到 DEFAULT_DINO_MODEL_DIR
    model_path = resolve_local_dino_model_dir()

    extractor = DinoV3Extractor(model_path)

    # ===== 单张测试 =====
    print("\n===== 单张图片测试 =====")
    test_image = "test.jpg"

    if os.path.exists(test_image):
        feat = extractor.extract_single(test_image)
        print("embedding shape:", feat.shape)
        print("前5个值:", feat[:5])
    else:
        print("请放一张 test.jpg 在当前目录")

    # ===== 批量测试 =====
    print("\n===== 批量图片测试 =====")

    image_dir = "images"

    if os.path.isdir(image_dir):
        image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        features, paths = extractor.extract_batch(image_paths, batch_size=8)

        if features is not None:
            print("图片数量:", len(paths))
            print("embedding shape:", features.shape)

            # =========================
            # ✅ 保存 embedding
            # =========================

            # 1️⃣ 保存所有向量
            np.save("features.npy", features)
            print("已保存: features.npy")

            # 2️⃣ 保存对应图片路径（很重要）
            with open("paths.txt", "w") as f:
                for p in paths:
                    f.write(p + "\n")
            print("已保存: paths.txt")

        else:
            print("没有有效图片")
    else:
        print("请创建 images 文件夹并放入图片")
