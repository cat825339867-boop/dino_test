import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from dino import build_default_extractor


# 配置日志，便于独立排查 DINO 服务问题
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("dino_service")


class BoxItem(BaseModel):
    """单个候选框参数。"""

    x1: int = Field(..., description="左上角 x 坐标")
    y1: int = Field(..., description="左上角 y 坐标")
    x2: int = Field(..., description="右下角 x 坐标")
    y2: int = Field(..., description="右下角 y 坐标")


class ExtractFeatureRequest(BaseModel):
    """DINO 特征提取请求。"""

    image_path: str = Field(..., description="待处理图片路径")
    boxes: list[BoxItem] = Field(..., min_length=1, description="待裁剪的框列表")


def crop_image_by_box(image: Image.Image, box: BoxItem) -> Image.Image:
    """
    按照候选框从原图中裁剪子图。
    这里只做坐标处理和裁图，不涉及任何检索逻辑。
    """
    width, height = image.size
    x1 = max(0, min(box.x1, width - 1))
    y1 = max(0, min(box.y1, height - 1))
    x2 = max(0, min(box.x2, width))
    y2 = max(0, min(box.y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"裁剪框无效: {(box.x1, box.y1, box.x2, box.y2)}")

    return image.crop((x1, y1, x2, y2))


@lru_cache(maxsize=1)
def get_feature_extractor():
    """
    缓存 DINO 提取器，避免重复加载模型。
    """
    logger.info("开始初始化 DINO 特征提取器")
    extractor = build_default_extractor()
    logger.info("DINO 特征提取器初始化完成")
    return extractor


app = FastAPI(
    title="DINO 特征提取服务",
    description="只负责根据图片和框列表裁剪并提取 DINOv3 特征，不包含任何 FAISS 检索逻辑。",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/dino/extract-features")
def extract_features(request_body: ExtractFeatureRequest) -> dict[str, Any]:
    """
    根据图片路径和框列表提取特征。
    返回每个框的特征向量，供上游编排服务继续调用 FAISS 服务。
    """
    logger.info(
        "收到 DINO 特征提取请求: image_path=%s, box_count=%s",
        request_body.image_path,
        len(request_body.boxes),
    )

    extractor = get_feature_extractor()

    try:
        with Image.open(request_body.image_path) as image:
            rgb_image = image.convert("RGB")
            image_width, image_height = rgb_image.size
            results: list[dict[str, Any]] = []

            for index, box in enumerate(request_body.boxes):
                try:
                    crop_image = crop_image_by_box(rgb_image, box)
                    feature = extractor.extract_from_image(crop_image)

                    results.append({
                        "box_index": index,
                        "box": box.model_dump(),
                        "feature": feature.tolist(),
                        "feature_dim": int(feature.shape[0]),
                        "crop_size": {
                            "width": crop_image.size[0],
                            "height": crop_image.size[1],
                        },
                    })
                except Exception as exc:
                    logger.exception("提取第 %s 个框的特征失败", index)
                    results.append({
                        "box_index": index,
                        "box": box.model_dump(),
                        "feature": None,
                        "feature_dim": None,
                        "crop_size": None,
                        "error_message": str(exc),
                    })

        return {
            "image_path": request_body.image_path,
            "image_size": {"width": image_width, "height": image_height},
            "results": results,
        }
    except FileNotFoundError as exc:
        logger.exception("图片不存在")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("DINO 特征提取失败")
        raise HTTPException(status_code=500, detail=f"DINO 特征提取失败: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_dino_service:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
    )
