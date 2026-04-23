import json
import logging
import os
from functools import lru_cache
from typing import Any
from urllib import error, request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# Box 服务只做编排，不直接依赖 DINO 或 FAISS 的本地实现
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("box_match_service")


DEFAULT_THRESHOLD = 0.65
DEFAULT_TOPK = 1
DEFAULT_DINO_SERVICE_URL = "http://127.0.0.1:8002"
DEFAULT_FAISS_SERVICE_URL = "http://127.0.0.1:8001"


class BoxItem(BaseModel):
    """单个候选框参数。"""

    x1: int = Field(..., description="左上角 x 坐标")
    y1: int = Field(..., description="左上角 y 坐标")
    x2: int = Field(..., description="右下角 x 坐标")
    y2: int = Field(..., description="右下角 y 坐标")

    @field_validator("x2")
    @classmethod
    def validate_x2(cls, value: int, info):
        x1 = info.data.get("x1")
        if x1 is not None and value <= x1:
            raise ValueError("x2 必须大于 x1")
        return value

    @field_validator("y2")
    @classmethod
    def validate_y2(cls, value: int, info):
        y1 = info.data.get("y1")
        if y1 is not None and value <= y1:
            raise ValueError("y2 必须大于 y1")
        return value


class MatchRequest(BaseModel):
    """框匹配请求。"""

    image_path: str = Field(..., description="待处理图片的本地路径")
    label_name: str = Field(..., description="标签名称")
    boxes: list[BoxItem] = Field(..., min_length=1, description="待校验的候选框列表")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="判定正确的相似度阈值")
    topk: int = Field(DEFAULT_TOPK, ge=1, le=20, description="向量库返回的候选数量")

    @field_validator("image_path", "label_name")
    @classmethod
    def validate_not_empty(cls, value: str):
        if not value or not value.strip():
            raise ValueError("字段不能为空")
        return value.strip()


class DinoExtractRequest(BaseModel):
    """发给 DINO 服务的请求。"""

    image_path: str
    boxes: list[BoxItem]


class FaissSearchRequest(BaseModel):
    """发给 FAISS 服务的检索请求。"""

    label_name: str
    feature: list[float]
    topk: int


class RemoteJsonClient:
    """
    统一的远程 JSON 客户端。
    这里只封装 HTTP 调用，不内嵌 DINO/FAISS 的业务逻辑。
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            logger.exception("远程服务返回 HTTP 错误: url=%s, status=%s, body=%s", url, exc.code, body)
            raise RuntimeError(f"调用远程服务失败，HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            logger.exception("远程服务不可达: url=%s", url)
            raise RuntimeError(f"远程服务不可达: {exc}") from exc
        except Exception as exc:
            logger.exception("调用远程服务异常: url=%s", url)
            raise RuntimeError(f"调用远程服务失败: {exc}") from exc


@lru_cache(maxsize=1)
def get_dino_client() -> RemoteJsonClient:
    """获取 DINO 服务客户端。"""
    base_url = os.getenv("DINO_SERVICE_URL", DEFAULT_DINO_SERVICE_URL)
    logger.info("使用 DINO 服务地址: %s", base_url)
    return RemoteJsonClient(base_url)


@lru_cache(maxsize=1)
def get_faiss_client() -> RemoteJsonClient:
    """获取 FAISS 服务客户端。"""
    base_url = os.getenv("FAISS_SERVICE_URL", DEFAULT_FAISS_SERVICE_URL)
    logger.info("使用 FAISS 服务地址: %s", base_url)
    return RemoteJsonClient(base_url)


app = FastAPI(
    title="框选结果编排服务",
    description="只负责调用独立的 DINO 服务和 FAISS 服务，汇总每个框的正确性判断。",
    version="3.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/match-boxes")
def match_boxes(request_body: MatchRequest) -> dict[str, Any]:
    """
    Box 服务主流程：
    1. 调 DINO 服务提取每个框的特征
    2. 调 FAISS 服务对每个特征做检索
    3. 基于阈值汇总每个框的正确/错误结果
    """
    image_path = os.path.abspath(request_body.image_path)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_path}")

    logger.info(
        "收到框匹配请求: image_path=%s, label_name=%s, box_count=%s, threshold=%.4f",
        image_path,
        request_body.label_name,
        len(request_body.boxes),
        request_body.threshold,
    )

    dino_client = get_dino_client()
    faiss_client = get_faiss_client()

    try:
        dino_response = dino_client.post_json(
            "/dino/extract-features",
            DinoExtractRequest(
                image_path=image_path,
                boxes=request_body.boxes,
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("调用 DINO 服务失败")
        raise HTTPException(status_code=500, detail=f"调用 DINO 服务失败: {exc}") from exc

    image_size = dino_response.get("image_size")
    dino_results = dino_response.get("results", [])

    results: list[dict[str, Any]] = []
    best_box: dict[str, Any] | None = None

    for dino_item in dino_results:
        box_index = dino_item.get("box_index")
        box_data = dino_item.get("box")
        feature = dino_item.get("feature")

        if feature is None:
            results.append({
                "box_index": box_index,
                "box": box_data,
                "score": None,
                "is_correct": False,
                "status": "error",
                "top_match": None,
                "matches": [],
                "error_message": dino_item.get("error_message", "DINO 特征提取失败"),
            })
            continue

        try:
            search_response = faiss_client.post_json(
                "/faiss/search",
                FaissSearchRequest(
                    label_name=request_body.label_name,
                    feature=feature,
                    topk=request_body.topk,
                ).model_dump(),
            )

            matches = search_response.get("matches", [])
            top_match = matches[0] if matches else None
            score = float(top_match["score"]) if top_match else -1.0
            is_correct = score >= request_body.threshold

            item = {
                "box_index": box_index,
                "box": box_data,
                "score": round(score, 6),
                "is_correct": is_correct,
                "status": "correct" if is_correct else "wrong",
                "top_match": top_match,
                "matches": matches,
                "feature_dim": dino_item.get("feature_dim"),
                "crop_size": dino_item.get("crop_size"),
            }
            results.append(item)

            if best_box is None or item["score"] > best_box["score"]:
                best_box = item
        except Exception as exc:
            logger.exception("调用 FAISS 服务失败: box_index=%s", box_index)
            results.append({
                "box_index": box_index,
                "box": box_data,
                "score": None,
                "is_correct": False,
                "status": "error",
                "top_match": None,
                "matches": [],
                "feature_dim": dino_item.get("feature_dim"),
                "crop_size": dino_item.get("crop_size"),
                "error_message": str(exc),
            })

    correct_boxes = [item for item in results if item["status"] == "correct"]
    wrong_boxes = [item for item in results if item["status"] == "wrong"]
    error_boxes = [item for item in results if item["status"] == "error"]

    logger.info(
        "框匹配完成: image_path=%s, label_name=%s, correct=%s, wrong=%s, error=%s",
        image_path,
        request_body.label_name,
        len(correct_boxes),
        len(wrong_boxes),
        len(error_boxes),
    )

    return {
        "image_path": image_path,
        "label_name": request_body.label_name,
        "image_size": image_size,
        "threshold": request_body.threshold,
        "dino_service_url": get_dino_client().base_url,
        "faiss_service_url": get_faiss_client().base_url,
        "total_boxes": len(request_body.boxes),
        "correct_count": len(correct_boxes),
        "wrong_count": len(wrong_boxes),
        "error_count": len(error_boxes),
        "best_box": best_box,
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_box_match_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
