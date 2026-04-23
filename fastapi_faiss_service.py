import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# FAISS 服务只负责向量检索，不依赖 DINO 或图片处理逻辑
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("faiss_service")


DEFAULT_STORE_ROOT = "./faiss_label_store"


class SearchRequest(BaseModel):
    """FAISS 检索请求。"""

    label_name: str = Field(..., description="标签名称")
    feature: list[float] = Field(..., min_length=1, description="待检索向量")
    topk: int = Field(1, ge=1, le=20, description="返回前 k 个结果")

    @field_validator("label_name")
    @classmethod
    def validate_label_name(cls, value: str):
        if not value or not value.strip():
            raise ValueError("label_name 不能为空")
        cleaned = value.strip().replace("\\", "/").strip("/")
        if ".." in cleaned:
            raise ValueError("label_name 非法，不能包含 '..'")
        return cleaned


class LabelVectorStore:
    """标签向量库结构。"""

    def __init__(self, label_name: str, index: faiss.Index, id_to_path: dict[int, str], source: str):
        self.label_name = label_name
        self.index = index
        self.id_to_path = id_to_path
        self.source = source


@lru_cache(maxsize=64)
def get_label_store(label_name: str) -> LabelVectorStore:
    """
    获取指定标签的向量库。
    FAISS 服务只接受预构建索引，不负责从图片提特征建库。
    """
    store_root = Path(os.getenv("LABEL_FAISS_ROOT", DEFAULT_STORE_ROOT)).resolve()
    label_store_dir = store_root / label_name
    index_path = label_store_dir / "val.faiss"
    mapping_path = label_store_dir / "id_to_path.json"

    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            f"未找到标签 {label_name} 的预构建索引。"
            f" 期望文件: {index_path} 和 {mapping_path}"
        )

    logger.info("开始加载标签索引: label=%s, index=%s", label_name, index_path)
    index = faiss.read_index(str(index_path))

    with open(mapping_path, "r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    id_to_path = {int(k): str(v) for k, v in raw_mapping.items()}
    logger.info("标签索引加载完成: label=%s, count=%s", label_name, index.ntotal)
    return LabelVectorStore(label_name, index, id_to_path, source="prebuilt_faiss")


def do_search(label_name: str, feature: list[float], topk: int) -> dict[str, Any]:
    """
    对指定标签的向量库执行检索。
    这里仅做向量归一化和 FAISS 检索，不触碰任何 DINO 逻辑。
    """
    store = get_label_store(label_name)

    query = np.asarray(feature, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    if query.ndim != 2:
        raise ValueError(f"查询向量维度不正确: {query.shape}")

    faiss.normalize_L2(query)
    scores, ids = store.index.search(query, topk)

    matches = []
    for score, idx in zip(scores[0], ids[0]):
        idx = int(idx)
        if idx == -1:
            continue
        matches.append({
            "match_id": idx,
            "score": float(score),
            "matched_reference_path": store.id_to_path.get(idx),
        })

    return {
        "label_name": label_name,
        "topk": topk,
        "vector_store_source": store.source,
        "match_count": len(matches),
        "matches": matches,
    }


app = FastAPI(
    title="FAISS 检索服务",
    description="只负责加载预构建向量库并执行检索，不包含任何 DINOv3 特征提取逻辑。",
    version="2.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok"}


@app.post("/faiss/search")
def search_label_vectors(request_body: SearchRequest) -> dict[str, Any]:
    """
    根据标签名和特征向量执行相似度检索。
    """
    logger.info(
        "收到向量检索请求: label=%s, topk=%s, dim=%s",
        request_body.label_name,
        request_body.topk,
        len(request_body.feature),
    )

    try:
        result = do_search(
            label_name=request_body.label_name,
            feature=request_body.feature,
            topk=request_body.topk,
        )
        logger.info(
            "向量检索完成: label=%s, matches=%s, source=%s",
            request_body.label_name,
            result["match_count"],
            result["vector_store_source"],
        )
        return result
    except FileNotFoundError as exc:
        logger.exception("标签向量库不存在")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("检索参数错误")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("FAISS 检索失败")
        raise HTTPException(status_code=500, detail=f"FAISS 检索失败: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_faiss_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
