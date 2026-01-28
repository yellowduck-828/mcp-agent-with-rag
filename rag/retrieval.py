import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .embedding import Embedder
from .index_construction import (
    DATA_DIR,
    FAISS_INDEX_PATH,
    INDEX_PATH,
    META_PATH,
    build_index,
    load_index,
)

try:  # 可选 FAISS
    import faiss
except Exception:  # pragma: no cover - faiss 非必需
    faiss = None


def _dot_similarity(a: List[float], b: List[float]) -> float:
    """向量已归一化，可直接用内积表示余弦相似度。"""
    return sum(x * y for x, y in zip(a, b))


def _load_faiss_index() -> Tuple[object, List[dict]]:
    if faiss is None:
        return None, []
    if not FAISS_INDEX_PATH.exists() or not META_PATH.exists():
        return None, []
    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return index, meta
    except Exception:
        return None, []


def search(
    query: str,
    top_k: int = 5,
    min_score: float = 0.2,
    ensure_index: bool = True,
    index_path: Path = INDEX_PATH,
    use_faiss: bool = True,
) -> List[Dict[str, object]]:
    if ensure_index and not Path(index_path).exists():
        build_index(data_dir=DATA_DIR, index_path=index_path)

    embedder = Embedder()
    query_vec = embedder.encode([query])[0]  # numpy, 已归一化

    # 优先使用 FAISS
    if use_faiss:
        faiss_index, meta = _load_faiss_index()
        if faiss_index and meta:
            D, I = faiss_index.search(np.expand_dims(query_vec, axis=0), top_k)
            results: List[Dict[str, object]] = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1:
                    continue
                if score < min_score:
                    continue
                rec = meta[idx]
                results.append(
                    {
                        "id": rec.get("id"),
                        "score": float(score),
                        "source": rec.get("source"),
                        "dish_name": rec.get("dish_name"),
                        "category": rec.get("category"),
                        "difficulty": rec.get("difficulty"),
                        "content": rec.get("content"),
                        "parent_id": rec.get("parent_id"),
                    }
                )
            return results

    # 回退纯 Python 检索
    index = load_index(index_path=index_path)
    if not index:
        raise RuntimeError("索引为空，请先构建索引。")

    results: List[Dict[str, object]] = []
    query_embedding = query_vec.tolist()
    for record in index:
        emb = record.get("embedding") or []
        if not emb:
            continue
        score = _dot_similarity(query_embedding, emb)
        if score < min_score:
            continue
        results.append(
            {
                "id": record.get("id"),
                "score": score,
                "source": record.get("source"),
                "dish_name": record.get("dish_name"),
                "category": record.get("category"),
                "difficulty": record.get("difficulty"),
                "content": record.get("content"),
                "parent_id": record.get("parent_id"),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[: max(top_k, 1)]


def format_context(results: List[Dict[str, object]]) -> str:
    """将检索结果串接为上下文字符串，供上游模型调用。"""
    lines: List[str] = []
    for item in results:
        title = item.get("dish_name") or item.get("source") or "菜谱"
        category = item.get("category") or ""
        difficulty = item.get("difficulty") or ""
        header = f"[{title}]({item.get('source', '')})  score={item.get('score'):.3f}  {category} {difficulty}".strip()
        lines.append(header)
        lines.append(item.get("content", ""))
        lines.append("")
    return "\n".join(lines).strip()


def rag_search_tool(query: str, top_k: int = 5) -> Dict[str, object]:
    results = search(query=query, top_k=top_k)
    return {
        "query": query,
        "top_k": top_k,
        "results": results,
        "context": format_context(results),
    }


def rebuild_index_tool() -> Dict[str, object]:
    return build_index(data_dir=DATA_DIR, index_path=INDEX_PATH)


if __name__ == "__main__":
    demo = rag_search_tool("怎么做鱼香肉丝？", top_k=3)
    print(json.dumps(demo, ensure_ascii=False, indent=2))

