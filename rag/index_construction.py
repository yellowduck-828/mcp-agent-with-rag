import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data_preparation import DataPreparationModule
from .embedding import Embedder

try:  # 可选：FAISS 加速
    import faiss
except Exception:  # pragma: no cover - faiss 非必需
    faiss = None

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"
INDEX_PATH = INDEX_DIR / "index.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"


def _batched(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_index(
    data_dir: Path = DATA_DIR,
    index_path: Path = INDEX_PATH,
    batch_size: int = 16,
) -> Dict[str, object]:
    """加载菜谱 markdown，分块后生成向量索引文件。"""
    data_dir = Path(data_dir)
    index_path = Path(index_path)

    prep = DataPreparationModule(str(data_dir))
    prep.load_documents()
    chunks = prep.chunk_documents()

    embedder = Embedder()
    embeddings: List[List[float]] = []
    contents = [chunk.page_content for chunk in chunks]
    for batch in _batched(contents, batch_size):
        vecs = embedder.encode(batch)  # numpy array, already normalized
        embeddings.extend(vec.tolist() for vec in vecs)

    records: List[Dict[str, object]] = []
    for chunk, emb in zip(chunks, embeddings):
        meta = chunk.metadata or {}
        records.append(
            {
                "id": meta.get("chunk_id"),
                "parent_id": meta.get("parent_id"),
                "source": meta.get("source"),
                "dish_name": meta.get("dish_name"),
                "category": meta.get("category"),
                "difficulty": meta.get("difficulty"),
                "content": chunk.page_content,
                "embedding": emb,
            }
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    # 构建 FAISS 索引（如可用），加速检索
    faiss_ok = False
    if faiss is not None:
        try:
            vec_matrix = np.array(embeddings, dtype="float32")
            dim = vec_matrix.shape[1] if vec_matrix.size else 0
            index = faiss.IndexFlatIP(dim)
            if vec_matrix.size:
                index.add(vec_matrix)
            faiss.write_index(index, str(FAISS_INDEX_PATH))

            # 保存 metadata（去掉 embedding）供检索时返回
            meta_records = []
            for rec in records:
                meta_records.append({k: v for k, v in rec.items() if k != "embedding"})
            with open(META_PATH, "w", encoding="utf-8") as mf:
                json.dump(meta_records, mf, ensure_ascii=False)
            faiss_ok = True
        except Exception:
            faiss_ok = False

    stats = prep.get_statistics()
    return {
        "message": "索引已构建",
        "chunks": len(records),
        "index_path": str(index_path),
        "faiss_index": str(FAISS_INDEX_PATH) if faiss_ok else None,
        "meta_path": str(META_PATH) if faiss_ok else None,
        "stats": stats,
    }


def load_index(index_path: Path = INDEX_PATH) -> List[Dict[str, object]]:
    if not Path(index_path).exists():
        raise FileNotFoundError(f"索引文件不存在，请先执行 build_index，路径：{index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    result = build_index()
    print(json.dumps(result, ensure_ascii=False, indent=2))

