from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:

    model_name: str = "BAAI/bge-small-zh-v1.5"

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        返回 shape = (N, D) 的 float32 向量（已归一化，可直接用内积做相似度）。
        """
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 归一化后内积等价于 cosine
        )
        return vecs.astype(np.float32)

