import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, embedding_size, index_path=None):
        self.embedding_size = int(embedding_size)
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.id_to_chunk_id = []
        self.index_path = Path(index_path) if index_path else None

    def reset(self):
        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.id_to_chunk_id = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def add_item(self, embedding, chunk_id):
        embedding = self._normalize(embedding)
        self.index.add(embedding)
        self.id_to_chunk_id.append(chunk_id)

    def add_items(self, embeddings, chunk_ids):
        embeddings = self._normalize(embeddings)
        self.index.add(embeddings)
        self.id_to_chunk_id.extend(chunk_ids)

    def search(self, query_vector, top_k=3):
        if self.is_empty():
            return [], np.array([], dtype="float32")

        top_k = min(int(top_k), self.index.ntotal)
        query_vector = self._normalize(query_vector)
        similarities, faiss_ids = self.index.search(query_vector, top_k)

        chunk_ids = []
        valid_scores = []
        for idx, score in zip(faiss_ids[0], similarities[0]):
            if idx < 0 or idx >= len(self.id_to_chunk_id):
                continue
            chunk_ids.append(self.id_to_chunk_id[idx])
            valid_scores.append(float(score))
        return chunk_ids, np.array(valid_scores, dtype="float32")

    def save_index(self, save_path):
        save_path = Path(save_path)
        faiss.write_index(self.index, str(save_path))
        with open(str(save_path.with_suffix(".pkl")), "wb") as f:
            pickle.dump(self.id_to_chunk_id, f)
        print(f"FAISS 索引已保存到: {save_path}")

    def load_index(self, load_path):
        load_path = Path(load_path)
        if load_path.exists():
            self.index = faiss.read_index(str(load_path))
            with open(str(load_path.with_suffix(".pkl")), "rb") as f:
                self.id_to_chunk_id = pickle.load(f)
            print(f"FAISS 索引已加载自: {load_path}")
        else:
            print(f"FAISS 索引文件不存在: {load_path}, 将重新构建索引")
            raise FileNotFoundError(f"FAISS 索引文件不存在: {load_path}")

    def is_empty(self):
        return self.index.ntotal == 0
