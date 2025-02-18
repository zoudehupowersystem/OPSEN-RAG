import faiss
import numpy as np
import pickle
from pathlib import Path

class VectorStore:
    def __init__(self, embedding_size, index_path=None):
        """
        初始化 VectorStore。

        Args:
            embedding_size (int): 向量维度。
            index_path (str, optional): FAISS 索引的保存路径. Defaults to None.
        """
        self.embedding_size = embedding_size
        self.index = faiss.IndexFlatL2(embedding_size)  # 选择 FAISS 索引类型 (L2 距离)
        self.id_to_chunk_id = [] # 存储索引 ID 到 chunk_id 的映射
        self.index_path = Path(index_path) if index_path else None

    def add_item(self, embedding, chunk_id):
        """添加单个向量到索引。"""
        embedding = embedding.astype('float32').reshape(1, -1) # 确保数据类型和形状
        self.index.add(embedding)
        self.id_to_chunk_id.append(chunk_id)


    def add_items(self, embeddings, chunk_ids):
        """批量添加向量到索引，并进行归一化。"""
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # 归一化
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        self.id_to_chunk_id.extend(chunk_ids)


    def search(self, query_vector, top_k=3):
        """使用 FAISS 索引进行相似度搜索。"""
        query_vector = query_vector / np.linalg.norm(query_vector) # 归一化查询向量
        query_vector = query_vector.astype('float32').reshape(1, -1)
        distances, faiss_ids = self.index.search(query_vector, top_k)
        chunk_ids = [self.id_to_chunk_id[idx] for idx in faiss_ids[0]] # 根据 FAISS 返回的索引 ID 找到 chunk_id
        return chunk_ids, distances[0] # 返回 chunk_id 列表和对应的距离


    def save_index(self, save_path):
        """保存 FAISS 索引到文件。"""
        faiss.write_index(self.index, str(save_path))
        with open(str(Path(save_path).with_suffix('.pkl')), 'wb') as f: # 同时保存 id_to_chunk_id 映射
            pickle.dump(self.id_to_chunk_id, f)
        print(f"FAISS 索引已保存到: {save_path}")

    def load_index(self, load_path):
        """从文件加载 FAISS 索引。"""
        load_path = Path(load_path)
        if load_path.exists():
            self.index = faiss.read_index(str(load_path))
            with open(str(load_path.with_suffix('.pkl')), 'rb') as f:
                self.id_to_chunk_id = pickle.load(f)
            print(f"FAISS 索引已加载自: {load_path}")
        else:
            print(f"FAISS 索引文件不存在: {load_path}, 将重新构建索引")

    def is_empty(self):
        """检查索引是否为空。"""
        return self.index.ntotal == 0