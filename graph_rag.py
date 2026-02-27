from pathlib import Path
from io import BytesIO
import networkx as nx
import pickle
import json
import ollama
from markdown import markdown
from bs4 import BeautifulSoup
import re
import numpy as np
from typing import List, Tuple, Dict, Any
import base64
import jieba
import fitz
from vector_store import VectorStore # 导入 VectorStore 类

_sentence_transformers_import_error = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as e:
    SentenceTransformer = None
    _sentence_transformers_import_error = e


def _require_dependency(dep_obj, package_name: str, import_error: Exception):
    if dep_obj is None:
        raise ModuleNotFoundError(
            f"缺少依赖 `{package_name}`。请先执行: pip install {package_name}"
        ) from import_error


class PDFToMarkdownConverter:
    """使用多模态模型将 PDF 转换为 Markdown。"""

    def __init__(self, model: str = "qwen3-vl:8b", page_dpi: int = 220):
        self.model = model
        self.page_dpi = page_dpi

    def convert(self, pdf_path: Path, output_dir: Path) -> Path:
        """将 PDF 文件转换为 Markdown 文件。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdf_path.stem}.md"

        doc = fitz.open(pdf_path)
        page_markdowns = []
        for page_index, page in enumerate(doc, 1):
            page_image_b64 = self._render_page_to_base64(page)
            page_markdown = self._recognize_markdown(page_image_b64, page_index)
            page_markdowns.append(page_markdown)

        markdown_text = "\n\n".join(page_markdowns)
        output_path.write_text(markdown_text, encoding="utf-8")
        return output_path

    def _render_page_to_base64(self, page) -> str:
        """渲染页面为图片并编码为 base64。"""
        scale = self.page_dpi / 72
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_buffer = BytesIO(pix.tobytes("png"))
        return base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    def _recognize_markdown(self, image_b64: str, page_index: int) -> str:
        """通过多模态模型识别页面内容并输出 Markdown。"""
        prompt = (
            "你是一个 PDF 到 Markdown 的专业转换助手。"
            "请将这页 PDF 内容严格转换为 Markdown，要求：\n"
            "1) 保留标题、列表、表格和段落结构。\n"
            "2) 所有公式必须识别为 LaTeX。独立一行公式使用 $$...$$ 包裹并单独成行；"
            "行内公式使用 $...$。\n"
            "3) 如果页面有图片，请不要输出图片链接，改为在对应位置写简要中文描述，格式为："
            "[图示说明：...]。\n"
            "4) 仅输出 Markdown 内容，不要解释，不要添加额外前后缀。"
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            options={
                "temperature": 0,
            },
        )

        content = response.get("message", {}).get("content", "").strip()
        if not content:
            return f"## 第 {page_index} 页\n\n[图示说明：该页未识别到可用文本内容。]"
        return content

class DocumentProcessor:
    def __init__(self, pdf_model: str = "qwen3-vl:8b"):
        _require_dependency(SentenceTransformer, "sentence-transformers", _sentence_transformers_import_error)
        self.encoder = SentenceTransformer('shibing624/text2vec-base-chinese')
        self.pdf_converter = PDFToMarkdownConverter(model=pdf_model)

    def convert_pdfs_to_markdown(self, data_dir: Path) -> List[Path]:
        """将目录内 PDF 转换为 Markdown。"""
        converted_files = []
        pdf_files = sorted(data_dir.glob("*.pdf"))

        for pdf_file in pdf_files:
            output_md = data_dir / f"{pdf_file.stem}.md"
            if output_md.exists() and output_md.stat().st_mtime >= pdf_file.stat().st_mtime:
                print(f"跳过 PDF 转换（Markdown 已是最新）: {pdf_file.name}")
                converted_files.append(output_md)
                continue

            print(f"开始转换 PDF -> Markdown: {pdf_file.name}")
            try:
                converted_path = self.pdf_converter.convert(pdf_file, data_dir)
                converted_files.append(converted_path)
                print(f"PDF 转换完成: {converted_path.name}")
            except Exception as e:
                print(f"PDF 转换失败 {pdf_file.name}: {e}")

        return converted_files

    def process_markdown(self, file_path: str) -> list:
        """处理单个Markdown文件为文本块。"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html = markdown(f.read())

        # 提取纯文本
        text = BeautifulSoup(html, 'html.parser').get_text()
        text = re.sub(r'\s+', ' ', text)

        # 中文分块
        return self._chinese_chunk(text, file_path)

    def _chinese_chunk(self, text: str, source: str) -> list:
        """基于分词和标点符号的智能分块。"""
        chunks = []
        buffer = []
        current_len = 0

        # 使用 Jieba 进行分词
        words = jieba.cut(text)
        sentence = ''

        for word in words:
            sentence += word
            # 判断是否遇到句子结尾的标点符号来分割句子
            if re.search(r'[。！？；]', word):
                sentence = sentence.strip()
                if sentence:
                    sent_len = len(sentence)
                    if current_len + sent_len > 500 and buffer:  # 最大块长500字
                        chunks.append({
                            "content": "".join(buffer),
                            "source": source
                        })
                        buffer = []
                        current_len = 0

                    buffer.append(sentence)
                    current_len += sent_len
                    sentence = ''

        # 添加剩余的句子
        if buffer:
            chunks.append({
                "content": "".join(buffer),
                "source": source
            })

        return chunks

class GraphRAG:
    def __init__(self, data_dir="data", save_dir="model_files"):
        """
        初始化 GraphRAG。

        Args:
            data_dir (str): 知识库数据目录.
            save_dir (str): 模型文件保存目录.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"知识库文件夹 {data_dir} 不存在！")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True) # 确保模型保存目录存在

        self.processor = DocumentProcessor()
        self.graph = nx.DiGraph()  # 使用有向图
        _require_dependency(SentenceTransformer, "sentence-transformers", _sentence_transformers_import_error)
        self.embeddings_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        self.node_embeddings = {}
        self.graph_save_path = self.save_dir / "graph_data.pkl" # 图谱保存路径
        self.chunk_contents = {} # 存储原始文本块

    def extract_entities_and_relations(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """改进的实体关系抽取，增加错误处理和JSON解析的健壮性。"""
        prompt = f"""
        从以下文本中提取电力系统、高性能计算、AI、工程经验、计算机技术等技术相关的实体和关系（重点是电力系统）。
        **请严格按照如下JSON格式输出，不要包含任何额外的文字或说明，不要忘记relations的最后一个"]"符号：**
        {{
            "entities": ["实体1", "实体2", "实体3"],
            "relations": [["实体1", "关系描述", "实体2"], ["实体1", "关系描述", "实体3"],...]
        }}

        文本：{text}
        再次强调,输出必须是一个完整的,有效的JSON对象
        """

        try:
            response = ollama.generate(
                model='deepseek-r1:7b',
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 0.9
                }
            )

            # 清理和规范化 JSON 字符串
            response_text = response['response'].strip()
            response_text = re.sub(r'//.*', '', response_text)  # 删除 // 及其后的所有内容
            # 查找 JSON 对象的开始和结束位置
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误：{e}\n原始JSON字符串：{json_str}")
                    return [], []

                entities = result.get('entities', [])
                relations = result.get('relations', [])

                # 数据清理和验证
                entities = list(set([e.strip() for e in entities if isinstance(e, str) and len(e.strip()) > 1]))
                valid_relations = []
                for rel in relations:
                    if isinstance(rel, list) and len(rel) == 3:
                        subj, pred, obj = [str(x).strip() for x in rel]
                        if all(len(x) > 1 for x in [subj, pred, obj]):
                            valid_relations.append((subj, pred, obj))

                return entities, valid_relations
            else:
                print("未找到有效的JSON结构")
                return [], []

        except Exception as e:
            print(f"实体关系抽取过程出错: {str(e)}")
            return [], []

    def build_graph(self, all_chunks, force_rebuild=False):
        """构建知识图谱，增加错误处理和进度显示。"""
        if not force_rebuild and self.graph_save_path.exists():
            try:
                self.load_graph()
                print("已从本地加载知识图谱")
                return
            except Exception as e:
                print(f"加载已有图谱失败: {e}，将重新构建")

        print("开始构建知识图谱...")
        self.graph = nx.DiGraph()
        self.chunk_contents = {}
        self.node_embeddings = {}

        total_chunks = len(all_chunks)
        for idx, chunk in enumerate(all_chunks, 1):
            if idx % 10 == 0:
                print(f"正在处理第 {idx}/{total_chunks} 个文本块...")

            content = chunk["content"]
            chunk_id = f"chunk_{idx-1}" # 使用 chunk 索引作为 chunk_id
            self.chunk_contents[chunk_id] = content

            try:
                entities, relations = self.extract_entities_and_relations(content)

                # 添加实体节点
                for entity in entities:
                    if entity not in self.graph:
                        self.graph.add_node(entity)
                        self.node_embeddings[entity] = self.embeddings_model.encode(entity)
                    # 将实体与文本块关联
                    self.graph.add_edge(entity, chunk_id, type="appears_in")

                # 添加关系
                for subj, pred, obj in relations:
                    if subj in entities and obj in entities:
                        self.graph.add_edge(subj, obj, relation=pred)

            except Exception as e:
                print(f"处理文本块 {chunk_id} 时出错: {e}")
                continue

        print(f"知识图谱构建完成: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")

        try:
            self.save_graph()
            print("知识图谱已保存至本地")
        except Exception as e:
            print(f"保存图谱时出错: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """增强的语义搜索 (基于图谱)。"""
        query_embedding = self.embeddings_model.encode(query)

        # 计算与查询最相关的实体
        entity_scores = {}
        for node in self.graph.nodes():
            if node in self.node_embeddings:  # 只考虑实体节点
                similarity = self.cosine_similarity(query_embedding, self.node_embeddings[node])
                entity_scores[node] = similarity

        # 获取top_k相关实体及其邻居
        relevant_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        seen_chunks = set()

        for entity, score in relevant_entities:
            # 获取实体相关的文本块
            for _, chunk_id in self.graph.out_edges(entity):
                if chunk_id.startswith('chunk_') and chunk_id not in seen_chunks:
                    chunk_content = self.chunk_contents.get(chunk_id, '')
                    if chunk_content:
                        results.append({
                            'content': chunk_content,
                            'score': score,
                            'entity': entity
                        })
                        seen_chunks.add(chunk_id)

            # 获取实体的关系信息
            for _, neighbor in self.graph.out_edges(entity):
                if not neighbor.startswith('chunk_'):
                    edge_data = self.graph.get_edge_data(entity, neighbor)
                    relation = edge_data.get('relation', 'related_to')
                    results.append({
                        'content': f"{entity} --{relation}--> {neighbor}",
                        'score': score * 0.8,  # 降低关系信息的权重
                        'entity': entity
                    })

        # 按相关性排序并返回结果
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        return results

    def generate_answer(self, query: str, context: List[Dict[str, Any]], max_tokens: int = 800) -> str:
        """改进的答案生成。"""
        context_text = "\n".join([
            f"- {item['content']}" for item in context
        ])

        prompt = f"""
        基于以下知识库信息回答问题。要求：
        1. 只回答与电力系统和电力电子技术相关的问题
        2. 综合运用知识库信息和专业知识
        3. 保持专业性和逻辑性
        4. 分点说明，简明扼要

        知识库信息：
        {context_text}

        问题：{query}
        """

        try:
            response = ollama.generate(
                model="deepseek-r1:7b",
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2
                }
            )
            return response['response']
        except Exception as e:
            print(f"生成答案失败: {e}")
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度。"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def save_graph(self):
        """保存图谱和相关数据。"""
        data = {
            "graph": self.graph,
            "node_embeddings": self.node_embeddings,
            "chunk_contents": self.chunk_contents
        }
        with open(self.graph_save_path, "wb") as f:
            pickle.dump(data, f)

    def load_graph(self):
        """加载图谱和相关数据。"""
        load_path = Path(self.graph_save_path)
        if load_path.exists():
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self.graph = data["graph"]
                self.node_embeddings = data["node_embeddings"]
                self.chunk_contents = data.get("chunk_contents", {})
            print(f"知识图谱已加载自: {load_path}")
        else:
            print(f"知识图谱文件不存在: {load_path}, 将重新构建图谱")
            raise FileNotFoundError(f"知识图谱文件不存在: {load_path}") # 抛出异常


class ImprovedGraphRAG(GraphRAG): # 继承自 GraphRAG
    def __init__(self, data_dir="data", save_dir="model_files"):
        """
        初始化 ImprovedGraphRAG，集成 VectorStore。

        Args:
            data_dir (str): 知识库数据目录.
            save_dir (str): 模型文件保存目录，用于存放向量索引和图谱数据.
        """
        super().__init__(data_dir=data_dir, save_dir=save_dir) # 调用父类 GraphRAG 的初始化方法
        self.vector_store = VectorStore(embedding_size=768, # 假设 sentence-transformers 模型输出维度是 768
                                        index_path=self.save_dir / "vector_index.faiss") # 指定向量索引保存路径

    def process_documents(self):
        """处理文档，构建向量索引和知识图谱。"""
        all_chunks = []
        self.processor.convert_pdfs_to_markdown(self.data_dir)
        md_files = list(self.data_dir.glob("*.md"))
        total_files = len(md_files)

        for idx, md_file in enumerate(md_files, 1):
            print(f"处理文档 [{idx}/{total_files}]: {md_file.name}")
            try:
                chunks = self.processor.process_markdown(str(md_file))
                print(f"文档 {md_file.name} 处理完成，共生成 {len(chunks)} 个文本块")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"处理文档 {md_file.name} 时出错: {e}")
                continue

        # 构建向量索引
        print("开始构建向量索引...")
        all_embeddings = self.processor.encoder.encode(
            [c["content"] for c in all_chunks],
            batch_size=32,
            show_progress_bar=True
        )
        chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))] # 生成 chunk_id 列表
        self.vector_store.add_items(all_embeddings, chunk_ids) # 使用 VectorStore 添加向量和 chunk_id
        self.vector_store.save_index(self.vector_store.index_path) # 保存向量索引
        print("向量索引构建完成并保存。")

        # 构建知识图谱
        self.build_graph(all_chunks=all_chunks, force_rebuild=True) # 构建知识图谱, 传入 all_chunks


    def load(self):
        """加载模型 (包括向量索引和图谱)。"""
        try:
            self.vector_store.load_index(self.vector_store.index_path) # 加载向量索引
            self.load_graph() # 加载知识图谱
            print("模型加载完成 (包括向量索引和图谱)。")
        except FileNotFoundError as e:
            print(f"加载模型失败: {e}, 请先处理文档构建模型。")
            raise e #  向上层抛出异常，提示需要先处理文档


    def hybrid_search(self, query: str, top_k_vector: int = 3, top_k_graph: int = 2) -> List[Dict[str, Any]]:
        """结合向量检索和图谱结构的混合搜索策略。"""
        query_embedding = self.embeddings_model.encode(query)

        # 1. 向量检索
        relevant_chunk_ids, distances = self.vector_store.search(query_embedding, top_k=top_k_vector)
        vector_results = []
        for chunk_id, distance in zip(relevant_chunk_ids, distances):
            content = self.chunk_contents.get(chunk_id, '')
            if content:
                vector_results.append({
                    'content': content,
                    'score': 1 - distance / 2, # 欧氏距离转换为相似度分数
                    'source': 'vector_search',
                    'chunk_id': chunk_id
                })

        # 2. 图谱实体扩展和关系发现
        graph_results = []
        seen_entities = set()
        for result_item in vector_results: # 基于向量检索结果进行图谱扩展
            chunk_id = result_item['chunk_id']

            # 查找包含此 chunk_id 的实体
            related_entities = [node for node, neighbor in self.graph.out_edges(chunk_id) if neighbor != chunk_id]

            for entity in related_entities:
                if entity not in seen_entities:
                    seen_entities.add(entity)
                    # 获取实体及其关系
                    entity_info = self._get_entity_and_relations(entity, query_embedding, score_boost=result_item['score']) # 实体分数加入 score_boost
                    graph_results.extend(entity_info)


        # 3. 结果融合和排序
        combined_results = vector_results + graph_results
        combined_results = sorted(combined_results, key=lambda x: x['score'], reverse=True)[:(top_k_vector + top_k_graph)] #  混合结果取 top_k

        return combined_results


    def _get_entity_and_relations(self, entity, query_embedding, score_boost=1.0):
        """获取实体信息及其相关关系，并计算相关性得分，可以根据 score_boost 调整分数。"""
        entity_results = []

        # 计算实体与查询的语义相似度
        if entity in self.node_embeddings:
            entity_embedding = self.node_embeddings[entity]
            entity_score = self.cosine_similarity(query_embedding, entity_embedding) * score_boost #  加入 score_boost 调整实体初始分数
        else:
            entity_score = 0.5 * score_boost # 默认分数

        entity_results.append({
            'content': f"**实体**: {entity}",
            'score': entity_score,
            'source': 'graph_entity',
            'entity': entity
        })

        # 获取实体的关系信息
        for _, neighbor in self.graph.out_edges(entity):
            if not neighbor.startswith('chunk_'): #  排除文本块节点
                edge_data = self.graph.get_edge_data(entity, neighbor)
                relation = edge_data.get('relation', 'related_to')
                relation_content = f"**关系**: {entity} --{relation}--> {neighbor}"
                relation_score = entity_score * 0.8 # 关系的分数可以适当降低
                entity_results.append({
                    'content': relation_content,
                    'score': relation_score,
                    'source': 'graph_relation',
                    'entity': entity,
                    'relation': relation,
                    'neighbor_entity': neighbor
                })
        return entity_results


    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]: #  search 方法现在调用混合搜索
        """修改 search 方法，默认使用混合搜索策略。"""
        return self.hybrid_search(query, top_k_vector=top_k, top_k_graph=top_k // 2) #  可以调整 top_k_vector 和 top_k_graph 的比例
