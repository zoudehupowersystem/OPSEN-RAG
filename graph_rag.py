from pathlib import Path
from io import BytesIO
import os
import base64
import copy
import json
import pickle
import re
import time
import traceback
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz
import jieba
import networkx as nx
import numpy as np
import ollama
from bs4 import BeautifulSoup
from markdown import markdown

from vector_store import VectorStore

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


_embedding_model_singleton: Dict[str, Any] = {}


DEFAULT_RUNTIME_CONFIG = {
    "models": {
        "embedding": "shibing624/text2vec-base-chinese",
        "pdf_vision": "qwen3-vl:30b",
        "entity_extraction": "qwen3-vl:30b",
        "answer_generation": "qwen3-vl:30b",
    },
    "pdf_conversion": {
        "mode": "llm",  # llm | auto | local
        "page_dpi": 220,
        "show_llm_interaction": True,
        "max_retries": 5,
        "retry_wait_s": 2.0,
        "release_vram_each_page": True,
        "num_ctx": 16384,
        "max_output_tokens": 4096,
        "export_page_json": True,
        "text_density_threshold": 80,
        "min_text_blocks_for_local": 3,
        "clean_char_ratio_threshold": 0.82,
        "tiny_text_min_pt": 5.0,
        "off_page_tolerance_pt": 6.0,
        "detect_repeated_header_footer": True,
        "header_footer_margin_ratio": 0.08,
        "xycut_min_gap_pt": 18.0,
    },
    "retrieval": {
        "top_k_vector": 8,
        "top_k_graph": 4,
        "neighbor_window": 1,
        "candidate_multiplier": 4,
        "candidate_floor": 20,
    },
    "ollama_options": {
        "entity_extraction": {"temperature": 0.0, "top_p": 0.9},
        "answer_generation": {"temperature": 0.0, "top_p": 0.9, "repeat_penalty": 1.2},
    },
    "prompts": {
        "pdf_conversion": (
            "你是一个 PDF 到 Markdown 的专业转换助手。"
            "请将这页 PDF 内容严格转换为 Markdown，要求：\n"
            "1) 保留标题、列表、表格和段落结构。\n"
            "2) 所有公式必须识别为 LaTeX。独立一行公式使用 $$...$$ 包裹并单独成行；"
            "行内公式使用 $...$。\n"
            "3) 如果页面有图片，请不要输出图片链接，改为在对应位置写简要中文描述，格式为："
            "[图示说明：...]。\n"
            "4) 仅输出 Markdown 内容，不要解释，不要添加额外前后缀。"
        ),
        "entity_extraction": (
            "从以下文本中提取电力系统、高性能计算、AI、工程经验、计算机技术等技术相关的实体和关系（重点是电力系统）。\n"
            "**请严格按照如下JSON格式输出，不要包含任何额外的文字或说明，不要忘记relations的最后一个\"]\"符号：**\n"
            "{{\n"
            "    \"entities\": [\"实体1\", \"实体2\", \"实体3\"],\n"
            "    \"relations\": [[\"实体1\", \"关系描述\", \"实体2\"], [\"实体1\", \"关系描述\", \"实体3\"],...]\n"
            "}}\n\n"
            "文本：{text}\n"
            "再次强调,输出必须是一个完整的,有效的JSON对象"
        ),
        "answer_generation": (
            "你是电力系统问答助手。请严格基于“知识库信息”回答，优先引用原文证据，减少自行发挥。\n\n"
            "回答要求：\n"
            "1. 只回答与电力系统和电力电子技术相关的问题。\n"
            "2. 先给出证据归纳，再给出结论。\n"
            "3. 每个关键结论后都要附带证据编号，如 [证据1]、[证据3]。\n"
            "4. 证据中的来源、页码和标题信息应尽可能保留。\n"
            "5. 若检索内容不足以支撑结论，明确写“根据当前检索内容无法确定”。\n"
            "6. 不要编造未在证据中出现的事实。\n\n"
            "知识库信息：\n"
            "{context_text}\n\n"
            "问题：{query}\n"
        ),
    },
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_runtime_config(config_path: str = "config/rag_config.json") -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_RUNTIME_CONFIG)
    path = Path(config_path)
    if path.exists():
        user_config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(user_config, dict):
            raise ValueError(f"配置文件必须是 JSON 对象: {config_path}")
        _deep_merge_dict(config, user_config)
    return config


def _get_hf_cache_roots() -> List[Path]:
    roots: List[Path] = []
    env_hf_home = os.environ.get("HF_HOME")
    env_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    env_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")

    if env_hub_cache:
        roots.append(Path(env_hub_cache))
    if env_transformers_cache:
        roots.append(Path(env_transformers_cache))
    if env_hf_home:
        roots.append(Path(env_hf_home) / "hub")

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")

    unique_roots: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root.expanduser())
        if key not in seen:
            seen.add(key)
            unique_roots.append(root.expanduser())
    return unique_roots


def _has_local_hf_cache(model_name: str) -> bool:
    model_key = model_name.replace("/", "--")
    hub_prefix = f"models--{model_key}"
    sentence_transformer_dirname = model_name.split("/")[-1]

    for root in _get_hf_cache_roots():
        if not root.exists():
            continue

        hub_model_dir = root / hub_prefix
        if hub_model_dir.exists():
            refs_dir = hub_model_dir / "refs"
            snapshots_dir = hub_model_dir / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                return True
            if refs_dir.exists() and any(refs_dir.iterdir()):
                return True
            if any(hub_model_dir.glob("**/config.json")):
                return True

        st_model_dir = root / sentence_transformer_dirname
        modules_json = st_model_dir / "modules.json"
        if st_model_dir.exists() and (modules_json.exists() or any(st_model_dir.glob("**/config_sentence_transformers.json"))):
            return True

    return False


def get_shared_embedding_model(model_name: str = "shibing624/text2vec-base-chinese"):
    """获取共享的文本向量模型实例，优先复用本地缓存并避免不必要的联网检查。"""
    global _embedding_model_singleton
    _require_dependency(SentenceTransformer, "sentence-transformers", _sentence_transformers_import_error)
    if model_name not in _embedding_model_singleton:
        local_cache_available = _has_local_hf_cache(model_name)
        if local_cache_available:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            print(f"[Embedding] 检测到本地 HuggingFace 缓存，使用离线模式加载: {model_name}")
        else:
            print(f"[Embedding] 未检测到本地 HuggingFace 缓存，将按默认模式加载: {model_name}")

        try:
            _embedding_model_singleton[model_name] = SentenceTransformer(
                model_name,
                local_files_only=local_cache_available,
            )
        except TypeError:
            _embedding_model_singleton[model_name] = SentenceTransformer(model_name)
    return _embedding_model_singleton[model_name]


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _markdown_to_plain_text(md_text: str) -> str:
    html = markdown(md_text or "")
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    return _normalize_text(text)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _normalize_bbox(bbox: Iterable[Any]) -> Optional[List[float]]:
    try:
        x0, y0, x1, y1 = [round(float(v), 2) for v in bbox]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _union_bbox(boxes: Iterable[Optional[Iterable[Any]]]) -> Optional[List[float]]:
    valid = [_normalize_bbox(b) for b in boxes if b is not None]
    valid = [b for b in valid if b is not None]
    if not valid:
        return None
    return [
        round(min(b[0] for b in valid), 2),
        round(min(b[1] for b in valid), 2),
        round(max(b[2] for b in valid), 2),
        round(max(b[3] for b in valid), 2),
    ]


def _canonical_header_footer_text(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[·•▪■□◆◇◦○●]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _render_prompt_template(template: str, **kwargs: Any) -> str:
    """仅替换约定占位符，避免 JSON 花括号被 str.format 误解析。"""
    rendered = template or ""
    for key, value in kwargs.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered


def _natural_sort_key(path_like: Any):
    text = str(path_like)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


_SUSPICIOUS_OCR_FRAGMENTS = [
    "犌犅",
    "狆狅狑犲狉",
    "狊狔狊狋犲犿",
    "狉狅狋狅狉",
    "犪狀犵犾犲",
    "狊狋犪犫犻犾犻狋狔",
    "狋狉犪狀狊犻犲狀狋",
    "犱狔狀犪犿犻犮",
    "狉犲狀犲狑犪犫犾犲",
]
_SUSPICIOUS_OCR_RUN_PATTERN = re.compile(r"[犌犅狆狅狑犲狉狊狔狋犿犮狌狀犾犻犱犪犵]{4,}")


def _contains_suspicious_ocr_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if any(fragment in normalized for fragment in _SUSPICIOUS_OCR_FRAGMENTS):
        return True
    return bool(_SUSPICIOUS_OCR_RUN_PATTERN.search(normalized))


class LocalStructuredPDFExtractor:
    """
    借鉴 OpenDataLoader 的“结构化输出 + 阅读顺序 + local-first”思路，
    使用 PyMuPDF 先做确定性抽取，并为每页输出可回溯的结构化 JSON。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.text_density_threshold = int(self.config.get("text_density_threshold", 80))
        self.min_text_blocks_for_local = int(self.config.get("min_text_blocks_for_local", 3))
        self.clean_char_ratio_threshold = float(self.config.get("clean_char_ratio_threshold", 0.82))
        self.tiny_text_min_pt = float(self.config.get("tiny_text_min_pt", 5.0))
        self.off_page_tolerance_pt = float(self.config.get("off_page_tolerance_pt", 6.0))
        self.header_footer_margin_ratio = float(self.config.get("header_footer_margin_ratio", 0.08))
        self.detect_repeated_header_footer = bool(self.config.get("detect_repeated_header_footer", True))
        self.xycut_min_gap_pt = float(self.config.get("xycut_min_gap_pt", 18.0))

    def extract_document(self, pdf_path: Path) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        try:
            pages: List[Dict[str, Any]] = []
            for page_index, page in enumerate(doc, 1):
                page_record = self._extract_page(page, page_index, pdf_path.name)
                pages.append(page_record)

            if self.detect_repeated_header_footer and len(pages) > 1:
                repeated = self._detect_repeated_header_footer(pages)
                if repeated:
                    for page_record in pages:
                        filtered = []
                        for element in page_record.get("elements", []):
                            region = element.get("page_region")
                            key = element.get("canonical_text")
                            if region in {"header", "footer"} and key in repeated:
                                continue
                            filtered.append(element)
                        page_record["elements"] = filtered
                        page_record["markdown"] = self._elements_to_markdown(filtered)

            return pages
        finally:
            doc.close()

    def _extract_page(self, page, page_index: int, pdf_name: str) -> Dict[str, Any]:
        page_rect = page.rect
        page_width = float(page_rect.width)
        page_height = float(page_rect.height)
        raw_text = _normalize_text(page.get_text("text"))
        clean_ratio = self._clean_char_ratio(raw_text)

        tables = self._extract_tables(page, page_index)
        table_boxes = [table["bbox"] for table in tables]
        text_blocks = self._extract_text_blocks(page, page_index, page_width, page_height, table_boxes)
        elements = tables + text_blocks
        elements = self._sort_elements_xycut_inspired(elements)
        elements = self._annotate_page_regions(elements, page_height)
        markdown_text = self._elements_to_markdown(elements)

        suspicious_ocr = _contains_suspicious_ocr_text(raw_text) or _contains_suspicious_ocr_text(markdown_text)
        local_candidate = (
            len(raw_text) >= self.text_density_threshold
            and sum(1 for item in text_blocks if item.get("text")) >= self.min_text_blocks_for_local
            and clean_ratio >= self.clean_char_ratio_threshold
            and not suspicious_ocr
        )

        return {
            "source_pdf": pdf_name,
            "page_number": page_index,
            "page_size": [round(page_width, 2), round(page_height, 2)],
            "raw_text": raw_text,
            "clean_char_ratio": round(clean_ratio, 4),
            "suspicious_ocr": suspicious_ocr,
            "recommended_mode": "local" if local_candidate else "llm",
            "elements": elements,
            "markdown": markdown_text,
        }

    def _extract_tables(self, page, page_index: int) -> List[Dict[str, Any]]:
        elements: List[Dict[str, Any]] = []
        try:
            table_finder = page.find_tables()
            tables = getattr(table_finder, "tables", []) or []
        except Exception:
            tables = []

        for idx, table in enumerate(tables, 1):
            try:
                raw_rows = table.extract() or []
            except Exception:
                continue
            normalized_rows = []
            for row in raw_rows:
                if row is None:
                    continue
                normalized_rows.append([_normalize_text(cell or "") for cell in row])
            normalized_rows = [row for row in normalized_rows if any(cell for cell in row)]
            if not normalized_rows:
                continue
            markdown_text = self._table_rows_to_markdown(normalized_rows)
            bbox = _normalize_bbox(getattr(table, "bbox", None))
            if not bbox:
                continue
            elements.append(
                {
                    "id": f"p{page_index}_table_{idx}",
                    "type": "table",
                    "level": None,
                    "text": markdown_text,
                    "bbox": bbox,
                    "font_size": None,
                    "canonical_text": _canonical_header_footer_text(markdown_text[:120]),
                }
            )
        return elements

    def _extract_text_blocks(
        self,
        page,
        page_index: int,
        page_width: float,
        page_height: float,
        table_boxes: List[List[float]],
    ) -> List[Dict[str, Any]]:
        try:
            text_dict = page.get_text("dict")
        except Exception:
            text_dict = {"blocks": []}

        blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
        elements: List[Dict[str, Any]] = []
        text_like_blocks: List[Tuple[Dict[str, Any], float]] = []

        for block in blocks:
            if block.get("type") != 0:
                continue
            bbox = _normalize_bbox(block.get("bbox"))
            if not bbox or not self._bbox_is_valid(bbox, page_width, page_height):
                continue
            if self._overlaps_any_table(bbox, table_boxes):
                continue

            lines = block.get("lines", []) or []
            line_texts = []
            span_sizes = []
            for line in lines:
                spans = line.get("spans", []) or []
                span_text = "".join(span.get("text", "") for span in spans)
                span_text = _normalize_text(span_text)
                if span_text:
                    line_texts.append(span_text)
                for span in spans:
                    size = _safe_float(span.get("size"), 0.0)
                    if size > 0:
                        span_sizes.append(size)

            text = _normalize_text("\n".join(line_texts))
            if not text:
                continue

            max_font_size = max(span_sizes) if span_sizes else 0.0
            if max_font_size and max_font_size < self.tiny_text_min_pt:
                continue

            text_like_blocks.append(({"text": text, "bbox": bbox, "font_size": max_font_size}, max_font_size))

        font_sizes = [size for _, size in text_like_blocks if size > 0]
        base_font_size = float(np.median(font_sizes)) if font_sizes else 12.0

        for idx, (record, max_font_size) in enumerate(text_like_blocks, 1):
            text = record["text"]
            bbox = record["bbox"]
            block_type, level = self._infer_block_type(text, bbox, max_font_size or base_font_size, base_font_size, page_width)
            elements.append(
                {
                    "id": f"p{page_index}_text_{idx}",
                    "type": block_type,
                    "level": level,
                    "text": text,
                    "bbox": bbox,
                    "font_size": round(float(max_font_size), 2) if max_font_size else None,
                    "canonical_text": _canonical_header_footer_text(text[:120]),
                }
            )

        return elements

    def _detect_repeated_header_footer(self, pages: List[Dict[str, Any]]) -> set:
        candidates: Dict[str, int] = {}
        for page_record in pages:
            seen_on_page = set()
            for element in page_record.get("elements", []):
                key = element.get("canonical_text")
                region = element.get("page_region")
                text = _normalize_text(element.get("text", ""))
                if region not in {"header", "footer"} or not key or len(text) < 3:
                    continue
                if key in seen_on_page:
                    continue
                seen_on_page.add(key)
                candidates[key] = candidates.get(key, 0) + 1

        min_repeat = max(2, int(len(pages) * 0.5))
        return {key for key, count in candidates.items() if count >= min_repeat}

    def _annotate_page_regions(self, elements: List[Dict[str, Any]], page_height: float) -> List[Dict[str, Any]]:
        top_margin = page_height * self.header_footer_margin_ratio
        bottom_margin = page_height * (1 - self.header_footer_margin_ratio)
        for element in elements:
            bbox = element.get("bbox") or [0, 0, 0, 0]
            y0, y1 = bbox[1], bbox[3]
            region = "body"
            if y1 <= top_margin:
                region = "header"
            elif y0 >= bottom_margin:
                region = "footer"
            element["page_region"] = region
        return elements

    def _sort_elements_xycut_inspired(self, elements: List[Dict[str, Any]], depth: int = 0) -> List[Dict[str, Any]]:
        if len(elements) <= 1 or depth >= 8:
            return sorted(elements, key=lambda item: (item["bbox"][1], item["bbox"][0]))

        y_groups, y_gap = self._group_by_axis(elements, axis="y")
        x_groups, x_gap = self._group_by_axis(elements, axis="x")

        if len(y_groups) > 1 and (len(x_groups) == 1 or y_gap >= x_gap * 0.9):
            ordered: List[Dict[str, Any]] = []
            for group in sorted(y_groups, key=lambda group: min(item["bbox"][1] for item in group)):
                ordered.extend(self._sort_elements_xycut_inspired(group, depth + 1))
            return ordered

        if len(x_groups) > 1:
            ordered = []
            for group in sorted(x_groups, key=lambda group: min(item["bbox"][0] for item in group)):
                ordered.extend(self._sort_elements_xycut_inspired(group, depth + 1))
            return ordered

        return sorted(elements, key=lambda item: (item["bbox"][1], item["bbox"][0]))

    def _group_by_axis(self, elements: List[Dict[str, Any]], axis: str) -> Tuple[List[List[Dict[str, Any]]], float]:
        start_idx = 0 if axis == "x" else 1
        end_idx = 2 if axis == "x" else 3
        ordered = sorted(elements, key=lambda item: (item["bbox"][start_idx], item["bbox"][end_idx]))
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_end = None
        largest_gap = 0.0

        for element in ordered:
            start = element["bbox"][start_idx]
            end = element["bbox"][end_idx]
            if current_end is None:
                current = [element]
                current_end = end
                continue
            gap = start - current_end
            if gap > self.xycut_min_gap_pt:
                groups.append(current)
                current = [element]
            else:
                current.append(element)
            current_end = max(current_end, end)
            largest_gap = max(largest_gap, gap)

        if current:
            groups.append(current)

        return groups, largest_gap

    def _infer_block_type(
        self,
        text: str,
        bbox: List[float],
        font_size: float,
        base_font_size: float,
        page_width: float,
    ) -> Tuple[str, Optional[int]]:
        text = _normalize_text(text)
        width = bbox[2] - bbox[0]
        short_line = len(text) <= 60
        font_ratio = (font_size / base_font_size) if base_font_size > 0 else 1.0
        numbered_heading = bool(re.match(r"^(第[一二三四五六七八九十百千万0-9]+[章节条]|[0-9]+(\.[0-9]+){0,3}|[（(][0-9一二三四五六七八九十]+[)）])", text))
        list_item = bool(re.match(r"^([•·-]|[a-zA-Z]\)|[0-9]+[)）.]|[一二三四五六七八九十]+、)", text))

        if short_line and (font_ratio >= 1.25 or numbered_heading) and width <= page_width * 0.95:
            if font_ratio >= 1.8:
                return "heading", 1
            if font_ratio >= 1.45:
                return "heading", 2
            return "heading", 3

        if list_item and len(text) <= 200:
            return "list_item", None

        return "paragraph", None

    def _elements_to_markdown(self, elements: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for element in elements:
            text = _normalize_text(element.get("text", ""))
            if not text:
                continue
            if element.get("type") == "heading":
                level = int(element.get("level") or 2)
                level = max(1, min(level, 6))
                parts.append(f"{'#' * level} {text}")
            elif element.get("type") == "list_item":
                parts.append(f"- {text}")
            else:
                parts.append(text)
        return "\n\n".join(parts).strip()

    def _table_rows_to_markdown(self, rows: List[List[str]]) -> str:
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        header = normalized[0]
        body = normalized[1:] if len(normalized) > 1 else []
        divider = ["---"] * width
        lines = [
            "| " + " | ".join(cell or " " for cell in header) + " |",
            "| " + " | ".join(divider) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(cell or " " for cell in row) + " |")
        return "\n".join(lines)

    def _bbox_is_valid(self, bbox: List[float], page_width: float, page_height: float) -> bool:
        tol = self.off_page_tolerance_pt
        x0, y0, x1, y1 = bbox
        return not (x1 < -tol or y1 < -tol or x0 > page_width + tol or y0 > page_height + tol)

    def _overlaps_any_table(self, bbox: List[float], table_boxes: List[List[float]]) -> bool:
        for table_box in table_boxes:
            if self._intersection_ratio(bbox, table_box) >= 0.3:
                return True
        return False

    def _intersection_ratio(self, a: List[float], b: List[float]) -> float:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
        inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        a_area = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
        return inter_area / a_area

    def _clean_char_ratio(self, text: str) -> float:
        chars = [ch for ch in (text or "") if not ch.isspace()]
        if not chars:
            return 0.0

        allowed_pattern = re.compile(r"[A-Za-z0-9\u4e00-\u9fff\u3400-\u4dbf，。！？；：,.:%％、（）()\[\]{}《》<>“”‘’'\"/\\+\-=*_#&$@~—–·]", re.UNICODE)
        good = 0
        for ch in chars:
            category = unicodedata.category(ch)
            if allowed_pattern.match(ch) or category.startswith("P"):
                good += 1
        return good / len(chars)


class PDFToMarkdownConverter:
    """支持 local-first / hybrid / llm 三种模式的 PDF 分页转换器。"""

    def __init__(
        self,
        model: str = "qwen3-vl:30b",
        page_dpi: int = 220,
        show_llm_interaction: bool = True,
        max_retries: int = 5,
        retry_wait_s: float = 2.0,
        release_vram_each_page: bool = True,
        num_ctx: int = 16384,
        max_output_tokens: int = 4096,
        prompt_template: str = "",
        conversion_mode: str = "auto",
        export_page_json: bool = True,
        local_extractor: Optional[LocalStructuredPDFExtractor] = None,
    ):
        self.model = model
        self.page_dpi = page_dpi
        self.show_llm_interaction = show_llm_interaction
        self.max_retries = max_retries
        self.retry_wait_s = retry_wait_s
        self.release_vram_each_page = release_vram_each_page
        self.num_ctx = num_ctx
        self.max_output_tokens = max_output_tokens
        self.prompt_template = prompt_template
        self.conversion_mode = (conversion_mode or "auto").lower()
        self.export_page_json = export_page_json
        self.local_extractor = local_extractor or LocalStructuredPDFExtractor({})

    def get_page_count(self, pdf_path: Path) -> int:
        doc = fitz.open(pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    def get_expected_page_files(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        page_count = self.get_page_count(pdf_path)
        return [output_dir / f"{pdf_path.stem}_{i}.md" for i in range(1, page_count + 1)]

    def get_expected_page_json_files(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        page_count = self.get_page_count(pdf_path)
        return [output_dir / f"{pdf_path.stem}_{i}.json" for i in range(1, page_count + 1)]

    def convert(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = output_dir / "figs"
        fig_dir.mkdir(parents=True, exist_ok=True)

        structured_pages = self.local_extractor.extract_document(pdf_path)
        doc = fitz.open(pdf_path)
        output_paths: List[Path] = []
        json_paths: List[Path] = []
        figure_paths: List[Path] = []

        try:
            total_pages = doc.page_count
            print(f"开始逐页处理 {pdf_path.name}，共 {total_pages} 页，模式={self.conversion_mode}")
            for page_index, page in enumerate(doc, 1):
                page_struct = structured_pages[page_index - 1] if page_index - 1 < len(structured_pages) else None
                output_path = output_dir / f"{pdf_path.stem}_{page_index}.md"
                json_path = output_dir / f"{pdf_path.stem}_{page_index}.json"
                figure_path = fig_dir / f"{pdf_path.stem}_{page_index}.png"
                try:
                    print(f"  -> 处理第 {page_index}/{total_pages} 页")
                    page_markdown, page_json = self._convert_page(
                        page=page,
                        page_index=page_index,
                        pdf_name=pdf_path.name,
                        page_struct=page_struct,
                        figure_path=figure_path,
                    )
                    output_path.write_text(page_markdown, encoding="utf-8")
                    output_paths.append(output_path)
                    figure_paths.append(figure_path)
                    if self.export_page_json:
                        json_path.write_text(json.dumps(page_json, ensure_ascii=False, indent=2), encoding="utf-8")
                        json_paths.append(json_path)
                    print(
                        f"  -> 第 {page_index} 页完成，输出: {output_path.name}"
                        + (f"，结构化元数据: {json_path.name}" if self.export_page_json else "")
                        + f"，中间图像: figs/{figure_path.name}"
                    )
                except Exception as e:
                    print(f"  -> 第 {page_index} 页识别失败，将写入失败占位内容: {e}")
                    failure_md = f"## 第 {page_index} 页\n\n[图示说明：该页识别失败。错误：{e}]"
                    fallback_json = {
                        "source_pdf": pdf_path.name,
                        "page_number": page_index,
                        "page_size": [round(float(page.rect.width), 2), round(float(page.rect.height), 2)],
                        "mode": "failed",
                        "elements": [
                            {
                                "id": f"p{page_index}_error_1",
                                "type": "paragraph",
                                "level": None,
                                "text": failure_md,
                                "bbox": [0.0, 0.0, round(float(page.rect.width), 2), round(float(page.rect.height), 2)],
                                "font_size": None,
                            }
                        ],
                    }
                    output_path.write_text(failure_md, encoding="utf-8")
                    output_paths.append(output_path)
                    if self.export_page_json:
                        json_path.write_text(json.dumps(fallback_json, ensure_ascii=False, indent=2), encoding="utf-8")
                        json_paths.append(json_path)
        finally:
            doc.close()

        self._cleanup_stale_outputs(pdf_path, output_dir, output_paths, json_paths, figure_paths)
        print(f"完成 {pdf_path.name} 分页转换，产出 {len(output_paths)} 个 Markdown 文件")
        return output_paths

    def _convert_page(
        self,
        page,
        page_index: int,
        pdf_name: str,
        page_struct: Optional[Dict[str, Any]],
        figure_path: Path,
    ) -> Tuple[str, Dict[str, Any]]:
        page_image_b64 = self._render_page_to_base64(page, figure_path)
        page_mode = self._resolve_page_mode(page_struct)

        if page_mode == "local" and page_struct:
            page_json = {
                "source_pdf": pdf_name,
                "page_number": page_index,
                "page_size": page_struct.get("page_size"),
                "mode": "local",
                "clean_char_ratio": page_struct.get("clean_char_ratio"),
                "elements": page_struct.get("elements", []),
            }
            return page_struct.get("markdown", ""), page_json

        page_markdown = self._recognize_markdown(page_image_b64, page_index, pdf_name)
        plain_text = _markdown_to_plain_text(page_markdown)
        page_json = {
            "source_pdf": pdf_name,
            "page_number": page_index,
            "page_size": [round(float(page.rect.width), 2), round(float(page.rect.height), 2)],
            "mode": "llm",
            "elements": [
                {
                    "id": f"p{page_index}_llm_1",
                    "type": "paragraph",
                    "level": None,
                    "text": plain_text or page_markdown,
                    "markdown": page_markdown,
                    "bbox": [0.0, 0.0, round(float(page.rect.width), 2), round(float(page.rect.height), 2)],
                    "font_size": None,
                }
            ],
        }
        return page_markdown, page_json

    def _resolve_page_mode(self, page_struct: Optional[Dict[str, Any]]) -> str:
        if self.conversion_mode == "local":
            return "local"
        if self.conversion_mode == "llm":
            return "llm"
        if page_struct:
            if page_struct.get("suspicious_ocr"):
                return "llm"
            if _contains_suspicious_ocr_text(page_struct.get("raw_text", "")):
                return "llm"
            if page_struct.get("recommended_mode") == "local":
                return "local"
        return "llm"

    def _cleanup_stale_outputs(
        self,
        pdf_path: Path,
        output_dir: Path,
        output_paths: List[Path],
        json_paths: List[Path],
        figure_paths: List[Path],
    ):
        legacy_path = output_dir / f"{pdf_path.stem}.md"
        if legacy_path.exists():
            legacy_path.unlink()

        valid_md = {p.name for p in output_paths}
        valid_json = {p.name for p in json_paths}
        valid_fig = {p.name for p in figure_paths}

        for stale_file in sorted(output_dir.glob(f"{pdf_path.stem}_*.md")):
            if stale_file.name not in valid_md:
                stale_file.unlink()
        for stale_file in sorted(output_dir.glob(f"{pdf_path.stem}_*.json")):
            if stale_file.name not in valid_json:
                stale_file.unlink()
        for stale_file in sorted((output_dir / "figs").glob(f"{pdf_path.stem}_*.png")):
            if stale_file.name not in valid_fig:
                stale_file.unlink()

    def _render_page_to_base64(self, page, figure_path: Path) -> str:
        scale = self.page_dpi / 72
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pix.tobytes("png")
        figure_path.write_bytes(image_bytes)
        image_buffer = BytesIO(image_bytes)
        return base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    def _is_retryable_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return (
            "status code: 503" in msg
            or "timeout" in msg
            or "temporarily" in msg
            or "返回空内容" in msg
        )

    def _recognize_markdown(self, image_b64: str, page_index: int, pdf_name: str) -> str:
        prompt = self.prompt_template.strip()
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.show_llm_interaction:
                    print(
                        f"    [LLM请求] 文件={pdf_name}, 页={page_index}, 尝试={attempt}/{self.max_retries}, "
                        f"模型={self.model}, keep_alive={'0s' if self.release_vram_each_page else '5m'}, "
                        f"num_ctx={self.num_ctx}, num_predict={self.max_output_tokens}, prompt长度={len(prompt)}"
                    )

                keep_alive = "0s" if self.release_vram_each_page else "5m"
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
                        "num_ctx": self.num_ctx,
                        "num_predict": self.max_output_tokens,
                    },
                    keep_alive=keep_alive,
                )

                content = response.get("message", {}).get("content", "").strip()
                if self.show_llm_interaction:
                    preview = content[:200].replace("\n", " ")
                    print(
                        f"    [LLM响应] 文件={pdf_name}, 页={page_index}, 尝试={attempt}/{self.max_retries}, "
                        f"返回长度={len(content)}, 预览={preview}"
                    )
                if not content:
                    raise RuntimeError("LLM返回空内容")
                return content
            except Exception as e:
                last_error = e
                should_retry = self._is_retryable_error(e) and attempt < self.max_retries
                print(f"    [LLM异常] 文件={pdf_name}, 页={page_index}, 尝试={attempt}/{self.max_retries}, 错误={e}")
                if should_retry:
                    wait_s = self.retry_wait_s * attempt
                    print(f"    [LLM重试] {wait_s:.1f}s 后重试第 {page_index} 页...")
                    time.sleep(wait_s)
                    continue
                break

        raise RuntimeError(f"LLM识别失败（重试{self.max_retries}次后仍失败，或持续返回空内容）: {last_error}")


class DocumentProcessor:
    def __init__(
        self,
        pdf_model: str = "qwen3-vl:30b",
        encoder=None,
        release_vram_each_page: bool = True,
        pdf_conversion_config: Dict[str, Any] = None,
        pdf_prompt_template: str = "",
    ):
        self.encoder = encoder or get_shared_embedding_model()
        self.pdf_config = pdf_conversion_config or {}
        self.local_extractor = LocalStructuredPDFExtractor(self.pdf_config)
        self.pdf_converter = PDFToMarkdownConverter(
            model=pdf_model,
            release_vram_each_page=release_vram_each_page,
            page_dpi=self.pdf_config.get("page_dpi", 220),
            show_llm_interaction=self.pdf_config.get("show_llm_interaction", True),
            max_retries=self.pdf_config.get("max_retries", 5),
            retry_wait_s=self.pdf_config.get("retry_wait_s", 2.0),
            num_ctx=self.pdf_config.get("num_ctx", 16384),
            max_output_tokens=self.pdf_config.get("max_output_tokens", 4096),
            prompt_template=pdf_prompt_template,
            conversion_mode=self.pdf_config.get("mode", "auto"),
            export_page_json=self.pdf_config.get("export_page_json", True),
            local_extractor=self.local_extractor,
        )

    def convert_pdfs_to_markdown(self, data_dir: Path) -> List[Path]:
        converted_files = []
        pdf_files = sorted(data_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            expected_page_files = self.pdf_converter.get_expected_page_files(pdf_file, data_dir)
            expected_json_files = self.pdf_converter.get_expected_page_json_files(pdf_file, data_dir)
            if expected_page_files and all(
                file.exists() and file.stat().st_mtime >= pdf_file.stat().st_mtime
                for file in expected_page_files
            ) and all(
                file.exists() and file.stat().st_mtime >= pdf_file.stat().st_mtime
                for file in expected_json_files
            ):
                print(f"跳过 PDF 转换（分页 Markdown / JSON 已是最新）: {pdf_file.name}")
                converted_files.extend(expected_page_files)
                continue

            print(f"开始转换 PDF -> Markdown(按页): {pdf_file.name}")
            try:
                converted_paths = self.pdf_converter.convert(pdf_file, data_dir)
                converted_files.extend(converted_paths)
                print(f"PDF 分页转换完成: {pdf_file.name}, 共 {len(converted_paths)} 页")
            except Exception as e:
                print(f"PDF 转换失败 {pdf_file.name}: {e}")
                print(f"错误类型: {type(e).__name__}")
                print("详细堆栈信息：")
                print(traceback.format_exc())
                print(
                    "排查建议：1) 若使用 llm/auto 模式，确认 Ollama 服务已启动；"
                    "2) 确认多模态模型已拉取；3) 若只是普通可选文本 PDF，可将 pdf_conversion.mode 设为 local。"
                )
        return converted_files

    def process_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        json_path = path.with_suffix(".json")
        if json_path.exists():
            try:
                page_json = json.loads(json_path.read_text(encoding="utf-8"))
                chunks = self._chunk_from_structured_page(page_json, source=str(path))
                if chunks:
                    return chunks
            except Exception as e:
                print(f"读取结构化 JSON 失败 {json_path.name}: {e}，回退到 Markdown 纯文本分块")

        md_text = path.read_text(encoding="utf-8")
        text = _markdown_to_plain_text(md_text)
        inferred_page = self._infer_page_number(path.stem)
        return self._chinese_chunk(
            text,
            source=str(path),
            metadata={
                "page": inferred_page,
                "bbox": None,
                "heading": None,
                "pdf_source": path.name,
                "extraction_mode": "markdown_fallback",
            },
        )

    def _chunk_from_structured_page(self, page_json: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        page_number = page_json.get("page_number")
        page_size = page_json.get("page_size") or [0.0, 0.0]
        elements = page_json.get("elements", []) or []
        extraction_mode = page_json.get("mode", "unknown")
        pdf_source = page_json.get("source_pdf") or Path(source).name

        chunks: List[Dict[str, Any]] = []
        buffer_texts: List[str] = []
        buffer_boxes: List[List[float]] = []
        current_heading: Optional[str] = None
        max_chars = 520

        def flush():
            if not buffer_texts:
                return
            content = _normalize_text("\n".join(buffer_texts))
            if not content:
                buffer_texts.clear()
                buffer_boxes.clear()
                return
            bbox = _union_bbox(buffer_boxes) or [0.0, 0.0, float(page_size[0]), float(page_size[1])]
            chunks.append(
                {
                    "content": content,
                    "source": source,
                    "page": page_number,
                    "bbox": bbox,
                    "heading": current_heading,
                    "pdf_source": pdf_source,
                    "extraction_mode": extraction_mode,
                }
            )
            buffer_texts.clear()
            buffer_boxes.clear()

        for element in elements:
            element_type = element.get("type")
            raw_text = element.get("markdown") if extraction_mode == "llm" and element.get("markdown") else element.get("text", "")
            if extraction_mode == "llm" or element_type == "table":
                text = _markdown_to_plain_text(raw_text)
            else:
                text = _normalize_text(raw_text)
            if not text:
                continue
            bbox = element.get("bbox")

            if element_type == "heading":
                flush()
                current_heading = text
                buffer_texts.append(text)
                if bbox:
                    buffer_boxes.append(bbox)
                continue

            projected_len = sum(len(item) for item in buffer_texts) + len(text)
            if projected_len > max_chars and buffer_texts:
                flush()
            buffer_texts.append(text)
            if bbox:
                buffer_boxes.append(bbox)

        flush()
        return chunks

    def _chinese_chunk(self, text: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        metadata = metadata or {}
        buffer: List[str] = []
        current_len = 0
        words = jieba.cut(text)
        sentence = ""

        for word in words:
            sentence += word
            if re.search(r"[。！？；]", word):
                sentence = sentence.strip()
                if sentence:
                    sent_len = len(sentence)
                    if current_len + sent_len > 500 and buffer:
                        chunks.append({"content": "".join(buffer), "source": source, **metadata})
                        buffer = []
                        current_len = 0
                    buffer.append(sentence)
                    current_len += sent_len
                    sentence = ""

        if sentence.strip():
            if current_len + len(sentence) > 500 and buffer:
                chunks.append({"content": "".join(buffer), "source": source, **metadata})
                buffer = []
            buffer.append(sentence.strip())

        if buffer:
            chunks.append({"content": "".join(buffer), "source": source, **metadata})

        return chunks

    def _infer_page_number(self, stem: str) -> Optional[int]:
        match = re.search(r"_(\d+)$", stem)
        return int(match.group(1)) if match else None


class GraphRAG:
    def __init__(self, data_dir="data", save_dir="model_files", config_path: str = "config/rag_config.json"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"知识库文件夹 {data_dir} 不存在！")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_config = load_runtime_config(config_path)

        embedding_model_name = self.runtime_config["models"]["embedding"]
        self.embeddings_model = get_shared_embedding_model(embedding_model_name)
        pdf_config = self.runtime_config.get("pdf_conversion", {})
        self.processor = DocumentProcessor(
            pdf_model=self.runtime_config["models"]["pdf_vision"],
            encoder=self.embeddings_model,
            release_vram_each_page=pdf_config.get("release_vram_each_page", True),
            pdf_conversion_config=pdf_config,
            pdf_prompt_template=self.runtime_config["prompts"]["pdf_conversion"],
        )
        self.graph = nx.DiGraph()
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.graph_save_path = self.save_dir / "graph_data.pkl"
        self.chunk_contents: Dict[str, str] = {}
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}

    def extract_entities_and_relations(self, text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        prompt = _render_prompt_template(self.runtime_config["prompts"]["entity_extraction"], text=text)
        try:
            response = ollama.generate(
                model=self.runtime_config["models"]["entity_extraction"],
                prompt=prompt,
                options=self.runtime_config["ollama_options"]["entity_extraction"],
            )
            response_text = response["response"].strip()
            response_text = re.sub(r"//.*", "", response_text)
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误：{e}\n原始JSON字符串：{json_str}")
                    return [], []

                entities = result.get("entities", [])
                relations = result.get("relations", [])
                entities = list({e.strip() for e in entities if isinstance(e, str) and len(e.strip()) > 1})
                valid_relations = []
                for rel in relations:
                    if isinstance(rel, list) and len(rel) == 3:
                        subj, pred, obj = [str(x).strip() for x in rel]
                        if all(len(x) > 1 for x in [subj, pred, obj]):
                            valid_relations.append((subj, pred, obj))
                return entities, valid_relations
            print("未找到有效的JSON结构")
            return [], []
        except Exception as e:
            print(f"实体关系抽取过程出错: {str(e)}")
            return [], []

    def build_graph(self, all_chunks: List[Dict[str, Any]], force_rebuild: bool = False, show_entity_relations: bool = False):
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
        self.chunk_metadata = {}
        self.node_embeddings = {}

        total_chunks = len(all_chunks)
        for idx, chunk in enumerate(all_chunks, 1):
            if idx % 10 == 0:
                print(f"正在处理第 {idx}/{total_chunks} 个文本块...")

            content = chunk["content"]
            chunk_id = f"chunk_{idx - 1}"
            self.chunk_contents[chunk_id] = content
            self.chunk_metadata[chunk_id] = {
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "bbox": chunk.get("bbox"),
                "heading": chunk.get("heading"),
                "pdf_source": chunk.get("pdf_source"),
                "extraction_mode": chunk.get("extraction_mode"),
            }
            self.graph.add_node(chunk_id, node_type="chunk")

            try:
                entities, relations = self.extract_entities_and_relations(content)

                if show_entity_relations:
                    if entities:
                        print(f"[实体抽取] {chunk_id}: {', '.join(entities)}")
                    if relations:
                        rel_text = "；".join([f"({subj} -{pred}-> {obj})" for subj, pred, obj in relations])
                        print(f"[关系抽取] {chunk_id}: {rel_text}")

                for entity in entities:
                    if entity not in self.graph:
                        self.graph.add_node(entity, node_type="entity")
                        self.node_embeddings[entity] = self.embeddings_model.encode(entity)
                    self.graph.add_edge(entity, chunk_id, type="appears_in")

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
        query_embedding = self.embeddings_model.encode(query)
        entity_scores = {}
        for node in self.graph.nodes():
            if node in self.node_embeddings:
                similarity = self.cosine_similarity(query_embedding, self.node_embeddings[node])
                entity_scores[node] = similarity

        relevant_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        seen_chunks = set()

        for entity, score in relevant_entities:
            for _, chunk_id in self.graph.out_edges(entity):
                if chunk_id.startswith("chunk_") and chunk_id not in seen_chunks:
                    chunk_content = self.chunk_contents.get(chunk_id, "")
                    if chunk_content:
                        meta = self.chunk_metadata.get(chunk_id, {})
                        results.append(
                            {
                                "content": chunk_content,
                                "score": score,
                                "entity": entity,
                                "source": meta.get("source", "graph_search"),
                                "page": meta.get("page"),
                                "bbox": meta.get("bbox"),
                                "heading": meta.get("heading"),
                                "chunk_id": chunk_id,
                            }
                        )
                        seen_chunks.add(chunk_id)

            for _, neighbor in self.graph.out_edges(entity):
                if not neighbor.startswith("chunk_"):
                    edge_data = self.graph.get_edge_data(entity, neighbor)
                    relation = edge_data.get("relation", "related_to")
                    results.append(
                        {
                            "content": f"{entity} --{relation}--> {neighbor}",
                            "score": score * 0.8,
                            "entity": entity,
                            "source": "graph_relation",
                        }
                    )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def _normalize_ollama_text(self, response: Any) -> str:
        if response is None:
            return ""

        if isinstance(response, dict):
            text = response.get("response") or response.get("message", {}).get("content", "")
        else:
            text = getattr(response, "response", "") or getattr(getattr(response, "message", None), "content", "")

        text = (text or "").strip()
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def _build_fallback_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        evidence_lines = []
        answer_lines = []
        for i, item in enumerate(context[:3], 1):
            source = item.get("source") or item.get("pdf_source") or "unknown"
            page = item.get("page")
            heading = item.get("heading")
            location_bits = [f"来源={source}"]
            if page is not None:
                location_bits.append(f"页码={page}")
            if heading:
                location_bits.append(f"标题={heading}")

            snippet = re.sub(r"\s+", " ", item.get("content", "")).strip()
            evidence_lines.append(f"- [证据{i}] {' | '.join(location_bits)}")
            if snippet:
                summary = snippet[:220] + ("..." if len(snippet) > 220 else "")
                answer_lines.append(f"{i}. {summary} [证据{i}]")

        if not answer_lines:
            return (
                "一、证据归纳\n"
                "- 当前未检索到可用证据。\n\n"
                "二、回答\n"
                f"根据当前检索内容，无法回答“{query}”。"
            )

        return (
            "一、证据归纳\n"
            + "\n".join(evidence_lines)
            + "\n\n二、回答\n"
            + "\n".join(answer_lines)
            + "\n\n三、不确定性与适用条件\n"
            + "- 本回答由检索结果自动整理生成，因为答案模型返回了空内容；建议复核上述证据原文。"
        )

    def generate_answer(self, query: str, context: List[Dict[str, Any]], max_tokens: int = 800) -> Optional[str]:
        evidence_sections = []
        for i, item in enumerate(context, 1):
            source = item.get("source") or item.get("pdf_source") or "unknown"
            page = item.get("page")
            heading = item.get("heading")
            location_bits = [f"来源={source}"]
            if page is not None:
                location_bits.append(f"页码={page}")
            if heading:
                location_bits.append(f"标题={heading}")
            location = " | ".join(location_bits)
            evidence_sections.append(f"[证据{i}] ({location})\n{item.get('content', '')}")

        context_text = "\n\n".join(evidence_sections)
        prompt = _render_prompt_template(self.runtime_config["prompts"]["answer_generation"], context_text=context_text, query=query)
        answer_options = dict(self.runtime_config["ollama_options"]["answer_generation"])
        answer_options["num_predict"] = max_tokens

        try:
            response = ollama.generate(
                model=self.runtime_config["models"]["answer_generation"],
                prompt=prompt,
                options=answer_options,
            )
            answer_text = self._normalize_ollama_text(response)
            if answer_text:
                return answer_text

            print("答案模型返回空内容，已切换为基于检索证据的兜底答案。")
            return self._build_fallback_answer(query, context)
        except Exception as e:
            print(f"生成答案失败: {e}")
            if context:
                print("已切换为基于检索证据的兜底答案。")
                return self._build_fallback_answer(query, context)
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / denom)

    def save_graph(self):
        data = {
            "graph": self.graph,
            "node_embeddings": self.node_embeddings,
            "chunk_contents": self.chunk_contents,
            "chunk_metadata": self.chunk_metadata,
        }
        with open(self.graph_save_path, "wb") as f:
            pickle.dump(data, f)

    def load_graph(self):
        load_path = Path(self.graph_save_path)
        if load_path.exists():
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self.graph = data["graph"]
                self.node_embeddings = data["node_embeddings"]
                self.chunk_contents = data.get("chunk_contents", {})
                self.chunk_metadata = data.get("chunk_metadata", {})
            print(f"知识图谱已加载自: {load_path}")
        else:
            print(f"知识图谱文件不存在: {load_path}, 将重新构建图谱")
            raise FileNotFoundError(f"知识图谱文件不存在: {load_path}")


class ImprovedGraphRAG(GraphRAG):
    def __init__(self, data_dir="data", save_dir="model_files", config_path: str = "config/rag_config.json"):
        super().__init__(data_dir=data_dir, save_dir=save_dir, config_path=config_path)
        try:
            embedding_size = int(self.embeddings_model.get_sentence_embedding_dimension())
        except Exception:
            sample_vector = np.asarray(self.embeddings_model.encode("测试向量维度"))
            embedding_size = int(sample_vector.shape[-1])
        self.vector_store = VectorStore(
            embedding_size=embedding_size,
            index_path=self.save_dir / "vector_index.faiss",
        )
        self.retrieval_config = self.runtime_config.get("retrieval", {})

    def prepare_documents(self):
        self.processor.convert_pdfs_to_markdown(self.data_dir)

    def models_are_stale(self) -> bool:
        if not self.vector_store.index_path.exists() or not self.graph_save_path.exists():
            return True

        data_files = (
            list(self.data_dir.glob("*.md"))
            + list(self.data_dir.glob("*.json"))
            + list(self.data_dir.glob("*.pdf"))
        )
        if not data_files:
            return False

        latest_data_mtime = max(f.stat().st_mtime for f in data_files)
        latest_model_mtime = min(
            self.vector_store.index_path.stat().st_mtime,
            self.graph_save_path.stat().st_mtime,
        )
        return latest_data_mtime > latest_model_mtime

    def process_documents(self, show_entity_relations: bool = False):
        all_chunks: List[Dict[str, Any]] = []
        self.prepare_documents()
        md_files = sorted(self.data_dir.glob("*.md"), key=_natural_sort_key)
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

        if not all_chunks:
            raise RuntimeError("没有可用于建库的文本块，请检查 data 目录或 PDF 转换结果。")

        print("开始构建向量索引...")
        self.vector_store.reset()
        all_embeddings = self.processor.encoder.encode(
            [c["content"] for c in all_chunks],
            batch_size=32,
            show_progress_bar=True,
        )
        chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        self.vector_store.add_items(all_embeddings, chunk_ids)
        self.vector_store.save_index(self.vector_store.index_path)
        print("向量索引构建完成并保存。")

        self.build_graph(all_chunks=all_chunks, force_rebuild=True, show_entity_relations=show_entity_relations)

    def load(self):
        try:
            self.vector_store.load_index(self.vector_store.index_path)
            self.load_graph()
            print("模型加载完成 (包括向量索引和图谱)。")
        except FileNotFoundError as e:
            print(f"加载模型失败: {e}, 请先处理文档构建模型。")
            raise e

    def _keyword_overlap_score(self, query: str, content: str) -> float:
        query_terms = {w.strip() for w in jieba.cut(query) if len(w.strip()) > 1}
        if not query_terms:
            return 0.0
        content_terms = {w.strip() for w in jieba.cut(content) if len(w.strip()) > 1}
        if not content_terms:
            return 0.0
        overlap = query_terms & content_terms
        return len(overlap) / max(len(query_terms), 1)

    def _fuse_score(self, vector_score: float, overlap_score: float, graph_bonus: float = 0.0) -> float:
        return 0.72 * vector_score + 0.23 * overlap_score + 0.05 * graph_bonus

    def _neighbor_chunk_ids(self, chunk_id: str, window: Optional[int] = None) -> List[str]:
        if not chunk_id.startswith("chunk_"):
            return [chunk_id]
        if window is None:
            window = int(self.retrieval_config.get("neighbor_window", 1))
        try:
            idx = int(chunk_id.split("_")[1])
        except Exception:
            return [chunk_id]

        ids = []
        for i in range(idx - window, idx + window + 1):
            if i < 0:
                continue
            cand = f"chunk_{i}"
            if cand in self.chunk_contents:
                ids.append(cand)
        return ids or [chunk_id]

    def _expanded_chunk_content(self, chunk_id: str, window: Optional[int] = None) -> str:
        ids = self._neighbor_chunk_ids(chunk_id, window=window)
        sections = []
        chunk_idx = int(chunk_id.split("_")[1]) if chunk_id.startswith("chunk_") else 0
        for cid in ids:
            tag = "当前段"
            if cid != chunk_id and cid.startswith("chunk_"):
                neighbor_idx = int(cid.split("_")[1])
                tag = "前文段" if neighbor_idx < chunk_idx else "后文段"
            meta = self.chunk_metadata.get(cid, {})
            prefix_bits = [tag, cid]
            if meta.get("page") is not None:
                prefix_bits.append(f"页{meta['page']}")
            if meta.get("heading"):
                prefix_bits.append(f"标题:{meta['heading']}")
            prefix = " | ".join(prefix_bits)
            sections.append(f"【{prefix}】{self.chunk_contents.get(cid, '')}")
        return "\n".join(sections)

    def hybrid_search(self, query: str, top_k_vector: Optional[int] = None, top_k_graph: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k_vector is None:
            top_k_vector = int(self.retrieval_config.get("top_k_vector", 8))
        if top_k_graph is None:
            top_k_graph = int(self.retrieval_config.get("top_k_graph", 4))

        query_embedding = self.embeddings_model.encode(query)
        candidate_multiplier = int(self.retrieval_config.get("candidate_multiplier", 4))
        candidate_floor = int(self.retrieval_config.get("candidate_floor", 20))
        candidate_k = max(top_k_vector * candidate_multiplier, candidate_floor)
        relevant_chunk_ids, similarities = self.vector_store.search(query_embedding, top_k=candidate_k)

        vector_results_by_chunk: Dict[str, Dict[str, Any]] = {}
        for chunk_id, similarity in zip(relevant_chunk_ids, similarities):
            base_content = self.chunk_contents.get(chunk_id, "")
            if not base_content:
                continue
            expanded_content = self._expanded_chunk_content(chunk_id)
            vector_score = (_safe_float(similarity) + 1.0) / 2.0
            overlap_score = self._keyword_overlap_score(query, expanded_content)
            fused = self._fuse_score(vector_score=vector_score, overlap_score=overlap_score)
            meta = self.chunk_metadata.get(chunk_id, {})

            old_item = vector_results_by_chunk.get(chunk_id)
            if old_item is None or fused > old_item["score"]:
                vector_results_by_chunk[chunk_id] = {
                    "content": expanded_content,
                    "score": fused,
                    "source": meta.get("source", "vector_search"),
                    "pdf_source": meta.get("pdf_source"),
                    "page": meta.get("page"),
                    "bbox": meta.get("bbox"),
                    "heading": meta.get("heading"),
                    "extraction_mode": meta.get("extraction_mode"),
                    "chunk_id": chunk_id,
                    "vector_score": vector_score,
                    "overlap_score": overlap_score,
                }

        vector_results = sorted(vector_results_by_chunk.values(), key=lambda x: x["score"], reverse=True)[:top_k_vector]

        graph_results = []
        seen_entities = set()
        for result_item in vector_results:
            chunk_id = result_item["chunk_id"]
            related_entities = [entity for entity, _ in self.graph.in_edges(chunk_id) if not entity.startswith("chunk_")]
            for entity in related_entities:
                if entity in seen_entities:
                    continue
                seen_entities.add(entity)
                entity_info = self._get_entity_and_relations(entity, query, query_embedding, score_boost=result_item["score"])
                graph_results.extend(entity_info)

        combined_results = vector_results + graph_results
        dedup: Dict[str, Dict[str, Any]] = {}
        for item in combined_results:
            content = str(item.get("content", "")).strip()
            score = _safe_float(item.get("score", 0.0), 0.0)
            if not content or not np.isfinite(score):
                continue
            dedup_key = item.get("chunk_id") or content
            old_item = dedup.get(dedup_key)
            if old_item is None or score > old_item["score"]:
                dedup[dedup_key] = {**item, "content": content, "score": score}

        return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[: (top_k_vector + top_k_graph)]

    def _get_entity_and_relations(self, entity: str, query: str, query_embedding, score_boost: float = 1.0):
        entity_results = []
        if entity in self.node_embeddings:
            entity_embedding = self.node_embeddings[entity]
            entity_score = self.cosine_similarity(query_embedding, entity_embedding) * score_boost
        else:
            entity_score = 0.5 * score_boost

        entity_overlap = self._keyword_overlap_score(query, entity)
        if entity_overlap > 0:
            entity_results.append(
                {
                    "content": f"**实体**: {entity}",
                    "score": self._fuse_score(entity_score, entity_overlap),
                    "source": "graph_entity",
                    "entity": entity,
                }
            )

        for _, neighbor in self.graph.out_edges(entity):
            if neighbor.startswith("chunk_"):
                continue
            edge_data = self.graph.get_edge_data(entity, neighbor)
            relation = edge_data.get("relation", "related_to")
            relation_content = f"**关系**: {entity} --{relation}--> {neighbor}"
            relation_overlap = self._keyword_overlap_score(query, relation_content)
            relation_score = self._fuse_score(entity_score * 0.8, relation_overlap)
            entity_results.append(
                {
                    "content": relation_content,
                    "score": relation_score,
                    "source": "graph_relation",
                    "entity": entity,
                    "relation": relation,
                    "neighbor_entity": neighbor,
                }
            )
        return entity_results

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = int(self.retrieval_config.get("top_k_vector", 8))
        return self.hybrid_search(query, top_k_vector=top_k, top_k_graph=max(2, top_k // 2))
