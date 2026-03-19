# OpenDataLoader 风格改造说明

本次改造的目标不是复制 OpenDataLoader，而是把它最有价值的设计原则迁移到当前 Graph-RAG 项目中：

## 1. Local-first / Hybrid PDF 解析

原实现：
- 所有 PDF 页面都渲染为图片，再调用多模态模型识别为 Markdown。
- 普通文本 PDF 也要走视觉模型，速度慢、成本高、可重复性差。

现在：
- `pdf_conversion.mode` 支持 `local | llm | auto`
- `local`：使用 PyMuPDF 本地结构化抽取
- `llm`：强制逐页多模态识别
- `auto`：按页根据文字密度、文本块数量、字符清洁度自动路由

这相当于把 OpenDataLoader 的“deterministic local mode + complex page fallback”思路移植到了你的项目里。

## 2. 每页结构化 JSON 输出

原实现：
- 只保存 `xxx_i.md`
- 后续 chunking 只能把 Markdown 压平为纯文本

现在：
- 每页除 `xxx_i.md` 外，还会生成 `xxx_i.json`
- JSON 中包含：
  - `page_number`
  - `mode`（local / llm）
  - `elements[]`
  - 每个 element 的 `type / level / text / bbox / font_size`

这使后续系统可以做：
- 页码引用
- 前端高亮
- 证据定位
- chunk 级别结构保真

## 3. XY-Cut 风格阅读顺序排序

原实现：
- PDF 主要依赖多模态模型“看图转写”
- 没有显式的版面阅读顺序控制

现在：
- 本地抽取分支增加了 `XY-cut inspired` 排序
- 依据文本块 bbox 在 x / y 方向上的空白间隙递归分组
- 优先恢复“先上后下、再左后右”的阅读顺序

这不是完整的 OpenDataLoader XY-Cut++，但已经把“阅读顺序是版面分析问题，而不是后处理拼接问题”这一思路真正引入了项目。

## 4. Header / Footer 噪声抑制

现在会检测页面顶部 / 底部重复出现的文本块，并在 local 抽取时过滤常见页眉页脚，减少检索噪声。

## 5. chunk 元数据保留

原实现：
- chunk 只有纯文本内容和来源路径

现在：
- chunk 额外保留：
  - `page`
  - `bbox`
  - `heading`
  - `pdf_source`
  - `extraction_mode`

## 6. 检索结果可追溯

原实现：
- CLI 中只能看到来源和内容

现在：
- CLI 检索结果会显示：
  - 来源
  - PDF 文件名
  - 页码
  - 标题
  - BBox
  - 相关度

## 7. 向量索引修正

原实现：
- 维度硬编码为 768
- 重建索引时可能重复叠加数据
- 使用 L2 距离，但业务层做了余弦式归一化假设

现在：
- 自动读取 embedding 维度
- 重建前显式 `reset()`
- 索引切换为 `IndexFlatIP`，与归一化向量更匹配

## 8. 生成提示词增强

答案生成阶段会携带来源、页码、标题等证据信息，鼓励模型输出“基于证据的结论”，而不是只做内容拼接。

## 9. 当前边界

这次改造已经把项目从“PDF OCR + RAG”推进到“结构化 PDF RAG”，但还有几个后续增强点值得继续做：

1. 更强的表格结构恢复（当前仅使用 PyMuPDF `find_tables()` 做基础抽取）
2. 更完整的多栏阅读顺序算法（当前是 XY-cut inspired heuristic）
3. 页内图片 / 公式 / 图题的单独 element 建模
4. chunk 到原 PDF 高亮跳转的前端展示
5. 检索重排器（cross-encoder 或本地 reranker）

如果继续沿这个方向演进，你的项目会逐步接近“面向文档理解和可追溯问答”的系统，而不只是一个传统 Graph-RAG demo。
