# 基于图谱增强的RAG电力系统知识应用（Graph-Enhanced RAG for Power System Knowledge）

本项目实现了一个检索增强生成（RAG）系统，并结合了知识图谱增强功能，专门用于查询与电力系统相关的信息。它结合了基于向量相似度搜索和基于图推理的优势，以提供更准确和全面的答案。

## 功能特性

*   **混合搜索：** 将向量相似度搜索（使用 FAISS）与图遍历相结合，以检索相关上下文。
*   **知识图谱：** 从文档中提取实体和关系，构建知识图谱（使用 NetworkX），捕捉概念之间的联系。
*   **文档处理：** 处理 Markdown 文件，智能地将文本分块以进行最佳索引。
*   **PDF 智能转 Markdown：** 支持将 PDF 自动按页转换为 Markdown（如 `xxx_1.md` 到 `xxx_N.md`），中间渲染图片保存到 `data/figs/`，并通过本地多模态模型（如 `qwen3-vl:8b/30b`）识别公式与附图说明。
*   **答案生成：** 使用大型语言模型（Ollama 和deepseek-r1:7b）根据检索到的上下文合成答案。
*   **持久化：** 保存和加载向量索引和知识图谱，以便高效地重复使用。
*   **错误处理：** 包含针对文档处理和 JSON 解析的健壮错误处理。
*   **可定制：** 轻松调整参数，例如检索结果的数量 (top_k)、温度和 LLM 模型。

## 安装

1.  **先决条件：**
    *   Python 3.7+
    *   安装并运行 Ollama。从 [https://ollama.ai/](https://ollama.ai/) 下载。
    *   安装deepseek的本地版本（根据你的电脑显卡配置，如果是4090可以选择更高的deepseek版本）
    ```bash
     ollama run deepseek-r1:7b
    ```

    *   安装所需的 Python 包：

        ```bash
        pip install -r requirements.txt
        ```
        创建一个 `requirements.txt` 文件，内容如下：
        ```
        networkx
        sentence-transformers
        markdown
        beautifulsoup4
        jieba
        numpy
        faiss-cpu  # 如果你有支持 CUDA 的 GPU，则使用 faiss-gpu
        ollama
        pymupdf
        ```
2.  **克隆仓库：**

    ```bash
    git clone <你的仓库 URL>
    cd <你的仓库名称>
    ```

3.  **将您的 Markdown/PDF 文档放入 `data/` 目录。** 我目前放了几篇发表在"深入理解电力系统"公众号的文章
4.  如果需要 PDF 转换能力，请先在 Ollama 中拉取并运行多模态模型，例如：
    ```bash
    ollama run qwen3-vl:8b
    ```

## Linux 环境详细操作流程

以下以 Ubuntu 为例，其他 Linux 发行版命令基本相同。

1. **安装系统依赖（可选但推荐）**

   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip python3-venv git
   ```

2. **克隆并进入项目目录**

   ```bash
   git clone <你的仓库 URL>
   cd <你的仓库名称>
   ```

3. **创建并激活虚拟环境（推荐）**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **安装 Python 依赖**

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. **准备 Ollama 与模型**

   ```bash
   ollama run deepseek-r1:7b
   ollama run qwen3-vl:8b
   ```

6. **运行程序**

   ```bash
   python main.py
   ```

---

## Windows 11 + VSCode 详细操作流程

以下流程适用于你当前这种在 VSCode 中直接运行 `.py` 的场景。

1. **在 VSCode 打开项目根目录**
   - 必须打开包含 `main.py` 和 `requirements.txt` 的目录。

2. **选择正确 Python 解释器**
   - 按 `Ctrl + Shift + P`，选择 `Python: Select Interpreter`。
   - 选择你的 Conda 环境解释器，例如：
     `C:/Users/1/.conda/envs/torch/python.exe`。

3. **打开 VSCode 终端并确认环境**

   ```powershell
   python -c "import sys; print(sys.executable)"
   python -c "import os; print(os.getcwd())"
   ```

   - 第一个命令应输出你选择的解释器路径。
   - 第二个命令应是项目根目录路径。

4. **安装依赖（务必在项目根目录执行）**

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   > 若出现 `No such file or directory: requirements.txt`，说明终端不在项目目录，请先 `cd` 到项目目录再执行。

5. **准备 Ollama 模型**

   ```powershell
   ollama run deepseek-r1:7b
   ollama run qwen3-vl:8b
   ```

6. **运行程序**

   ```powershell
   python main.py
   ```

7. **（可选）配置 VSCode 一键运行**
   - 创建 `.vscode/launch.json`，将 `program` 指向 `${workspaceFolder}/main.py`，`cwd` 设为 `${workspaceFolder}`。
   - 这样点击运行按钮时路径不会错位。

---

## 使用方法

> PDF 多模态转换默认使用 `qwen3-vl:8b`（无需 deepseek）。
> 为降低多页 PDF 的显存峰值，已默认开启“每页请求后释放模型驻留”（`keep_alive=0s`）。
> 对长页内容默认提高上下文与输出上限（`num_ctx=16384`, `num_predict=4096`）。


1.  **运行应用程序：**

    ```bash
    python main.py
    ```

2.  第一次运行脚本时，它将先把 `data/` 目录中的 PDF **按页**转换为 Markdown（例如 `某文档_1.md` 到 `某文档_N.md`，公式输出为 LaTeX，附图转为简要描述），并将每页渲染图片保存到 `data/figs/` 目录，再统一处理 Markdown 文档，构建向量索引和知识图谱，并将它们保存到 `model_files/` 目录。这可能需要一些时间，具体取决于您的文档大小和硬件。

3.  后续运行将加载预先构建的索引和图（速度更快）。

4.  脚本将提示您输入问题。键入您的问题并按 Enter 键。

5.  要退出，请输入“退出”。



### 检索与回答策略（新）

- 首次构建知识图谱时会打印每个 chunk 的实体与关系抽取结果（便于核验抽取质量）。
- 检索结果默认会携带“当前段 + 前文段 + 后文段”上下文，扩大证据覆盖范围。
- 最终回答提示词已加强：优先引用检索证据，结论后附 `[证据N]`，无法确定时明确说明。

### PDF 按页转换时的控制台输出

PDF 转换阶段会逐页打印处理进度，并显示与多模态模型的交互记录（请求信息与响应预览），便于调试：

- `开始逐页处理 xxx.pdf，共 N 页`
- `-> 处理第 i/N 页`
- `[LLM请求] 文件=..., 页=..., 模型=..., keep_alive=0s, prompt长度=...`
- `[LLM响应] 文件=..., 页=..., 返回长度=..., 预览=...`
- `[LLM异常]/[LLM重试]`：遇到 `503`、超时或“返回空内容”会自动重试（指数退避）
- `-> 第 i 页完成，输出: xxx_i.md，中间图像: figs/xxx_i.png`
- 若某页在重试后仍失败，会写入失败占位 Markdown，避免整本 PDF 处理中断



### 常见运行日志说明（基于实际运行样例）

- `bert.embeddings.position_ids | UNEXPECTED`：通常可忽略（不同任务/架构下的权重加载提示，不代表失败）。
- `跳过 PDF 转换（分页 Markdown 已是最新）`：表示增量预处理生效，PDF 未变化时不会重复转换。
- `模型已从本地加载` + 图谱节点/边数量：表示本地向量索引与图谱已成功复用。
- 若检索结果中出现 `graph_entity` 且内容较弱，系统会通过关键词匹配与去重过滤，尽量保留与问题更相关的证据段。

## 示例

### 提问

电力系统调度系统的历史？

### 回答

电力系统调度系统的早期历史可以分为以下几个关键时期：

1. **计算机应用的开端**（1970年）
   - 美国新英格兰电网是最早将计算机用于电网调度控制的案例。这一技术标志着调度自动化从手动监控向智能化管理的转变。

2. **中国起步与探索**
   - 1978年，南京自动化研究所研制的SD-176系统在京津唐电网应用，具备监控功能但未形成完整系统。
   - 浙江大学和电科院的研究团队对状态估计等高级应用进行了研究，并于1979-1980年间派队到魁北克进修，以提升调度自动化技术水平。

这些阶段奠定了中国电力系统调度技术的基础，并为后续的发展提供了重要参考。
