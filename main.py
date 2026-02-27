from pathlib import Path

if __name__ == "__main__":
    try:
        from graph_rag import ImprovedGraphRAG  # 导入 ImprovedGraphRAG 类
    except ModuleNotFoundError as e:
        print(f"依赖缺失，无法启动程序: {e}")
        print("请先安装依赖，例如: pip install -r requirements.txt")
        raise

    data_dir = "data"  # 数据目录
    save_dir = "model_files" # 模型保存目录
    graph_rag = ImprovedGraphRAG(data_dir=data_dir, save_dir=save_dir) # 初始化 ImprovedGraphRAG

    # 先预处理 PDF -> Markdown，再决定是否重建模型
    graph_rag.prepare_documents()

    # 检查是否已存在模型文件
    try:
        graph_rag.load() # 尝试加载模型
        if graph_rag.models_are_stale():
            print("检测到 data 目录文档有更新，将重新构建模型...")
            graph_rag.process_documents() #  处理文档并构建模型
        else:
            print("模型已从本地加载 (包括向量索引和图谱).")
            print(f"图谱信息：{graph_rag.graph.number_of_nodes()} 节点, {graph_rag.graph.number_of_edges()} 边")
    except FileNotFoundError: #  捕获 FileNotFoundError 异常
        print("未找到已保存的模型，开始处理文档并构建模型...")
        graph_rag.process_documents() #  处理文档并构建模型


    while True:  # 循环接收用户查询
        query = input("\n请输入您的问题（输入 '退出' 结束）：")
        if query.lower() == '退出':
            break

        try:
            print(f"问题：{query}")

            # 检索相关内容 (使用混合搜索)
            relevant_context = graph_rag.search(query)
            print("\n检索到的相关内容：")
            for item in relevant_context:
                print(f"- 来源：{item['source']}") #  显示结果来源
                print(f"- 相关度：{item['score']:.3f}")
                print(f"- 内容：{item['content']}")
                print()

            # 生成答案
            answer = graph_rag.generate_answer(query, relevant_context)
            if answer:
                print("\n生成的答案：")
                print(answer)

        except Exception as e:
            print(f"程序执行出错: {e}")