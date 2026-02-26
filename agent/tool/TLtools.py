from langchain_core.tools import tool

from infrastructure.logger import get_logger

# 构建日志记录器
logger = get_logger(service="TL_tool")

@tool
def search_milvus(query: str):
    """在本地教学知识库中搜索关于 ILO（预期学习成果）和课程资料的内容。"""
    # 这里放你原有的 Milvus 检索逻辑
    logger.info(f"search_milvus: {query}")
    return f"从知识库找到关于 {query} 的 3 条量规参考资料..."

# @tool
# def web_search(query: str):
#     """当本地知识库信息不足时，搜索互联网获取最新的教育标准或定义。"""
#     return f"互联网搜索结果：{query} 的最新定义是..."

# 将工具放入列表
tltools = [search_milvus]