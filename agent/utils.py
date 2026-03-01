import uuid
from typing import List, Dict, Any, Optional, Union, Literal
from langchain_core.documents import Document


def new_uuid() -> str:
    return str(uuid.uuid4())


def _generate_uuid(seed: str) -> str:
    """
    Generate a deterministic UUID based on the provided seed.

    Falls back to a random UUID when the seed is empty.
    """
    if not seed:
        return new_uuid()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def reduce_docs(
        existing: Optional[list[Document]],
        new: Union[
            list[Document],
            list[dict[str, Any]],
            list[str],
            str,
            Literal["delete"],
        ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": _generate_uuid(new)})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid") or _generate_uuid(
                    item.get("page_content", "")
                )

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**{**item, "metadata": {**metadata, "uuid": item_id}})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid", "")
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list


def format_docs(docs: List[Document]) -> str:
    """将文档列表格式化为字符串

    Args:
        docs: 文档列表

    Returns:
        格式化后的字符串
    """
    if not docs:
        return "无相关文档。"

    formatted_docs = []
    for i, doc in enumerate(docs):
        content = doc.page_content
        metadata = doc.metadata
        source = metadata.get("source", "未知来源")
        formatted_docs.append(f"[{i + 1}] {content}\n来源: {source}\n")

    return "\n".join(formatted_docs)


def validate_generation_tasks(tasks: list[dict]) -> str:
    """
    校验 PBL 任务列表中的生成任务冲突
    逻辑：
    1. 统计生成类任务的数量
    2. 若 T&L-generation 与 rubric-generation 同时存在 -> fail
    3. 若只有一个生成任务 -> 返回对应类型
    4. 若只有问答任务 -> 返回默认执行路径
    """
    # 提取所有生成类任务的类型
    gen_tasks = [t['type'] for t in tasks if t['type'] in ['T&L-generation', 'rubric-generation']]

    # 转换为集合去重，方便判断
    gen_set = set(gen_tasks)

    # 校验逻辑
    # 同时含有 rubric-generation 和 T&L-generation
    if 'rubric-generation' in gen_set and 'T&L-generation' in gen_set:
        return "fail"

    # 只含有一个 T&L-generation
    if 'T&L-generation' in gen_set and len(gen_set) == 1:
        return "T&L-generation"

    # 只含有一个 rubric-generation
    if 'rubric-generation' in gen_set and len(gen_set) == 1:
        return "rubric-generation"

    # 如果没有生成任务（全是 concept-query 等问答）
    return "pure-query"




def interrupt(data: Dict[str, Any]) -> str:
    """中断函数，用于人工干预

    Args:
        data: 中断数据

    Returns:
        人工干预的结果
    """
    return data.get("question", "")