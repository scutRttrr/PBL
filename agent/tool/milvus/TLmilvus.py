# import json
#
# from langchain_huggingface import HuggingFaceEmbeddings
# from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
# import numpy as np
#
# # 1. 连接 Milvus
# connections.connect("default", host="localhost", port="19530")
#
# # 2. 定义 Schema
# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="ilo", dtype=DataType.VARCHAR, max_length=1000),
#     FieldSchema(name="tc", dtype=DataType.VARCHAR, max_length=3000),
#     # 假设向量维度为 768 (例如使用 BERT 模型)
#     FieldSchema(name="ilo_vector", dtype=DataType.FLOAT_VECTOR, dim=512)
# ]
# schema = CollectionSchema(fields, "Education content collection")
# collection = Collection("EducationModules", schema)
#
# # 3. 准备数据
#
#
# data=[]
# ilo_data=[]
# tc_data=[]
# ilo_vectors=[]
# dense_embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )
#
#
# try:
#     with open('/Users/gzlzhao/PycharmProjects/PBL/data/distill_pair.jsonl', 'r', encoding='utf-8') as f:
#         for line in f:
#             # 去掉行尾换行符并解析
#             item = json.loads(line.strip())
#             ilo_data.append(item["ILO"])
#             tc_data.append(item["TC"])
#
#         ilo_vectors = dense_embeddings.embed_documents(ilo_data)
#         data = [{"ilo":ilo_data[i],"tc":tc_data[i],"ilo_vector":ilo_vectors[i]} for i in range(len(ilo_data))]
#
# except FileNotFoundError:
#         print(f"错误：找不到 JSON 文件rubrics_dataset.json")
# except json.JSONDecodeError:
#     print(f"错误：rubrics_dataset.json 不是合法的 JSON 格式")
# except KeyError as e:
#     print(f"错误：JSON 配置中缺失字段 {e}")
#
# # 4. 插入并创建索引
# collection.insert(data)
# index_params = {
#     "metric_type": "L2",
#     "index_type": "IVF_FLAT",
#     "params": {"nlist": 128}
# }
# collection.create_index(field_name="ilo_vector", index_params=index_params)
# collection.load()
#

from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
# 1. 获取已有的集合
collection = Collection("EducationModules")

# 2. 释放加载，准备修改索引
collection.release()

# 3. 删除旧的 L2 索引（如果原来有索引的话）
try:
    collection.drop_index()
    print("旧索引已成功删除")
except Exception as e:
    print(f"删除索引时出错或索引不存在: {e}")

# 4. 创建全新的 COSINE 索引
# 注意：COSINE 是处理重写后“短语匹配”最理想的度量方式
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="ilo_vector",  # 请确保这和你 schema 里的向量字段名一致
    index_params=index_params
)
print("COSINE 索引创建成功")

# 5. 重新加载，使检索生效
collection.load()
print("集合已重新加载，现在可以使用余弦相似度检索了")