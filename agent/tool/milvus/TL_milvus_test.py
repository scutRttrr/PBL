import json

from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np

# 1. 连接 Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("EducationModules")
collection.load()

dense_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

query_text="stakeholder map"

query_vector = dense_embeddings.embed_query(query_text)
print(len(query_vector))


search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_vector], # 用第一条数据的向量查自己
    anns_field="ilo_vector",
    param=search_params,
    limit=5,
    output_fields=["ilo","tc"]
)

print(f"\n查询内容: '{query_text}'")
for hits in results:
    for hit in hits:
        print("-" * 30)
        print(f"相似度距离 (Distance): {hit.distance:.4f}")
        print(f"ILO 内容: {hit.entity.get('ilo')}")
        print(f"TC 内容: {hit.entity.get('tc')[:100]}...") # 截取前100字展示