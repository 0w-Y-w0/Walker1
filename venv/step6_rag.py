# step6_rag.py
# 4.6 RAG 入门——sentence-transformers 向量化 + FAISS 检索 + 大模型回答
# 用途: 演示最小可用 RAG: 把知识切段→向量化存入 FAISS→按问题检索
#       top-k→拼接上下文让模型作答。当模型缺乏"私有知识"（如公司
#       退货政策）时，RAG 让回答有据可依。
# 关键参数:
#   model_name="BAAI/bge-small-zh-v1.5"  中文句向量模型（首次需联网下载权重）
#   top_k=2   检索最相关的 2 段
#   dimension 由模型决定（bge-small-zh 为 512）
# 预期输出: 打印检索到的片段与模型基于片段给出的回答（而非凭空编）。
# 常见报错与解决:
#   - 下载模型失败/被墙 -> 设镜像: os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
#   - FAISS IndexFlatL2 维度不匹配 -> 确保 embedding 维度与 index 创建时一致。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像, 加速下载

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from step1_client import client, model

# 1) 私有知识库（示例: 某公司退货政策条款）
knowledge = [
    "会员在收货后 7 天内可申请无理由退货，需保持商品完好。",
    "非会员退货期限为收货后 3 天，且需提供购买凭证。",
    "食品、定制类商品一经售出不支持退换。",
    "退货审核通过后，退款将在 3 个工作日内原路返回。",
]

# 2) 向量化并建索引
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
doc_vectors = embedder.encode(knowledge, normalize_embeddings=True)
dim = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.asarray(doc_vectors, dtype="float32"))

# 3) 用户提问 -> 检索
question = "我是会员，昨天收到的衣服想退，可以吗？"
q_vec = embedder.encode([question], normalize_embeddings=True)
_, idx = index.search(np.asarray(q_vec, dtype="float32"), k=2)
retrieved = [knowledge[i] for i in idx[0]]
print("检索到的相关条款:")
for t in retrieved:
    print("  -", t)

# 4) 拼接上下文让模型回答
context = "\n".join(retrieved)
rag_prompt = f"根据以下政策条款回答问题，不要超出条款范围。\n条款:\n{context}\n\n问题:{question}"
ans = client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": "你是客服助手，严格依据给定条款回答。"},
              {"role": "user", "content": rag_prompt}],
    temperature=0.0, max_tokens=200)
print("\n模型回答:", ans.choices[0].message.content)
