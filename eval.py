"""检索效果评测：同一评测集上比较 BM25 / 字符 TF-IDF / RRF 混合三路。"""
import json

from rag import Retriever

# (查询, 相关文档 id)。标注口径：能直接回答该问题的文档。
QUERIES = [
    ("RAG 和微调应该怎么选", ["r02"]),
    ("检索增强生成的完整链路有哪些步骤", ["r03"]),
    ("朴素 RAG 为什么召回不准", ["r04"]),
    ("多轮对话里用户说那它呢，检索不到怎么办", ["g05"]),
    ("HyDE 假设文档检索是什么原理", ["r06"]),
    ("稀疏检索和稠密检索为什么要一起用", ["r07"]),
    ("回答怎么绑定来源片段做溯源", ["r08"]),
    ("扫描版 PDF 怎么解析", ["c01"]),
    ("固定窗口分块的重叠比例一般设多少", ["c02"]),
    ("父子分块的子块和父块大小怎么定", ["c04"]),
    ("分块是不是越小越好", ["c05"]),
    ("表格和代码块应该怎么切分", ["c06"]),
    ("每个块需要带哪些元数据", ["c07"]),
    ("页眉页脚和重复段落怎么清理", ["c08"]),
    ("中文 embedding 模型怎么选型", ["v01"]),
    ("余弦相似度和内积是什么关系", ["v03"]),
    ("FAISS 和 Milvus 应该怎么选", ["v04"]),
    ("HNSW 的 M 和 efSearch 参数是什么", ["v05"]),
    ("向量数量太多内存放不下怎么办", ["v06"]),
    ("文档删除后向量索引怎么同步", ["v07"]),
    ("向量为什么必须做 L2 归一化", ["v08"]),
    ("BM25 的 k1 和 b 怎么调", ["s01"]),
    ("RRF 倒数排名融合的公式是什么", ["s02"]),
    ("cross-encoder 重排为什么比向量召回慢", ["s03"]),
    ("混合检索两路分数怎么融合", ["s04"]),
    ("召回阶段取多少条比较合适", ["s05"]),
    ("中文分词质量会影响 BM25 吗", ["s06"]),
    ("用户说的词和文档用词不一致怎么办", ["s07"]),
    ("top-k 结果重复度太高怎么处理", ["s08"]),
    ("送给模型的上下文应该怎么排序", ["g01"]),
    ("系统提示词应该怎么写约束", ["g02"]),
    ("怎么抑制大模型幻觉", ["g03"]),
    ("上下文是不是越长效果越好", ["g04"]),
    ("怎么让模型稳定输出 JSON", ["g06"]),
    ("流式输出时引用信息怎么返回", ["g07"]),
    ("Recall@K 指标是什么意思", ["e01"]),
    ("MRR 平均倒数排名怎么算", ["e02"]),
    ("nDCG 和 Recall 有什么区别", ["e03"]),
    ("端到端评测要看哪些指标", ["e04"]),
    ("RAG 评测集应该怎么构建", ["e05"]),
    ("用大模型当裁判打分靠谱吗", ["e06"]),
    ("怎么用 FastAPI 暴露检索服务", ["o01"]),
    ("索引更新后查询缓存怎么失效", ["o02"]),
    ("重排模型超时了怎么降级", ["o03"]),
    ("知识库文档里藏了恶意指令怎么办", ["a01"]),
    ("检索前怎么按用户权限过滤", ["a02"]),
    ("多租户知识库怎么隔离", ["a05"]),
    ("审计日志需要记录哪些内容", ["a07"]),
]


def evaluate(mode: str, k: int = 5) -> dict:
    retriever = Retriever()
    hits = 0
    rr = 0.0
    misses = []
    for query, gold in QUERIES:
        ranked = [d["doc_id"] for d in retriever.search(query, k=20, mode=mode)]
        topk = ranked[:k]
        if any(g in topk for g in gold):
            hits += 1
        else:
            misses.append({"query": query, "gold": gold, "top5": topk})
        first = next((i for i, d in enumerate(ranked) if d in gold), None)
        rr += 1.0 / (first + 1) if first is not None else 0.0
    n = len(QUERIES)
    return {
        "mode": mode,
        "queries": n,
        "k": k,
        "recall_at_k": round(hits / n, 4),
        "mrr_at_20": round(rr / n, 4),
        "misses": misses,
    }


def main():
    results = [evaluate(m) for m in ("bm25", "tfidf", "hybrid")]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
