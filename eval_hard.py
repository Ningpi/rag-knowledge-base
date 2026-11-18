"""硬查询集：口语化、间接问法、换词表达，用来暴露真实检索差距。"""
import json

from rag import Retriever

HARD_QUERIES = [
    ("公司想把内部资料接进大模型，又不想重新训练参数", ["r02"]),
    ("文档天天更新，难道每次都要重新训一遍", ["r02"]),
    ("搜出来的前几条总是答非所问，问题出在哪", ["r04"]),
    ("回答里的话在原文里找不到出处，怎么让结论可核对", ["r08"]),
    ("几十页的扫描件读不出文字", ["c01"]),
    ("一段话被从中间截断了，答案总是不完整", ["c02"]),
    ("内容切得太碎，模型看不懂上下文", ["c05"]),
    ("表格被拆成好几行，检索出来缺表头", ["c06"]),
    ("同一份文件存了好几个版本，搜出来总是重复", ["c08"]),
    ("用户换个说法就搜不到了", ["s07"]),
    ("前五条全是同一段话的重复", ["s08"]),
    ("模型编造了资料里根本没有的数字", ["g03"]),
    ("明明资料里有，模型却说没有", ["g04"]),
    ("怎么判断到底是检索错了还是生成错了", ["e04"]),
    ("改了参数以后效果变好还是变坏，说不清", ["e05"]),
    ("上线之后没人用，怎么知道有没有价值", ["e07"]),
    ("文档里有人写了忽略以上指令，模型照做了", ["a01"]),
    ("普通员工搜到了高管薪酬文档", ["a02"]),
    ("两个客户的资料混在一起了", ["a05"]),
    ("回答太慢，用户等不及", ["g07"]),
    ("索引重建的时候查询一直抖", ["o06"]),
    ("大模型调用太贵，成本压不下来", ["o07"]),
    ("向量太多，机器内存撑不住", ["v06"]),
    ("删掉的文件还能被搜出来", ["v07"]),
    ("相似度算出来和预期对不上", ["v03"]),
    ("换了一个模型效果差很多，怎么选", ["v01"]),
    ("词被切坏了，搜不到东西", ["s06"]),
    ("一句话里问了好几件事，检索只命中一件", ["r05"]),
    ("多轮聊天，第二句就不知道在说谁", ["g05"]),
    ("资料明明给了，模型却漏掉了中间那段", ["g04"]),
]


def main():
    retriever = Retriever()
    out = []
    for mode in ("bm25", "tfidf", "hybrid", "hybrid_rm3"):
        hits5 = hits1 = 0
        rr = 0.0
        misses = []
        for q, gold in HARD_QUERIES:
            ranked = [d["doc_id"] for d in retriever.search(q, k=20, mode=mode)]
            if any(g in ranked[:5] for g in gold):
                hits5 += 1
            else:
                misses.append({"q": q, "gold": gold, "top5": ranked[:5]})
            if any(g in ranked[:1] for g in gold):
                hits1 += 1
            first = next((i for i, d in enumerate(ranked) if d in gold), None)
            rr += 1.0 / (first + 1) if first is not None else 0.0
        n = len(HARD_QUERIES)
        out.append({
            "mode": mode, "queries": n,
            "recall_at_1": round(hits1 / n, 4),
            "recall_at_5": round(hits5 / n, 4),
            "mrr_at_20": round(rr / n, 4),
            "misses": misses,
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("eval_hard_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
