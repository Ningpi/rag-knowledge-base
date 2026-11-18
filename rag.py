"""混合检索核心：词级 BM25 + 字符二元组 TF-IDF，RRF 融合。

设计取舍：
- BM25 走 jieba 词级，擅长专有名词、编号、术语的精确匹配；
- TF-IDF 走字符二元组，不依赖分词质量，擅长同义改写与错别字兜底；
- 两路排名用 RRF 融合，只依赖名次，不需要把两种分数归一化。
"""
import math
import re
from collections import Counter

import jieba

from corpus import DOCS

STOPWORDS = set(
    "的 了 是 在 和 与 或 及 等 这 那 有 为 对 从 到 中 上 下 也 就 都 而 被 把 让 给 "
    "一个 一种 我们 你 我 他 它 可以 需要 进行 通过 以及 如果 因为 所以 什么 怎么 哪些 "
    "一种 这个 那个 是不是 有没有".split()
)


def word_tokens(text: str) -> list[str]:
    out = []
    for t in jieba.lcut(text.lower()):
        t = t.strip()
        if len(t) < 2 or t in STOPWORDS or re.fullmatch(r"[\W_]+", t):
            continue
        out.append(t)
    return out


def char_bigrams(text: str) -> list[str]:
    s = re.sub(r"\s+", "", text.lower())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.N)
        self.tfs = [Counter(d) for d in docs]
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf in self.tfs:
            dl = sum(tf.values())
            s = 0.0
            for t in query:
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out


class TfidfIndex:
    def __init__(self, docs: list[list[str]]):
        self.N = len(docs)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log((self.N + 1) / (n + 1)) + 1 for t, n in df.items()}
        self.vecs = [self._vector(d) for d in docs]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {
            t: (1 + math.log(f)) * self.idf[t]
            for t, f in tf.items() if t in self.idf
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def scores(self, query: list[str]) -> list[float]:
        qv = self._vector(query)
        return [sum(qv.get(t, 0.0) * w for t, w in vec.items()) for vec in self.vecs]


def rank(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    fused: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(fused, key=lambda i: fused[i], reverse=True)


class Retriever:
    def __init__(self):
        self.docs = DOCS
        self.ids = [d[0] for d in self.docs]
        self.titles = [d[2] for d in self.docs]
        self.texts = [d[3] for d in self.docs]
        self.cats = [d[1] for d in self.docs]
        self.bm25 = BM25([
            word_tokens(self.titles[i] + " " + self.titles[i] + " " + self.texts[i])
            for i in range(len(self.texts))
        ])
        self.tfidf = TfidfIndex([
            char_bigrams(self.titles[i] + " " + self.texts[i])
            for i in range(len(self.texts))
        ])

    def rm3_expand(self, query_tokens: list[str], top_k: int = 8, n_terms: int = 10) -> list[str]:
        """RM3 伪相关反馈：用首轮 top-k 文档的高权重词扩充查询，缓解口语化问法与原文措辞不一致。"""
        scores = self.bm25.scores(query_tokens)
        order = rank(scores)[:top_k]
        weights: Counter = Counter()
        for i in order:
            if scores[i] <= 0:
                continue
            for term, freq in self.bm25.tfs[i].items():
                weights[term] += scores[i] * freq * self.bm25.idf.get(term, 0.0)
        seen = set(query_tokens)
        expansion = [t for t, _ in weights.most_common(n_terms * 2) if t not in seen][:n_terms]
        # 原查询权重高于扩展词：原文重复两次，扩展词一次
        return query_tokens * 2 + expansion

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        q_words = word_tokens(query)
        if mode == "bm25":
            order = rank(self.bm25.scores(q_words))
        elif mode == "tfidf":
            order = rank(self.tfidf.scores(char_bigrams(query)))
        elif mode == "hybrid_rm3":
            order = rrf([
                rank(self.bm25.scores(self.rm3_expand(q_words))),
                rank(self.tfidf.scores(char_bigrams(query))),
            ])
        else:
            order = rrf([
                rank(self.bm25.scores(q_words)),
                rank(self.tfidf.scores(char_bigrams(query))),
            ])
        return [
            {"doc_id": self.ids[i], "title": self.titles[i],
             "category": self.cats[i], "text": self.texts[i]}
            for i in order[:k]
        ]

    def answer(self, query: str, k: int = 3) -> dict:
        """抽取式回答：返回最相关片段并附引用，避免无依据生成。"""
        hits = self.search(query, k=k)
        if not hits:
            return {"answer": "知识库中未找到相关内容。", "citations": []}
        body = "\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
        return {
            "answer": f"根据知识库检索到以下依据：\n{body}",
            "citations": [{"id": h["doc_id"], "title": h["title"]} for h in hits],
        }
