# 역할: 정책 문서를 BM25 또는 CPU Dense 검색으로 찾아 GPU 학습 루프에 전달합니다.
import math
import re
from collections import Counter


def lexical_tokens(text):
    # 형태소 분석기 없는 첫 실험용: 단어 + 한글 2음절 조각으로 조사 변화에 일부 대응합니다.
    words = re.findall(r"[가-힣a-z0-9]+", text.casefold())
    return words + [word[i:i + 2] for word in words if re.fullmatch(r"[가-힣]{3,}", word)
                    for i in range(len(word) - 1)]


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.counts = [Counter(lexical_tokens(d["contents"])) for d in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_len = max(sum(self.lengths) / len(documents), 1.0)
        df = Counter(term for counts in self.counts for term in counts)
        self.idf = {term: math.log(1 + (len(documents) - count + 0.5) / (count + 0.5))
                    for term, count in df.items()}

    def search(self, query, top_k=3):
        terms = set(lexical_tokens(query))
        scores = []
        for index, counts in enumerate(self.counts):
            norm = 1.5 * (0.25 + 0.75 * self.lengths[index] / self.avg_len)
            score = sum(self.idf.get(t, 0) * counts[t] * 2.5 / (counts[t] + norm)
                        for t in terms)
            scores.append((score, index))
        return [{**self.documents[i], "score": score}
                for score, i in sorted(scores, key=lambda x: (-x[0], x[1]))[:top_k]
                if score > 0]


class DenseRetriever:
    def __init__(self, documents, model_name):
        from sentence_transformers import SentenceTransformer
        self.documents = documents
        self.model = SentenceTransformer(model_name, device="cpu")
        self.embeddings = self.model.encode(
            ["passage: " + d["contents"] for d in documents],
            normalize_embeddings=True, show_progress_bar=True,
        )

    def search(self, query, top_k=3):
        vector = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
        scores = self.embeddings @ vector
        indices = scores.argsort()[::-1][:top_k]
        return [{**self.documents[i], "score": float(scores[i])} for i in indices]


def build_retriever(config, documents):
    if config.retriever == "sqlite_bm25":
        return documents
    if config.retriever == "bm25":
        return BM25Retriever(documents)
    return DenseRetriever(documents, config.dense_model)
