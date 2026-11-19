"""500-token document chunks, dense + BM25 retrieval with reciprocal rank fusion."""

import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .io import digest, write_json
from .types import Document


def chunk_documents(documents, chunk_tokens=500, overlap=50):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_tokens,
        chunk_overlap=overlap,
    )
    chunks = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc.text)):
            chunks.append(Document(f"{doc.id}-{i}", text, doc.app_id, doc.source))
    if not chunks:
        raise ValueError("RAG requires at least one nonempty document")
    return chunks


class SentenceEmbeddings:
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer

        # Keep GPU memory available for the response/judge/prompt model.
        self.model = SentenceTransformer(model, device="cpu")
        self.identity = model

    def encode(self, texts):
        # MiniLM accepts 256 tokens. Pool multiple windows so 500-token documents
        # are not silently truncated to their beginning by the embedding model.
        tokenizer = self.model.tokenizer
        window = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(False)
        windows, owners = [], []
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, max(len(ids), 1), window):
                windows.append(tokenizer.decode(ids[start : start + window]))
                owners.append(i)
        vectors = self.model.encode(windows, normalize_embeddings=True, show_progress_bar=False)
        result = np.stack([vectors[np.array(owners) == i].mean(axis=0) for i in range(len(texts))])
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


class OpenAIEmbeddings:
    def __init__(self, model):
        from openai import OpenAI

        self.client = OpenAI(timeout=90, max_retries=3)
        self.model = model
        self.identity = f"{self.client.base_url}:{model}"

    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), 64):
            response = self.client.embeddings.create(model=self.model, input=texts[start : start + 64])
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return np.asarray(vectors, dtype=np.float32)


def tokens(text):
    return re.findall(r"\w+", text.casefold()) or ["<empty>"]


class HybridRetriever:
    def __init__(self, chunks, embedder, directory):
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("No RAG chunks")
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokens(c.text) for c in chunks])
        self.fingerprint = digest(
            {"chunks": [asdict(c) for c in chunks], "embedding": embedder.identity, "pooling_version": 1}
        )
        path = Path(directory) / f"vectors-{self.fingerprint[:16]}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.vectors = np.load(path, allow_pickle=False)
        else:
            self.vectors = np.asarray(embedder.encode([c.text for c in chunks]), dtype=np.float32)
            temp = path.with_suffix(".tmp.npy")
            np.save(temp, self.vectors, allow_pickle=False)
            temp.replace(path)
        if self.vectors.ndim != 2 or len(self.vectors) != len(chunks) or not np.isfinite(self.vectors).all():
            raise ValueError("Invalid cached embedding matrix; remove the vectors cache and retry")
        self.vectors = self.vectors / np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)
        write_json(path.with_suffix(".json"), [asdict(c) for c in chunks])
        self._query_cache = {}

    def retrieve(self, query, app_id, top_k=4):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        key = (query, app_id, top_k)
        if key in self._query_cache:
            return self._query_cache[key]
        eligible = [i for i, c in enumerate(self.chunks) if c.app_id == app_id]
        if not eligible:
            return []
        query_vector = np.asarray(self.embedder.encode([query])[0])
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        dense = self.vectors @ query_vector
        sparse = self.bm25.get_scores(tokens(query))
        fused = {i: 0.0 for i in eligible}
        for values in (dense, sparse):
            for rank, i in enumerate(sorted(eligible, key=lambda j: (-float(values[j]), j)), 1):
                fused[i] += 1 / (60 + rank)
        chosen = sorted(eligible, key=lambda i: (-fused[i], i))[:top_k]
        if len(self._query_cache) >= 4096:  # A long-running server sees endless unique reviews.
            self._query_cache.clear()
        self._query_cache[key] = [self.chunks[i] for i in chosen]
        return self._query_cache[key]


def build_retriever(config, documents):
    chunks = chunk_documents(documents, config.chunk_tokens, config.chunk_overlap)
    if config.rag_backend == "azure":
        from .azure_rag import AzureRetriever

        return AzureRetriever(
            chunks, OpenAIEmbeddings(config.azure_embedding_model), config.azure_index_prefix
        )
    return HybridRetriever(
        chunks, SentenceEmbeddings(config.embedding_model), Path(config.output_dir) / "rag"
    )


def context_payload(chunks):
    return [{"id": c.id, "source": c.source, "text": c.text} for c in chunks]
