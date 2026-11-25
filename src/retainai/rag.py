"""RAG over the retention policy documents, stored in Chroma."""
from __future__ import annotations

import re
from pathlib import Path

import chromadb

from . import config

COLLECTION = "retention_policies"


class HashingEmbedding(chromadb.EmbeddingFunction):
    """Offline bag-of-words embedding (no model download). Used in tests and when
    RETAINAI_EMBEDDINGS=hashing. The default Chroma model (all-MiniLM-L6-v2) is better."""

    def __init__(self, n_features: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vec = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2",
                                      stop_words="english", ngram_range=(1, 2))

    def __call__(self, input):  # noqa: A002 - Chroma's required signature
        return [row.toarray()[0].astype("float32") for row in self._vec.transform(list(input))]

    @staticmethod
    def name() -> str:
        return "retainai-hashing"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbedding":
        return HashingEmbedding()


def default_embedding():
    import os
    return HashingEmbedding() if os.getenv("RETAINAI_EMBEDDINGS") == "hashing" else None


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a policy doc into one chunk per '## ' section, each prefixed with the doc title.

    Sections are short and self-contained (one offer, one rule), so section-level chunks keep
    eligibility rules together with the offer they belong to.
    """
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source
    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    chunks = []
    for i, part in enumerate(parts[1:], start=1):
        heading, _, body = part.partition("\n")
        body = body.strip()
        if body:
            chunks.append({
                "id": f"{source}#{i}",
                "text": f"{title} > {heading.strip()}\n{body}",
                "metadata": {"source": source, "section": heading.strip()},
            })
    return chunks


def load_chunks(policy_dir: Path | None = None) -> list[dict]:
    policy_dir = policy_dir or config.POLICY_DIR
    chunks: list[dict] = []
    for path in sorted(policy_dir.glob("*.md")):
        chunks += chunk_markdown(path.read_text(encoding="utf-8"), path.name)
    return chunks


def _client(persist_dir: Path | None, ephemeral: bool = False):
    if ephemeral:
        return chromadb.EphemeralClient()
    persist_dir = persist_dir or config.CHROMA_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def build_index(policy_dir: Path | None = None, persist_dir: Path | None = None,
                embedding_function=None, ephemeral: bool = False):
    """(Re)build the collection. Uses Chroma's default local embedding model unless one is passed."""
    embedding_function = embedding_function or default_embedding()
    client = _client(persist_dir, ephemeral)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    kwargs = {"embedding_function": embedding_function} if embedding_function else {}
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}, **kwargs)
    chunks = load_chunks(policy_dir)
    col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks])
    return col if ephemeral else len(chunks)


class PolicyRetriever:
    def __init__(self, collection=None, persist_dir: Path | None = None, embedding_function=None):
        if collection is None:
            embedding_function = embedding_function or default_embedding()
            kwargs = {"embedding_function": embedding_function} if embedding_function else {}
            collection = _client(persist_dir).get_collection(COLLECTION, **kwargs)
        self.col = collection

    def search(self, query: str, k: int = 3) -> list[dict]:
        res = self.col.query(query_texts=[query], n_results=k)
        return [
            {"source": m["source"], "section": m["section"], "text": d, "distance": round(dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
