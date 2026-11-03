"""Citable retrieval with deterministic lexical embeddings and Qdrant storage."""

import hashlib
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import HashingVectorizer

DIMENSIONS = 512
COLLECTION = "maintenance_docs"
EMBEDDER = HashingVectorizer(
    n_features=DIMENSIONS, alternate_sign=False, norm="l2", ngram_range=(1, 2), stop_words="english"
)


def chunks(text, size=180, overlap=35):
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield " ".join(words[start : start + size])


class KnowledgeBase:
    def __init__(self, url=None, path=None):
        self.client = (
            QdrantClient(url=url, timeout=5)
            if url
            else QdrantClient(path=str(path))
            if path
            else QdrantClient(":memory:")
        )
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(
                    size=DIMENSIONS, distance=models.Distance.COSINE
                ),
            )

    def ingest(self, directory):
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Document directory not found: {directory}")
        count = 0
        for file in sorted(Path(directory).rglob("*")):
            if file.suffix.lower() not in (".txt", ".md", ".pdf"):
                continue
            if file.suffix.lower() == ".pdf":
                from pypdf import PdfReader

                pages = [
                    (i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(file).pages)
                ]
            else:
                pages = [(1, file.read_text(encoding="utf-8"))]
            source = file.relative_to(directory).as_posix()
            # Re-ingestion replaces the source, including chunks removed by an edit.
            self.client.delete(
                COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source", match=models.MatchValue(value=source)
                            )
                        ]
                    )
                ),
            )
            for page, text in pages:
                for index, chunk in enumerate(chunks(text)):
                    identity = hashlib.sha256(f"{source}:{page}:{index}".encode()).hexdigest()
                    vector = EMBEDDER.transform([chunk]).toarray()[0].tolist()
                    self.client.upsert(
                        COLLECTION,
                        [
                            models.PointStruct(
                                id=str(uuid.UUID(identity[:32])),
                                vector=vector,
                                payload={
                                    "source": source,
                                    "page": page,
                                    "chunk": index,
                                    "text": chunk,
                                    "content_sha256": hashlib.sha256(chunk.encode()).hexdigest(),
                                },
                            )
                        ],
                    )
                    count += 1
        return count

    def search(self, query, limit=3):
        if not query.strip():
            return []
        query_words = set(re.findall(r"[a-z]{3,}", query.lower()))
        hits = self.client.query_points(
            COLLECTION,
            query=EMBEDDER.transform([query]).toarray()[0].tolist(),
            limit=limit,
            score_threshold=0.08,
        ).points
        # Shared vocabulary is required so hashing collisions alone cannot create evidence.
        return [
            {**hit.payload, "score": float(hit.score)}
            for hit in hits
            if query_words & set(re.findall(r"[a-z]{3,}", hit.payload["text"].lower()))
        ]

    def close(self):
        self.client.close()
