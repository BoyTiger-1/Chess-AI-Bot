from __future__ import annotations

import hashlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ai_business_assistant.shared.config import get_settings
from ai_business_assistant.worker.celery_app import celery_app


def _fake_embedding(text: str, *, dim: int = 64) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * ((dim * 4 // len(digest)) + 1))[: dim * 4]

    floats: list[float] = []
    for i in range(0, len(raw), 4):
        val = int.from_bytes(raw[i : i + 4], "big")
        floats.append((val % 10_000) / 10_000.0)
    return floats


@celery_app.task(name="ai_business_assistant.worker.tasks.embed_message")
def embed_message(*, message_id: str, content: str) -> dict[str, Any]:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)

    collection = "message_embeddings"
    dim = 64

    collections = {c.name for c in client.get_collections().collections}
    if collection not in collections:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    vector = _fake_embedding(content, dim=dim)
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=message_id,
                vector=vector,
                payload={"message_id": message_id},
            )
        ],
    )

    return {"message_id": message_id, "embedding_dim": dim}
