import os
from threading import Lock

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .settings import EMBEDDER, EMBEDDER_REVISION, RERANKER, RERANKER_REVISION


class Models:
    def __init__(self, *, load_reranker=True):
        torch.set_num_threads(int(os.getenv("SEARCH_TORCH_THREADS", "4")))
        self.embedder = SentenceTransformer(EMBEDDER, revision=EMBEDDER_REVISION, device="cpu")
        self.reranker = (
            CrossEncoder(RERANKER, revision=RERANKER_REVISION, device="cpu", max_length=512)
            if load_reranker else None
        )
        self.embedding_lock = Lock()
        self.reranking_lock = Lock()

    def embed(self, texts, *, batch_size=32):
        with self.embedding_lock:
            return self.embedder.encode(
                texts, batch_size=batch_size, normalize_embeddings=True,
                show_progress_bar=False,
            )

    def rerank(self, query, hits):
        pairs = [(query, f"{hit.title}. {hit.text}") for hit in hits]
        with self.reranking_lock:
            return self.reranker.predict(
                pairs, batch_size=16, activation_fn=torch.nn.Identity(),
                show_progress_bar=False,
            ).tolist()
