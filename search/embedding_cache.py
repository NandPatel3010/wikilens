import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from .settings import EMBEDDER, EMBEDDER_REVISION

DEFAULT_CACHE = Path("data/embeddings.sqlite3")


class EmbeddingCache:
    def __init__(self, connection):
        self.connection = connection
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (key BLOB PRIMARY KEY, vector BLOB NOT NULL)
        """)

    def load_batch(self, model_inputs, embed):
        keys = [
            hashlib.sha256(f"{EMBEDDER}@{EMBEDDER_REVISION}:normalized:{text}".encode()).digest()
            for text in model_inputs
        ]
        distinct = dict(zip(keys, model_inputs, strict=True))
        placeholders = ", ".join("?" for _ in distinct)
        existing = {row[0] for row in self.connection.execute(
            f"SELECT key FROM embeddings WHERE key IN ({placeholders})", tuple(distinct)
        )}
        missing = {key: text for key, text in distinct.items() if key not in existing}
        if missing:
            vectors = embed(list(missing.values()), batch_size=len(missing))
            if vectors.shape != (len(missing), 384):
                raise ValueError(f"Unexpected embedding shape: {vectors.shape}")
            new_rows = [
                (key, np.asarray(vector, dtype="<f4").tobytes())
                for key, vector in zip(missing, vectors, strict=True)
            ]
            self.connection.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", new_rows)
        return keys, len(missing)


def open_cache(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=30)
