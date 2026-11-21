import argparse
import hashlib
import json
from contextlib import closing
from pathlib import Path

import numpy as np
from .chunks import INDEX_PROFILES, chunk_article
from .dataset import DEFAULT_DATA
from .embedding_cache import DEFAULT_CACHE, EmbeddingCache, open_cache
from .index_release import activate, index_update, retain_previous, rollback


def prepare(source, cache, models, batch_size, *, profile="plain"):
    tokenizer = models.embedder.tokenizer
    max_length = models.embedder.max_seq_length
    batch = []
    generated = 0
    passages = 0

    def stage():
        keys, new = cache.load_batch([row[3] for row in batch], models.embed)
        cache.connection.executemany("INSERT INTO staged VALUES (?, ?, ?, ?)",
                                     [(*row[:3], key) for row, key in zip(batch, keys, strict=True)])
        cache.connection.commit()
        batch.clear()
        return new

    fingerprint = hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            fingerprint.update(line)
            row = json.loads(line)
            cache.connection.execute("INSERT INTO staged_documents VALUES (?, ?, ?, ?)",
                                     (row["id"], row["title"], row["text"], row["url"]))
            chunks = chunk_article(row["title"], row["text"], tokenizer, max_length,
                                   contextual=profile == "contextual")
            for ordinal, (text, model_input) in enumerate(chunks):
                batch.append((row["id"], ordinal, text, model_input))
                passages += 1
                if len(batch) == batch_size:
                    generated += stage()
                if passages % 1000 == 0:
                    print(f"Prepared {passages:,} passages ({generated:,} newly embedded).", flush=True)
    if batch:
        generated += stage()
    if not passages:
        raise ValueError("Corpus has no passages.")
    return passages, generated, fingerprint.hexdigest()


def write_index(documents, staged, fingerprint, profile, document_count, passage_count):
    with index_update() as connection:
        active = connection.execute("SELECT fingerprint, profile FROM evidence.index_state").fetchone()
        if active is None or (active["fingerprint"], active["profile"]) != (fingerprint, profile):
            retain_previous(connection)
        connection.execute("DELETE FROM evidence.documents")
        with connection.cursor().copy("""
            COPY evidence.documents (id, title, body, source_url) FROM STDIN
        """) as copy:
            for document in documents:
                copy.write_row(document)
        with connection.cursor().copy("""
            COPY evidence.passages (document_id, ordinal, passage, embedding) FROM STDIN
        """) as copy:
            for doc_id, ordinal, text, vector in staged:
                copy.write_row((doc_id, ordinal, text, np.frombuffer(vector, dtype="<f4")))
        activate(connection, fingerprint, profile, expected_documents=document_count,
                 expected_passages=passage_count)


def ingest(source, cache_path, batch_size, *, profile="plain"):
    from .models import Models

    models = Models(load_reranker=False)
    with closing(open_cache(cache_path)) as connection:
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("""
            CREATE TEMP TABLE staged (document_id TEXT, ordinal INTEGER, passage TEXT, key BLOB)
        """)
        connection.execute("""
            CREATE TEMP TABLE staged_documents (
                id TEXT PRIMARY KEY, title TEXT, body TEXT, url TEXT
            )
        """)
        passages, generated, fingerprint = prepare(source, EmbeddingCache(connection), models,
                                                    batch_size, profile=profile)
        document_count = connection.execute("SELECT count(*) FROM staged_documents").fetchone()[0]
        documents = connection.execute("SELECT id, title, body, url FROM staged_documents")
        staged = connection.execute("""
            SELECT s.document_id, s.ordinal, s.passage, e.vector
            FROM staged s JOIN embeddings e ON e.key = s.key
        """)
        write_index(documents, staged, fingerprint, profile, document_count, passages)
    print(f"Indexed {passages:,} passages. Computed {generated:,} new embeddings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed passages and replace the Wikipedia index.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--profile", choices=INDEX_PROFILES, default="plain")
    parser.add_argument("--rollback", action="store_true", help="Restore the previous verified snapshot.")
    parser.add_argument("--batch-size", type=int, default=64, choices=range(1, 257),
                        metavar="1-256")
    args = parser.parse_args()
    if args.rollback:
        rollback()
    else:
        ingest(args.data, args.cache, args.batch_size, profile=args.profile)
