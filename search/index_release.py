from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from .settings import database_url
from .store import configure


@contextmanager
def index_update():
    with psycopg.connect(database_url(), row_factory=dict_row, autocommit=True) as connection:
        configure(connection)
        with connection.transaction():
            connection.execute("SET LOCAL statement_timeout = 0")
            connection.execute("SELECT pg_advisory_xact_lock(7115063201)")
            yield connection


def retain_previous(connection):
    connection.execute("DELETE FROM evidence.previous_index_state")
    connection.execute("DELETE FROM evidence.previous_passages")
    connection.execute("DELETE FROM evidence.previous_documents")
    connection.execute("""
        INSERT INTO evidence.previous_documents (id, title, body, source_url)
        SELECT id, title, body, source_url FROM evidence.documents
        WHERE EXISTS (SELECT FROM evidence.index_state)
    """)
    connection.execute("""
        INSERT INTO evidence.previous_passages (document_id, ordinal, passage, embedding)
        SELECT document_id, ordinal, passage, embedding FROM evidence.passages
        WHERE EXISTS (SELECT FROM evidence.index_state)
    """)
    connection.execute("""
        INSERT INTO evidence.previous_index_state (fingerprint, profile)
        SELECT fingerprint, profile FROM evidence.index_state
    """)


def activate(connection, fingerprint, profile, *, expected_documents=None, expected_passages=None):
    row = connection.execute("""
        SELECT (SELECT count(*) FROM evidence.documents) AS documents,
               (SELECT count(*) FROM evidence.passages) AS passages,
               EXISTS (SELECT 1 FROM evidence.documents d WHERE NOT EXISTS (
                   SELECT 1 FROM evidence.passages p WHERE p.document_id = d.id
               )) AS missing_passages,
               EXISTS (SELECT 1 FROM evidence.passages WHERE vector_norm(embedding) = 0) AS zero_vector
    """).fetchone()
    documents, passages = row["documents"], row["passages"]
    if not documents or not passages or row["missing_passages"] or row["zero_vector"]:
        raise ValueError("Index validation failed. Check for empty documents, missing passages, or zero vectors.")
    if expected_documents is not None and (documents, passages) != (expected_documents, expected_passages):
        raise ValueError("Index validation failed. Loaded row counts differ from the staged corpus.")
    connection.execute("""
        INSERT INTO evidence.index_state (singleton, fingerprint, profile, generation, document_count, passage_count)
        VALUES (true, %s, %s, 1, %s, %s)
        ON CONFLICT (singleton) DO UPDATE SET fingerprint = EXCLUDED.fingerprint,
            profile = EXCLUDED.profile, generation = index_state.generation + 1,
            document_count = EXCLUDED.document_count, passage_count = EXCLUDED.passage_count
    """, (fingerprint, profile, documents, passages))
    connection.execute("ANALYZE evidence.documents")
    connection.execute("ANALYZE evidence.passages")


def rollback():
    with index_update() as connection:
        previous = connection.execute("SELECT fingerprint, profile FROM evidence.previous_index_state").fetchone()
        if previous is None:
            raise ValueError("No previous verified snapshot is available.")
        connection.execute("DELETE FROM evidence.documents")
        connection.execute("""
            INSERT INTO evidence.documents (id, title, body, source_url)
            SELECT id, title, body, source_url FROM evidence.previous_documents
        """)
        connection.execute("""
            INSERT INTO evidence.passages (document_id, ordinal, passage, embedding)
            SELECT document_id, ordinal, passage, embedding FROM evidence.previous_passages
        """)
        activate(connection, previous["fingerprint"], previous["profile"])
    print(f"Restored the previous verified index ({previous['profile']}).")
