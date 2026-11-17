from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .settings import database_url


@dataclass(frozen=True)
class Hit:
    id: str
    title: str
    text: str
    url: str
    score: float


DENSE_SQL = """
WITH neighbors AS MATERIALIZED (
    SELECT p.document_id, p.ordinal, p.passage, p.embedding <=> %(vector)s AS distance
    FROM evidence.passages p
    {limit}
),
best_passages AS (
    SELECT DISTINCT ON (document_id) document_id, passage, distance
    FROM neighbors
    ORDER BY document_id, distance, ordinal
)
SELECT d.id, d.title, b.passage AS text, d.source_url AS url, 1 - b.distance AS score
FROM best_passages b
JOIN evidence.documents d ON d.id = b.document_id
ORDER BY b.distance, d.id
LIMIT %(depth)s
"""

APPROXIMATE_SQL = DENSE_SQL.format(limit="""
    ORDER BY p.embedding <=> %(vector)s
    LIMIT %(passage_depth)s
""")
EXACT_SQL = DENSE_SQL.format(limit="")


KEYWORD_SQL = """
WITH keyword_query AS (
    SELECT to_tsquery('english', COALESCE(string_agg(quote_literal(term), ' | '), '')) AS q
    FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS terms(term)
)
SELECT r.id, r.title, best.passage AS text, r.source_url AS url, r.score
FROM (
    SELECT d.id, d.title, d.source_url, k.q, ts_rank_cd(d.search_terms, k.q, 32) AS score
    FROM evidence.documents d CROSS JOIN keyword_query k
    WHERE d.search_terms @@ k.q
    ORDER BY score DESC, d.id
    LIMIT %(depth)s
) r
CROSS JOIN LATERAL (
    SELECT p.passage FROM evidence.passages p WHERE p.document_id = r.id
    ORDER BY ts_rank_cd(p.passage_terms, r.q, 32) DESC, p.ordinal LIMIT 1
) best
ORDER BY r.score DESC, r.id
"""


def configure(connection):
    schema = connection.execute(
        "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n "
        "ON n.oid = e.extnamespace WHERE e.extname = 'vector'"
    ).fetchone()
    if schema is None:
        raise RuntimeError("Install the vector extension and apply db/schema.sql first.")
    connection.execute(sql.SQL("SET search_path TO evidence, pg_catalog, {}").format(
        sql.Identifier(schema["nspname"])
    ))
    connection.execute("SET statement_timeout = 12000")
    register_vector(connection)


def open_pool():
    return ConnectionPool(
        database_url(), min_size=1, max_size=4, timeout=12, open=False,
        configure=configure, check=ConnectionPool.check_connection,
        kwargs={"row_factory": dict_row, "autocommit": True,
                "prepare_threshold": None, "connect_timeout": 10},
    )


class Store:
    def __init__(self, pool):
        self.pool = pool

    def counts(self):
        with self.pool.connection() as connection:
            return connection.execute("""
                SELECT (SELECT count(*) FROM evidence.documents) AS documents,
                       (SELECT count(*) FROM evidence.passages) AS passages
            """).fetchone()

    def fingerprint(self):
        with self.pool.connection() as connection:
            row = connection.execute("SELECT fingerprint FROM evidence.index_state").fetchone()
        return row["fingerprint"] if row else None

    def keyword(self, query, depth):
        with self.pool.connection() as connection:
            rows = connection.execute(KEYWORD_SQL, {"query": query, "depth": depth}).fetchall()
        return [Hit(**row) for row in rows]

    def dense(self, vector, depth):
        with self.pool.connection() as connection, connection.transaction():
            connection.execute("SET LOCAL hnsw.ef_search = 240")
            params = {"vector": vector, "depth": depth, "passage_depth": depth * 4}
            rows = connection.execute(APPROXIMATE_SQL, params).fetchall()
            if len(rows) < depth:
                # A cluster of near-duplicate passages can crowd out distinct articles.
                rows = connection.execute(EXACT_SQL, {"vector": vector, "depth": depth}).fetchall()
        return [Hit(**row) for row in rows]
