CREATE SCHEMA IF NOT EXISTS evidence;
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- Support managed databases where vector already lives in another schema.
DO $$
BEGIN
    PERFORM set_config('search_path', 'evidence,public,extensions,' ||
        quote_ident((SELECT n.nspname FROM pg_extension e
                     JOIN pg_namespace n ON n.oid = e.extnamespace
                     WHERE e.extname = 'vector')), true);
    CREATE TABLE IF NOT EXISTS evidence.documents (
        id text PRIMARY KEY,
        title text NOT NULL,
        body text NOT NULL,
        source_url text NOT NULL,
        search_terms tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('english', title), 'A') ||
            setweight(to_tsvector('english', body), 'B')
        ) STORED
    );
    CREATE TABLE IF NOT EXISTS evidence.passages (
        document_id text NOT NULL REFERENCES evidence.documents(id) ON DELETE CASCADE,
        ordinal integer NOT NULL,
        passage text NOT NULL,
        embedding vector(384) NOT NULL,
        passage_terms tsvector GENERATED ALWAYS AS (to_tsvector('english', passage)) STORED,
        PRIMARY KEY(document_id, ordinal)
    );
    CREATE TABLE IF NOT EXISTS evidence.index_state (
        singleton boolean PRIMARY KEY CHECK (singleton),
        fingerprint text NOT NULL
    );
    ALTER TABLE evidence.index_state ADD COLUMN IF NOT EXISTS generation bigint NOT NULL DEFAULT 0;
    ALTER TABLE evidence.index_state ADD COLUMN IF NOT EXISTS profile text NOT NULL DEFAULT 'plain';
    ALTER TABLE evidence.index_state ADD COLUMN IF NOT EXISTS document_count bigint;
    ALTER TABLE evidence.index_state ADD COLUMN IF NOT EXISTS passage_count bigint;
    UPDATE evidence.index_state SET
        document_count = (SELECT count(*) FROM evidence.documents),
        passage_count = (SELECT count(*) FROM evidence.passages)
    WHERE document_count IS NULL OR passage_count IS NULL;
    -- The backup needs source rows only. Generated columns and ANN indexes stay on the live tables.
    CREATE TABLE IF NOT EXISTS evidence.previous_documents AS
        SELECT id, title, body, source_url FROM evidence.documents WITH NO DATA;
    CREATE TABLE IF NOT EXISTS evidence.previous_passages AS
        SELECT document_id, ordinal, passage, embedding FROM evidence.passages WITH NO DATA;
    CREATE TABLE IF NOT EXISTS evidence.previous_index_state AS
        SELECT fingerprint, profile FROM evidence.index_state WITH NO DATA;
    CREATE INDEX IF NOT EXISTS passages_cosine_idx ON evidence.passages
        USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS search_terms_idx ON evidence.documents
        USING gin (search_terms);
END $$;
