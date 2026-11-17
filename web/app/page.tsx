"use client";

import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

const modes = ["keyword", "semantic", "reranked"] as const;
type Mode = (typeof modes)[number];
const labels: Record<Mode, string> = { keyword: "Keyword", semantic: "Semantic", reranked: "Reranked" };
const descriptions: Record<Mode, string> = {
  keyword: "Matches words using PostgreSQL full-text search.",
  semantic: "Matches meaning using cosine similarity.",
  reranked: "Reorders semantic matches with a cross-encoder.",
};
const scoreLabels: Record<Mode, string> = { keyword: "Text rank", semantic: "Cosine", reranked: "Rerank score" };
const examples = [
  "Why is the sky blue?",
  "Why does a compass point north?",
  "How do plants make sugar from sunlight?",
];

type Hit = { id: string; title: string; text: string; url: string; score: number };
type ResultPage = {
  query: string; mode: Mode; total: number; page: number; page_size: number;
  has_next: boolean; latency_ms: number; cache_hit: boolean; results: Hit[];
};
type Timing = { p50_ms: number; p95_ms: number };
type Stats = {
  precision_at_10: number; recall_at_10: number; mrr_at_10: number; ndcg_at_10: number;
  uncached: Timing; cached: Timing;
};
type Benchmark = { available: false } | {
  available: true; queries: number; documents: number; passages: number;
  modes: Record<Mode, Stats>; unmatched_questions: string[];
};

async function api<T>(resource: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch("/api/" + resource, { signal });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : "Check the search parameters and try again.");
  }
  return data as T;
}

function Loading({ label }: { label: string }) {
  return <div className="loading" role="status">
    <p>{label}</p>
    <div aria-hidden="true">
      {[0, 1, 2].map((item) => <div className="skeleton-row" key={item}><span /><span /><span /></div>)}
    </div>
  </div>;
}

function Snippet({ text, query }: { text: string; query: string }) {
  const words = Array.from(query.toLowerCase().matchAll(/[\p{L}\p{N}]{3,}/gu), (part) => part[0]);
  const stop = new Set(["why", "how", "the", "what", "when", "does", "and", "with", "its", "for", "from", "can", "are"]);
  const terms = [...new Set(words.filter((word) => !stop.has(word)))].slice(0, 30);
  if (!terms.length) return text;
  const delimited = new RegExp("(?<![\\p{L}\\p{N}])(" + terms.join("|") + ")(?![\\p{L}\\p{N}])", "giu");
  return text.split(delimited).map((part, index) => index % 2 ? <mark key={index}>{part}</mark> : part);
}

function BenchmarkPanel() {
  const [data, setData] = useState<Benchmark>();
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    api<Benchmark>("benchmark", controller.signal).then(setData).catch((cause: Error) => {
      if (!controller.signal.aborted) setError(cause.message);
    });
    return () => controller.abort();
  }, []);

  return <section className="benchmark" aria-labelledby="benchmark-title">
    <h1 id="benchmark-title">Retrieval benchmark</h1>
    {error ? <p role="alert" className="error">{error}</p> : !data ? <p role="status">Loading benchmark…</p> : !data.available ?
      <p>The published report is available in the <a href="https://github.com/NandPatel3010/wikilens/actions/runs/44040068459">GitHub benchmark run</a>.</p> :
      <>
        <p>{data.queries} labeled questions over {data.documents.toLocaleString()} articles and {data.passages.toLocaleString()} passages.</p>
        <div className="precision">
          <h2>Precision@10</h2>
          {modes.map((mode) => {
            const value = data.modes[mode].precision_at_10 * 100;
            return <div key={mode} className="bar-row">
              <span>{labels[mode]}</span>
              <div className="bar-track" aria-hidden="true"><div className="bar" style={{ width: value + "%" }} /></div>
              <span>{value.toFixed(1)}%</span>
            </div>;
          })}
        </div>
        <div className="table-wrap">
          <table>
            <caption>Retrieval quality and latency</caption>
            <thead><tr><th scope="col">Mode</th><th scope="col">Precision@10</th><th scope="col">Recall@10</th><th scope="col">MRR@10</th>
              <th scope="col">nDCG@10</th><th scope="col">Uncached p50</th><th scope="col">Uncached p95</th><th scope="col">Cached p50</th></tr></thead>
            <tbody>{modes.map((mode) => {
              const s = data.modes[mode];
              return <tr key={mode}><th scope="row">{labels[mode]}</th><td>{s.precision_at_10.toFixed(4)}</td><td>{s.recall_at_10.toFixed(4)}</td>
                <td>{s.mrr_at_10.toFixed(4)}</td><td>{s.ndcg_at_10.toFixed(4)}</td><td>{s.uncached.p50_ms.toFixed(2)} ms</td>
                <td>{s.uncached.p95_ms.toFixed(2)} ms</td><td>{s.cached.p50_ms.toFixed(4)} ms</td></tr>;
            })}</tbody>
          </table>
        </div>
        {data.unmatched_questions.length > 0 && <details><summary>{data.unmatched_questions.length} questions without a labeled article in this snapshot</summary>
          <ul>{data.unmatched_questions.map((question) => <li key={question}>{question}</li>)}</ul></details>}
      </>}
  </section>;
}

export default function Page() {
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("reranked");
  const [tab, setTab] = useState<"search" | "benchmark">("search");
  const [result, setResult] = useState<ResultPage>();
  const [count, setCount] = useState<number>();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);

  const search = useCallback(async (query: string, selected: Mode, page = 1) => {
    const q = query.trim();
    if (!q) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setPending(true);
    setError("");
    setResult(undefined);
    setTab("search");
    setMode(selected);
    const params = new URLSearchParams({ q, mode: selected, page: String(page) });
    window.history.replaceState(null, "", "?" + params.toString());
    try {
      const data = await api<ResultPage>("search?" + params.toString(), controller.signal);
      if (!controller.signal.aborted) setResult(data);
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : "Search failed. Try again.");
      }
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    api<{ documents: number }>("health", controller.signal).then(({ documents }) => setCount(documents)).catch(() => {});
    const params = new URLSearchParams(window.location.search);
    const saved = params.get("q");
    const savedMode = params.get("mode") as Mode;
    if (saved) {
      const savedPage = Number(params.get("page"));
      const validPage = Number.isInteger(savedPage) && savedPage >= 1 && savedPage <= 7 ? savedPage : 1;
      setInput(saved.slice(0, 500));
      void search(saved.slice(0, 500), modes.includes(savedMode) ? savedMode : "reranked", validPage);
    }
    return () => { controller.abort(); request.current?.abort(); };
  }, [search]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void search(input, mode);
  }

  return <>
    <header className="site-header">
      <div className="header-inner">
        <button type="button" className="brand" onClick={() => setTab("search")}>WikiLens</button>
        <nav aria-label="Views">
          <button type="button" aria-current={tab === "search" ? "page" : undefined} onClick={() => setTab("search")}>Search</button>
          <button type="button" aria-current={tab === "benchmark" ? "page" : undefined} onClick={() => setTab("benchmark")}>Benchmark</button>
        </nav>
      </div>
    </header>

    <main className="content">
      {tab === "benchmark" ? <BenchmarkPanel /> : <>
        <section className="search-section" aria-labelledby="title">
          <div className="collection">
            <a href="https://simple.wikipedia.org/">Simple English Wikipedia</a>
            <span>November 2023 snapshot</span>
          </div>
          <h1 id="title">Search the encyclopedia.</h1>
          <form className="search-form" role="search" onSubmit={submit}>
            <label className="sr-only" htmlFor="query">Wikipedia search query</label>
            <input id="query" placeholder="Enter a topic or question" required maxLength={500}
              value={input} onChange={(event) => setInput(event.target.value)} />
            <button type="submit" disabled={!input.trim()}>Search</button>
          </form>
          <fieldset className="mode-controls">
            <legend className="sr-only">Retrieval mode</legend>
            <div className="modes">
              {modes.map((option) => <label key={option} className="mode">
                <input className="sr-only" type="radio" name="mode" checked={mode === option} onChange={() => {
                  setMode(option);
                  if (input.trim()) void search(input, option);
                }} />
                <span>{labels[option]}</span>
              </label>)}
            </div>
            <p>{descriptions[mode]}</p>
          </fieldset>
        </section>

        <section className="results" aria-label="Search results">
          {pending && <Loading label="Searching Wikipedia." />}
          {error && <p className="error" role="alert">{error}</p>}
          {!pending && result && <>
            <div className="results-meta" role="status">
              <h2>{result.total} ranked {result.total === 1 ? "article" : "articles"}</h2>
              <span>{result.latency_ms} ms{result.cache_hit ? " (cached)" : ""}</span>
            </div>
            {result.results.length === 0 && <p className="status">No articles found. Try a shorter query or another search mode.</p>}
            {result.results.length > 0 && <ol className="result-list" role="list" start={(result.page - 1) * result.page_size + 1}>
              {result.results.map((hit) => <li key={hit.id}>
                <div className="result-head">
                  <h3><a href={hit.url} target="_blank" rel="noreferrer">{hit.title}</a></h3>
                  <span className="score">{scoreLabels[result.mode]} {hit.score.toFixed(3)}</span>
                </div>
                <p><Snippet text={hit.text} query={result.query} /></p>
              </li>)}
            </ol>}
            {(result.has_next || result.page > 1) && <div className="pagination" aria-label="Result pages">
              <button type="button" disabled={result.page === 1} onClick={() => void search(result.query, result.mode, result.page - 1)}>Previous</button>
              <span>Page {result.page} of {Math.max(1, Math.ceil(result.total / result.page_size))}</span>
              <button type="button" disabled={!result.has_next} onClick={() => void search(result.query, result.mode, result.page + 1)}>Next</button>
            </div>}
          </>}
          {!pending && !result && !error && <div className="empty-state">
            <div className="results-meta">
              <h2>Try a search</h2>
              {count !== undefined && <span>{count.toLocaleString()} indexed articles</span>}
            </div>
            <ul className="examples">
              {examples.map((example) => <li key={example}><button type="button" onClick={() => {
                setInput(example);
                void search(example, mode);
              }}>{example}</button></li>)}
            </ul>
          </div>}
        </section>
      </>}
    </main>

    <footer className="site-footer">
      <div className="footer-inner">
        <a href="https://simple.wikipedia.org/">Article source</a>
        <details>
          <summary>Terms of use</summary>
          <p>WikiLens is an independent search tool. Results contain excerpts from Simple English Wikipedia. Follow the article link for its contributors, source text, and license information.</p>
          <a href="https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use">Wikimedia Terms of Use</a>
        </details>
        <details>
          <summary>Privacy policy</summary>
          <p>Search works without an account. Queries are sent to the WikiLens API and appear in the page URL. That URL can be saved in browser history or included when you share a link. Vercel hosts the site and API and can process technical request information.</p>
          <a href="https://vercel.com/legal/privacy-notice">Vercel Privacy Notice</a>
        </details>
      </div>
    </footer>
  </>;
}
