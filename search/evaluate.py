import argparse
import json
import platform
from pathlib import Path
from statistics import mean, median, quantiles
from time import perf_counter

from .dataset import CONFIG, DATASET, DATASET_REVISION, DEFAULT_DATA, questions, records, title_key
from .metrics import retrieval_metrics

REPORT_PATH = Path("benchmark/results.json")


def latency_summary(milliseconds):
    return {
        "p50_ms": round(median(milliseconds), 4),
        "p95_ms": round(quantiles(milliseconds, n=100, method="inclusive")[94], 4)
        if len(milliseconds) > 1 else round(milliseconds[0], 4),
    }


def render_markdown(report):
    rows = [
        "# Wikipedia retrieval benchmark",
        "",
        f"{report['queries']} labeled questions over {report['documents']:,} documents. "
        "Latency measures the retrieval service in-process, including the database round trip on cache misses.",
        "",
        "| Mode | Precision@10 | Recall@10 | MRR@10 | nDCG@10 | Cache-miss p50 | Cache-miss p95 | Cache-hit p50 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"keyword": "Keyword", "semantic": "Semantic", "reranked": "Reranked"}
    for mode, stats in report["modes"].items():
        metrics = [f"{stats[key + '_at_10']:.4f}" for key in ("precision", "recall", "mrr", "ndcg")]
        timings = [f"{stats[state][percentile]:.4f} ms" for state, percentile in
                   (("uncached", "p50_ms"), ("uncached", "p95_ms"), ("cached", "p50_ms"))]
        rows.append("| " + " | ".join([labels[mode], *metrics, *timings]) + " |")
    if report["unmatched_questions"]:
        rows.extend(("", "Questions with no labeled article in the snapshot:"))
        rows.extend(f"- {query}" for query in report["unmatched_questions"])
    return "\n".join(rows) + "\n"


def evaluate(data, output):
    from .engine import Engine, MODES
    from .models import Models
    from hashlib import file_digest

    from .settings import CANDIDATE_DEPTH, EMBEDDER, EMBEDDER_REVISION, RERANKER, RERANKER_REVISION
    from .store import Store, open_pool

    with data.open("rb") as stream:
        fingerprint = file_digest(stream, "sha256").hexdigest()
    corpus = {title_key(row["title"]): row["id"] for row in records(data)}
    qrels = {}
    unmatched = []
    for case in questions():
        labels = {corpus[key]: 1 for title in case["articles"]
                  if (key := title_key(title)) in corpus}
        if labels:
            qrels[case["query"]] = labels
        else:
            unmatched.append(case["query"])
    if len(qrels) < 30:
        raise ValueError("Fewer than 30 questions have a matching article in the selected corpus.")

    pool = open_pool()
    samples = {mode: {"quality": [], "uncached": [], "cached": []} for mode in MODES}
    try:
        pool.open(wait=True, timeout=30)
        store = Store(pool)
        counts = store.counts()
        if store.fingerprint() != fingerprint:
            raise ValueError("Index does not match the selected benchmark corpus. Run ingestion.")
        engine = Engine(store, Models())
        for mode in MODES:
            # Warm initialization with a query outside the judged test set.
            engine.rank("what gives a tiger its pattern", mode)

        for index, (query, labels) in enumerate(qrels.items(), 1):
            for offset in range(len(MODES)):
                mode = MODES[(index + offset) % len(MODES)]
                engine.cache.clear()
                start = perf_counter()
                hits, _ = engine.search(query, mode)
                uncached = (perf_counter() - start) * 1000
                start = perf_counter()
                _, cached = engine.search(query, mode)
                warm = (perf_counter() - start) * 1000
                if not cached:
                    raise RuntimeError("Repeated query was not cached.")
                stats = samples[mode]
                stats["uncached"].append(uncached)
                stats["cached"].append(warm)
                stats["quality"].append(retrieval_metrics(
                    (hit.id for hit in hits), labels,
                ))
            if index % 25 == 0:
                print(f"Evaluated {index}/{len(qrels)} queries.", flush=True)
        if store.fingerprint() != fingerprint:
            raise RuntimeError("The index changed during evaluation. Run the benchmark again.")
    finally:
        pool.close()

    report = {
        "dataset": DATASET, "snapshot": CONFIG, "dataset_revision": DATASET_REVISION,
        "queries": len(qrels), **counts,
        "unmatched_questions": unmatched, "corpus_sha256": fingerprint,
        "candidate_depth": CANDIDATE_DEPTH, "embedding_model": f"{EMBEDDER}@{EMBEDDER_REVISION}",
        "reranker": f"{RERANKER}@{RERANKER_REVISION}", "platform": platform.platform(), "modes": {},
    }
    for mode, stats in samples.items():
        quality = stats["quality"]
        report["modes"][mode] = {
            **{key: round(mean(row[key] for row in quality), 6) for key in quality[0]},
            "uncached": latency_summary(stats["uncached"]),
            "cached": latency_summary(stats["cached"]),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all retrieval modes on labeled Wikipedia questions.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    evaluate(args.data, args.output)
