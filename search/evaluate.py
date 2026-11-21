import argparse
import json
import platform
from hashlib import file_digest
from pathlib import Path
from statistics import mean, median, quantiles
from time import perf_counter

from .dataset import CONFIG, DATASET, DATASET_REVISION, DEFAULT_DATA, QUESTIONS, questions, records, title_key
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
        f"{report['queries']} labeled questions over {report['documents']:,} documents "
        f"using the {report.get('index_profile', 'plain')} index. "
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
    if "coverage_by_kind" in report:
        rows.extend(("", "| Query type | Scored / total | Keyword Recall@10 | Semantic Recall@10 | Reranked Recall@10 |",
                     "| --- | ---: | ---: | ---: | ---: |"))
        for kind, coverage in report["coverage_by_kind"].items():
            recalls = (report["modes"][mode]["recall_by_kind"].get(kind) for mode in labels)
            scores = " | ".join(f"{value:.2%}" if value is not None else "—" for value in recalls)
            rows.append(f"| {kind.replace('_', ' ')} | {coverage['scored']} / {coverage['total']} | {scores} |")
        target_met = report["modes"]["reranked"]["recall_at_10"] > 0.95
        rows.extend(("", f"Reranked Recall@10 above 95%: {'yes' if target_met else 'no'}."))
    if report["unmatched_questions"]:
        rows.extend(("", "Questions with no labeled article in the snapshot:"))
        rows.extend(f"- {query}" for query in report["unmatched_questions"])
    misses = report["modes"]["reranked"].get("misses", [])
    if misses:
        rows.extend(("", "## Reranked queries with incomplete recall", ""))
        for miss in misses:
            expected = ", ".join(miss["expected_articles"])
            returned = ", ".join(miss["top_10"][:3]) or "none"
            rows.append(f"- **{miss['query']}** — labeled: {expected}. Top three: {returned}.")
    return "\n".join(rows) + "\n"


def match_questions(data, cases):
    corpus = {title_key(row["title"]): row["id"] for row in records(data)}
    qrels = {}
    unmatched = []
    for case in cases:
        labels = {corpus[key]: 1 for title in case["articles"]
                  if (key := title_key(title)) in corpus}
        if labels:
            qrels[case["query"]] = labels
        else:
            unmatched.append(case["query"])
    return qrels, unmatched


def evaluate(data, output, *, expected_profile=None, question_path=QUESTIONS):
    from .chunks import INDEX_PROFILES
    from .engine import Engine, MODES
    from .models import Models
    from .settings import CANDIDATE_DEPTH, EMBEDDER, EMBEDDER_REVISION, RERANKER, RERANKER_REVISION
    from .store import Store, open_pool

    with data.open("rb") as stream:
        fingerprint = file_digest(stream, "sha256").hexdigest()
    cases = questions(question_path)
    qrels, unmatched = match_questions(data, cases)
    if len(qrels) < 30:
        raise ValueError("Fewer than 30 questions have a matching article in the selected corpus.")

    pool = open_pool()
    by_query = {case["query"]: case for case in cases}
    holdout = question_path.resolve() != QUESTIONS.resolve()
    samples = {mode: {"quality": [], "uncached": [], "cached": [], "by_kind": {}, "misses": []} for mode in MODES}
    excerpts = []
    try:
        pool.open(wait=True, timeout=30)
        store = Store(pool)
        state = store.index_state()
        if state is None or state["fingerprint"] != fingerprint:
            raise ValueError("Index does not match the selected benchmark corpus. Run ingestion.")
        profile = state["profile"]
        if profile not in INDEX_PROFILES or (expected_profile and profile != expected_profile):
            raise ValueError(f"Unexpected index profile {profile!r}. Reindex with the requested profile.")
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
                quality = retrieval_metrics((hit.id for hit in hits), labels)
                stats["quality"].append(quality)
                if holdout:
                    case = by_query[query]
                    kind = case.get("kind", "unclassified")
                    stats["by_kind"].setdefault(kind, []).append(quality["recall_at_10"])
                    if quality["recall_at_10"] < 1:
                        stats["misses"].append({"query": query, "expected_articles": case["articles"],
                                                "top_10": [hit.title for hit in hits[:10]],
                                                "recall_at_10": quality["recall_at_10"]})
                    if mode == "reranked" and "passage_judgment" in case:
                        excerpts.append({"query": query, "judgment": case["passage_judgment"],
                                         "top_article": hits[0].title if hits else None,
                                         "passage": hits[0].text if hits else None})
            if index % 25 == 0:
                print(f"Evaluated {index}/{len(qrels)} queries.", flush=True)
        if store.index_state() != state:
            raise RuntimeError("The index changed during evaluation. Run the benchmark again.")
    finally:
        pool.close()

    report = {
        "dataset": DATASET, "snapshot": CONFIG, "dataset_revision": DATASET_REVISION,
        "queries": len(qrels), "documents": state["documents"], "passages": state["passages"],
        "index_profile": profile,
        "unmatched_questions": unmatched, "corpus_sha256": fingerprint,
        "candidate_depth": CANDIDATE_DEPTH, "embedding_model": f"{EMBEDDER}@{EMBEDDER_REVISION}",
        "reranker": f"{RERANKER}@{RERANKER_REVISION}", "platform": platform.platform(), "modes": {},
    }
    if holdout:
        with question_path.open("rb") as stream:
            report["questions_sha256"] = file_digest(stream, "sha256").hexdigest()
        coverage = {}
        for case in cases:
            kind = case.get("kind", "unclassified")
            counts = coverage.setdefault(kind, {"total": 0, "scored": 0})
            counts["total"] += 1
            counts["scored"] += case["query"] in qrels
        report["coverage_by_kind"] = coverage
        report["passages_for_excerpt_review"] = excerpts
    for mode, stats in samples.items():
        quality = stats["quality"]
        report["modes"][mode] = {
            **{key: round(mean(row[key] for row in quality), 6) for key in quality[0]},
            "uncached": latency_summary(stats["uncached"]),
            "cached": latency_summary(stats["cached"]),
        }
        if holdout:
            report["modes"][mode].update(
                recall_by_kind={kind: round(mean(values), 6) for kind, values in stats["by_kind"].items()},
                misses=stats["misses"],
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    from .chunks import INDEX_PROFILES

    parser = argparse.ArgumentParser(description="Evaluate all retrieval modes on labeled Wikipedia questions.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=INDEX_PROFILES, help="Require an active index with this profile.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--coverage-only", action="store_true", help="Check answer articles without running search.")
    args = parser.parse_args()
    if args.coverage_only:
        matched, unmatched = match_questions(args.data, questions(args.questions))
        print(f"{len(matched)}/{len(matched) + len(unmatched)} questions have a labeled article in the corpus.")
        for query in unmatched:
            print(f"- {query}")
    else:
        if args.questions.resolve() != QUESTIONS.resolve() and (
            args.output is None or args.output.resolve() == REPORT_PATH.resolve()
        ):
            parser.error("Use a separate --output path with an alternate question set.")
        evaluate(args.data, args.output or REPORT_PATH, expected_profile=args.profile, question_path=args.questions)
