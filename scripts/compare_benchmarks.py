import argparse
import json
from math import isfinite
from pathlib import Path


def compare(base, candidate, check_latency):
    keys = ("corpus_sha256", "queries", "documents", "candidate_depth", "embedding_model",
            "reranker", "unmatched_questions")
    for key in keys:
        if key not in base or key not in candidate or base[key] != candidate[key]:
            raise ValueError(f"Benchmark inputs differ for {key}.")
    before_profile = base.get("index_profile", "plain")
    after_profile = candidate.get("index_profile", "plain")
    if check_latency and before_profile != after_profile:
        raise ValueError("A latency gate requires the same index profile in both reports.")
    if set(base["modes"]) != set(candidate["modes"]) or not base["modes"]:
        raise ValueError("Benchmark reports must have the same nonempty set of modes.")

    rows = [f"### WikiLens benchmark comparison ({before_profile} → {after_profile})", "",
            "| Mode | Baseline Recall@10 | Candidate Recall@10 | Change | Baseline uncached p95 | Candidate uncached p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    regressions = []
    for mode, before in base["modes"].items():
        after = candidate["modes"][mode]
        old_p95, new_p95 = before["uncached"]["p95_ms"], after["uncached"]["p95_ms"]
        old_recall, new_recall = before["recall_at_10"], after["recall_at_10"]
        if not all(isfinite(value) and value >= 0 for value in (old_p95, new_p95, old_recall, new_recall)):
            raise ValueError(f"Invalid measurements for {mode}.")
        rows.append(f"| {mode} | {old_recall:.1%} | {new_recall:.1%} | "
                    f"{(new_recall - old_recall) * 100:+.1f} pp | {old_p95:.2f} ms | {new_p95:.2f} ms |")
        if check_latency and new_p95 > old_p95 + max(15, old_p95 * 0.25):
            regressions.append(f"{mode}: {old_p95:.2f} → {new_p95:.2f} ms")
    return "\n".join(rows) + "\n", regressions


def main():
    parser = argparse.ArgumentParser(description="Compare WikiLens quality and latency reports.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--check-latency", action="store_true")
    parser.add_argument("--summary", type=Path, help="Append a Markdown report to this file.")
    args = parser.parse_args()
    try:
        table, regressions = compare(json.loads(args.baseline.read_text(encoding="utf-8")),
                                     json.loads(args.candidate.read_text(encoding="utf-8")), args.check_latency)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Benchmark comparison failed: {exc}\n")
    result = table
    if regressions:
        result += "\nUncached p95 exceeded the allowed increase of max(25%, 15 ms).\n\n"
        result += "\n".join(f"- {regression}" for regression in regressions) + "\n"
    print(result)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(result + "\n")
    if regressions:
        parser.exit(1)


if __name__ == "__main__":
    main()
