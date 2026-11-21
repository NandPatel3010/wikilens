import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import quantiles
from threading import Barrier
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener


def burst(queries, port):
    start = Barrier(len(queries))
    token = os.environ.get("SEARCH_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def search(query):
        params = urlencode({"q": query, "mode": "reranked", "page_size": 10})
        request = Request(f"http://127.0.0.1:{port}/search?{params}", headers=headers)
        start.wait(timeout=15)
        began = perf_counter()
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=25) as response:
                payload = json.load(response)
            mode = payload.get("effective_mode", payload["mode"])
            return mode, (perf_counter() - began) * 1000, payload["cache_hit"]
        except HTTPError as error:
            return f"HTTP {error.code}", (perf_counter() - began) * 1000, False
        except (URLError, TimeoutError):
            return "connection failure or timeout", (perf_counter() - began) * 1000, False

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        return list(pool.map(search, queries))


def main():
    parser = argparse.ArgumentParser(description="Run one cold-cache burst against local WikiLens.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--requests", type=int, choices=range(2, 33), default=16, metavar="2-32")
    args = parser.parse_args()
    dataset = Path(__file__).resolve().parents[1] / "benchmark" / "questions.json"
    data = json.loads(dataset.read_text(encoding="utf-8"))
    queries = list(dict.fromkeys(" ".join(row["query"].split()) for row in data))
    if len(queries) < args.requests:
        parser.error("The benchmark dataset has fewer distinct queries than requested.")
    responses = burst(queries[:args.requests], args.port)
    successful = [ms for mode, ms, _ in responses if mode in ("semantic", "reranked")]
    p95 = quantiles(successful, n=100, method="inclusive")[94] if len(successful) > 1 else max(successful, default=None)
    print(json.dumps({
        "requests": len(responses),
        "outcomes": dict(Counter(mode for mode, _, _ in responses)),
        "cache_hits": sum(cached for _, _, cached in responses),
        "successful_p95_ms": round(p95, 2) if p95 is not None else None,
    }, indent=2))
    if not successful:
        parser.exit(1, "The benchmark received no successful searches.\n")


if __name__ == "__main__":
    main()
