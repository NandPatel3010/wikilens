import argparse
import hashlib
import json
import re
import unicodedata
from heapq import heappush, heapreplace
from pathlib import Path
from urllib.parse import urlsplit

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.simple"
DATASET_REVISION = "3e1f92c331f318af862b87e2319ed5dc26d80f5d"
DEFAULT_DATA = Path("data/wikipedia.jsonl")
QUESTIONS = Path("benchmark/questions.json")
SAMPLE_SEED = "wiki-search-v1"


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def title_key(title):
    return " ".join(unicodedata.normalize("NFKC", title).replace("_", " ").casefold().split())


def questions(path=QUESTIONS):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_articles(source, count, target_titles):
    sample = []
    anchors = {}
    for row in source:
        article_id = row["id"]
        title = row["title"].strip()
        article = row["text"].strip()
        url = row["url"]
        if not (title and article and urlsplit(url).netloc == "simple.wikipedia.org"
                and url.startswith("https://") and not re.match(r"(?i)#redirect\b", article)):
            continue
        selected = {"id": article_id, "title": title, "text": article, "url": url}
        key = title_key(title)
        if key in target_titles:
            anchors[article_id] = selected
            continue
        priority = int.from_bytes(hashlib.blake2b(
            f"{SAMPLE_SEED}:{article_id}".encode(), digest_size=8,
        ).digest())
        # Keep the rows with the lowest deterministic hash regardless of source order.
        entry = (-priority, article_id, selected)
        if len(sample) < count:
            heappush(sample, entry)
        elif entry[:2] > sample[0][:2]:
            heapreplace(sample, entry)
    if len(anchors) > count:
        raise ValueError("Article limit is smaller than the labeled article set.")
    remaining = count - len(anchors)
    if len(sample) < remaining:
        raise ValueError(f"Dataset has fewer than {count} usable articles.")
    chosen = {**anchors, **{row_id: row for _, row_id, row in
                             sorted(sample, reverse=True)[:remaining]}}
    return [chosen[row_id] for row_id in sorted(chosen)]


def download(path, count, *, extra_questions=None):
    from datasets import load_dataset

    cases = questions()
    if extra_questions is not None:
        cases.extend(questions(extra_questions))
    titles = {title_key(title) for case in cases for title in case["articles"]}
    source = load_dataset(DATASET, CONFIG, revision=DATASET_REVISION, split="train", streaming=True)
    articles = sample_articles(source, count, titles)
    path.parent.mkdir(parents=True, exist_ok=True)
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                            delete=False) as stream:
        temp = Path(stream.name)
        try:
            for row in articles:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Selected {len(articles):,} Simple English Wikipedia articles at {path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample the Simple English Wikipedia snapshot.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--articles", type=int, default=5000)
    parser.add_argument("--extra-questions", type=Path, help="Include these potential answer articles in the sample.")
    args = parser.parse_args()
    if not 100 <= args.articles <= 50000:
        parser.error("--articles must be between 100 and 50000.")
    if args.extra_questions and (args.output is None or args.output.resolve() == DEFAULT_DATA.resolve()):
        parser.error("Use a separate --output path when adding benchmark questions to the sample.")
    download(args.output or DEFAULT_DATA, args.articles, extra_questions=args.extra_questions)
