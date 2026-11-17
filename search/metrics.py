from math import log2


def retrieval_metrics(ranked, judgments, k=10):
    if k < 1:
        raise ValueError("k must be positive.")
    document_ids = list(dict.fromkeys(ranked))[:k]
    grades = [judgments.get(doc_id, 0) for doc_id in document_ids]
    relevant_in_top_k = sum(grade > 0 for grade in grades)
    relevant_total = sum(grade > 0 for grade in judgments.values())
    ideal = sorted(judgments.values(), reverse=True)[:k]

    def discounted_gain(values):
        return sum((2 ** grade - 1) / log2(rank + 2) for rank, grade in enumerate(values))

    ideal_gain = discounted_gain(ideal)
    return {
        f"precision_at_{k}": relevant_in_top_k / k,
        f"recall_at_{k}": relevant_in_top_k / relevant_total if relevant_total else 0.0,
        f"mrr_at_{k}": next((1 / rank for rank, grade in enumerate(grades, 1) if grade > 0), 0.0),
        f"ndcg_at_{k}": discounted_gain(grades) / ideal_gain if ideal_gain else 0.0,
    }
