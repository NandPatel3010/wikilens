INDEX_PROFILES = ("plain", "contextual")


def chunk_article(title, body, tokenizer, max_length, overlap=32, *, contextual=False):
    """Split using the embedder's token budget, keeping the original text."""
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    if len(title_ids) > 48:
        title = tokenizer.decode(title_ids[:48], skip_special_tokens=True)
    prefix = f"{title}\n" if title else ""
    prefix_size = len(tokenizer.encode(prefix, add_special_tokens=False))
    input_budget = max_length - tokenizer.num_special_tokens_to_add()
    budget = input_budget - prefix_size
    if budget <= overlap:
        raise ValueError("Token budget must exceed overlap.")
    offsets = tokenizer(
        body, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
        verbose=False,
    )["offset_mapping"]
    contextual_prefix = prefix
    contextual_budget = budget
    if contextual and offsets:
        # Use an extractive summary from the article itself. Reserve room for overlap.
        lead_tokens = min(32, budget - overlap - 8, len(offsets))
        while lead_tokens > 0:
            lead = body[:offsets[lead_tokens - 1][1]].strip()
            candidate = f"{prefix}{lead}\n"
            remaining = input_budget - len(tokenizer.encode(candidate, add_special_tokens=False))
            if remaining >= overlap + 8:
                contextual_prefix, contextual_budget = candidate, remaining
                break
            lead_tokens -= 1
    if not offsets and title:
        yield title, title
    start = 0
    while start < len(offsets):
        model_prefix, segment_budget = (contextual_prefix, contextual_budget) if start else (prefix, budget)
        end = min(start + segment_budget, len(offsets))
        text = body[offsets[start][0]:offsets[end - 1][1]].strip()
        while text and len(tokenizer.encode(model_prefix + text, truncation=False)) > max_length:
            end -= 1
            if end - start <= overlap:
                raise ValueError("Cannot fit passage into the tokenizer budget.")
            text = body[offsets[start][0]:offsets[end - 1][1]].strip()
        if text:
            yield text, model_prefix + text
        if end == len(offsets):
            break
        start = end - overlap
