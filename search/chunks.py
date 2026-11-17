def chunk_article(title, body, tokenizer, max_length, overlap=32):
    """Split using the embedder's token budget, keeping the original text."""
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    if len(title_ids) > 48:
        title = tokenizer.decode(title_ids[:48], skip_special_tokens=True)
    prefix = f"{title}\n" if title else ""
    prefix_size = len(tokenizer.encode(prefix, add_special_tokens=False))
    budget = max_length - tokenizer.num_special_tokens_to_add() - prefix_size
    if budget <= overlap:
        raise ValueError("Token budget must exceed overlap.")
    offsets = tokenizer(
        body, add_special_tokens=False, truncation=False, return_offsets_mapping=True,
        verbose=False,
    )["offset_mapping"]
    if not offsets and title:
        yield title, title
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        text = body[offsets[start][0]:offsets[end - 1][1]].strip()
        while text and len(tokenizer.encode(prefix + text, truncation=False)) > max_length:
            end -= 1
            if end - start <= overlap:
                raise ValueError("Cannot fit passage into the tokenizer budget.")
            text = body[offsets[start][0]:offsets[end - 1][1]].strip()
        if text:
            yield text, prefix + text
        if end == len(offsets):
            break
        start = end - overlap
