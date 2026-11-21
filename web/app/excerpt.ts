const sentenceSegmenter = new Intl.Segmenter("en", { granularity: "sentence" });
const stopWords = new Set([
  "and", "are", "can", "does", "for", "from", "has", "have", "how", "into",
  "its", "make", "that", "the", "their", "them", "this", "was", "were",
  "what", "when", "where", "which", "who", "why", "will", "with", "you",
]);
const excerptLimit = 280;
const wordPattern = /[\p{L}\p{N}]{3,}/gu;

function wordKey(word: string) {
  const lower = word.toLowerCase();
  return lower.length > 4 && /[^isu]s$/.test(lower) ? lower.slice(0, -1) : lower;
}

function terms(text: string): Set<string> {
  const words = text.toLowerCase().match(wordPattern) ?? [];
  return new Set(words.filter((word) => !stopWords.has(word)).map(wordKey));
}

function clipSentence(sentence: string, queryTerms: Set<string>) {
  if (sentence.length <= excerptLimit) return sentence;
  const match = Array.from(sentence.matchAll(wordPattern)).find((word) => queryTerms.has(wordKey(word[0])));
  const offset = Math.max(0, (match?.index ?? 0) - 90);
  const start = offset ? sentence.lastIndexOf(" ", offset) + 1 : 0;
  const end = Math.min(sentence.length, start + excerptLimit - 2);
  const excerpt = sentence.slice(start, end);
  const boundary = excerpt.lastIndexOf(" ");
  return `${start ? "…" : ""}${end < sentence.length && boundary > 0 ? excerpt.slice(0, boundary) : excerpt}${end < sentence.length ? "…" : ""}`;
}

export function selectExcerpt(text: string, query: string) {
  const passage = text.replace(/\s+/g, " ").trim();
  const sentences = Array.from(sentenceSegmenter.segment(passage), ({ segment }) => segment.trim());
  const queryTerms = terms(query);
  const complete = sentences.filter((sentence) => /^[\p{Pe}\p{Pi}\p{Ps}"']*[\p{Lu}\p{N}]/u.test(sentence)
    && /[.!?][\p{Pe}\p{Pf}"']*$/u.test(sentence));
  const candidates = complete.length ? complete : sentences;
  let best = "";
  let bestScore = -Infinity;
  let matches = 0;
  for (const sentence of candidates) {
    const words = terms(sentence);
    const overlap = [...queryTerms].filter((term) => words.has(term)).length;
    const score = overlap + (sentence.length <= excerptLimit ? 0.25 : 0);
    if (score > bestScore) {
      best = sentence;
      bestScore = score;
      matches = overlap;
    }
  }

  const next = sentences[sentences.indexOf(best) + 1] ?? "";
  const context = best && best.length < 135 && complete.includes(next)
    && best.length + next.length + 1 <= excerptLimit ? next : "";
  const focus = clipSentence(best, queryTerms);
  return { focus, context, matched: matches > 0, hasMore: (context ? `${focus} ${context}` : focus) !== passage };
}
