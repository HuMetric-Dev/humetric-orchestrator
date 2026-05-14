# humetric-orchestrator

The LLM-facing layer of Humetric. Three jobs:

1. **Parse** free-text queries into structured `ParsedQuery` for retrieval/filters.
2. **Persist** a per-recruiter query history (JSONL) and surface a centroid embedding for personalization.
3. **Write** an explained feed: 1–2 sentences per candidate explaining the match.

Pluggable LLM backend via a small `LLMBackend` protocol. Two impls ship: `AnthropicBackend` (Claude API) and `OpenAIBackend` (works with vLLM's OpenAI-compatible endpoint, or the official OpenAI API).

```python
from humetric_orchestrator import AnthropicBackend, parse_query, write_feed

backend = AnthropicBackend(api_key=...)
parsed = parse_query(backend, "rust engineer who's shipped kafka").unwrap()
feed = write_feed(backend, parsed, candidates_with_text).unwrap()
```
