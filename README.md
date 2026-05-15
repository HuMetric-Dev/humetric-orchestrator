# humetric-orchestrator

The LLM-facing layer of Humetric. Three jobs:

1. **Parse** free-text queries into structured `ParsedQuery` for retrieval/filters.
2. **Persist** a per-recruiter query history (JSONL) and surface a centroid embedding for personalization.
3. **Write** an explained feed: 1–2 sentences per candidate explaining the match.

Pluggable LLM backend via a small `LLMBackend` protocol. Two impls ship:

- **`OpenAIBackend`** — default. Targets any OpenAI-compatible endpoint: local Ollama (`http://localhost:11434/v1`), vLLM, or the OpenAI API itself.
- **`AnthropicBackend`** — Claude API. Quality-ceiling fallback. The `anthropic` SDK is now an optional install: `uv sync --extra anthropic`.

```python
from humetric_orchestrator import OpenAIBackend, parse_query, write_feed

backend = OpenAIBackend(
    base_url="http://localhost:11434/v1",
    api_key="dummy",
    model="qwen3.5:9b-q5km",
)
parsed = parse_query(backend, "rust engineer who's shipped kafka").unwrap()
feed = write_feed(backend, parsed, candidates_with_text).unwrap()
```
