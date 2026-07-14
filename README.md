# infolang-crewai

InfoLang-backed storage for CrewAI's unified `Memory` system.

> **A note on CrewAI's API.** Older CrewAI docs describe `memory_config={"provider": ...}`
> and an `ExternalMemory` class. As of `crewai>=1.15`, both are gone. CrewAI now has a single
> `Memory` class (`crewai.memory.unified_memory.Memory`) backed by a pluggable
> `StorageBackend` protocol (`crewai.memory.storage.backend.StorageBackend`). This package
> targets *that* interface — the one actually shipped in the installed package — not the
> older docs. See [Known limitations](#known-limitations) for what that means in practice.

## Install

The InfoLang Python SDK is not yet on PyPI. Until it is, install both packages from source:

```bash
pip install "infolang @ git+ssh://git@github.com/InfoLang-Inc/infolang-sdk-python.git@v0.2.0"
pip install infolang-crewai
```

For local development against this repo:

```bash
pip install -e ".[dev]"
pip install -e "../sdk-python"   # editable install of the infolang SDK, for local testing only
```

## Quickstart

```python
from crewai import Agent, Crew, Task
from crewai.memory.unified_memory import Memory
from infolang_crewai import InfoLangEmbedder, InfoLangStorage

# One InfoLangStorage per Crew is the common case: it owns one InfoLang
# namespace (memory bank) for everything the crew remembers.
memory = Memory(
    storage=InfoLangStorage(api_key="il_live_...", namespace="research-crew"),
    # InfoLangEmbedder MUST be paired with InfoLangStorage -- see "Why the
    # custom embedder?" below. Leaving this out (or using a real embedding
    # model) makes InfoLangStorage.search() raise InfoLangCrewAIConfigError.
    embedder=InfoLangEmbedder(),
)

researcher = Agent(
    role="Researcher",
    goal="Find and remember useful facts",
    backstory="...",
)

task = Task(description="Research X and remember what you learn", agent=researcher)

crew = Crew(agents=[researcher], tasks=[task], memory=memory)
crew.kickoff()
```

`Crew(memory=True)` still works and builds CrewAI's default local (LanceDB)
store; pass a `Memory(storage=InfoLangStorage(...), embedder=InfoLangEmbedder())`
instance instead to route through InfoLang.

### Per-agent namespaces

Two ways to isolate memory per agent, at different levels:

**InfoLang-level (separate memory bank per agent)** — construct one
`InfoLangStorage`/`Memory` per agent, or use `namespace_resolver`:

```python
storage = InfoLangStorage(
    api_key="il_live_...",
    namespace="research-crew",  # fallback / used by get_record() and update()
    namespace_resolver=lambda scope: f"research-crew-{(scope or 'shared').strip('/').replace('/', '-') or 'shared'}",
)
```

`namespace_resolver` receives the CrewAI scope path (e.g. `"/agent/researcher"`)
for every call that carries one (`save`, `search`, `list_records`, `delete`,
`count`, `get_scope_info`, `list_scopes`, `list_categories`, `reset`).
`get_record()` and `update()` take no scope argument in the CrewAI protocol,
so they always use the static `namespace=` — not the resolver. See
[Known limitations](#known-limitations) for why namespace isolation does not
currently separate data on the hosted runtime.

**CrewAI-level (within one InfoLang namespace)** — use CrewAI's own
`MemoryScope`, which this package supports natively since it preserves the
full scope path on every record and filters on it client-side:

```python
from crewai.memory.memory_scope import MemoryScope

researcher.memory = MemoryScope(root_path="/agent/researcher").bind(memory)
```

## Why this exists

Reliability. Mem0's CrewAI integration has several open issues that all come
down to the same thing: an unvalidated assumption about a shape (a dict key
that isn't always there, a config value that silently isn't applied, a list
assumed where a string arrived). This package's test suite is written
specifically against that failure shape — see [Testing](#testing) — not as a
general-purpose claim of superiority, but because those are documented,
reproducible bugs and this package is tested against reproductions of them:

- **Config not silently ignored.** Every constructor argument is validated
  at `InfoLangStorage(...)` construction time (bad namespace, non-callable
  `namespace_resolver`, conflicting `client=`/`api_key=`, etc. all raise
  `InfoLangCrewAIConfigError` immediately, not "memory looks empty later").
- **No blind indexing into API responses.** Every InfoLang response is read
  through the typed `infolang` SDK models (or explicit multi-key lookups for
  the one loosely-typed route, `GET /v1/memories`); nothing does
  `result["memory"]`-style direct key access against an untyped dict.
  Malformed or missing fields raise `InfoLangCrewAIResponseError` with a
  clear message, or degrade gracefully to defaults on the read path — never
  a raw `KeyError`.
- **Malformed response shapes don't crash a call.** A `hits`/`chunks` field
  that arrives as something other than a list is caught (via the SDK's own
  response validation) and turned into a clear `InfoLangCrewAIResponseError`,
  not a raw `pydantic.ValidationError` escaping `search()`.
- **Provider/auth switching is exercised.** Tests cover constructing
  `InfoLangStorage` against both `api_key` and `dev_key` auth, and swapping
  the paired `embedder=` while `storage=` stays fixed, to catch
  `AttributeError`-class bugs from code that only worked under one
  configuration.

## Why the custom embedder?

CrewAI's `Memory` computes an embedding from your query/content with
`self.embedder` *before* it ever calls the storage backend — `StorageBackend`
methods only receive `list[float]`, never the original string
(`crewai.memory.storage.backend.StorageBackend.search`). InfoLang embeds
server-side from raw text (`POST /v1/recall` takes `query: str`, not a
vector). To bridge that gap without losing InfoLang's real semantic search,
`InfoLangEmbedder` doesn't compute a real embedding — it reversibly encodes
each string as a vector of UTF-8 byte values in `[0, 1]`, and
`InfoLangStorage` decodes that vector back into text before calling InfoLang.

Always pass both together. `InfoLangStorage.search()` validates every
incoming vector and raises `InfoLangCrewAIConfigError` if it doesn't look
like `InfoLangEmbedder`'s output (e.g. a real OpenAI embedding was wired in
by mistake) instead of silently returning wrong or empty results.

## Config reference

`InfoLangStorage(...)` keyword arguments:

| Argument | Default | Notes |
|---|---|---|
| `client` | `None` | Pass a pre-built `infolang.InfoLang` instance. Mutually exclusive with `api_key`/`dev_key`/`base_url`. |
| `api_key` | `None` | Managed cloud API key (`il_live_...`). Falls back to `INFOLANG_API_KEY` env var via the SDK if omitted. |
| `dev_key` | `None` | Self-hosted dev key (`key:namespace`), targets `127.0.0.1:8766`. |
| `base_url` | `None` | Override the API base URL. |
| `namespace` | `"default"` | InfoLang bank name. Must be non-empty. Used by every call, and always used by `get_record()`/`update()`. |
| `namespace_resolver` | `None` | `Callable[[str \| None], str]` mapping a CrewAI scope path to a namespace, for calls that carry scope. |
| `source` | `None` | Default `source` tag passed to `remember()` when a record doesn't set its own. |
| `search_pool` | `50` | `top_k` sent to `/v1/recall` is `max(limit, search_pool)`, oversampling so client-side category/metadata filtering still has enough candidates. |
| `list_page` | `500` | Page size for the `GET /v1/memories` scans backing `list_records`/`count`/`delete`/`update`/`get_record`/scope introspection (see below). |

## Known limitations

**Hosted-runtime namespace behavior (production caveat, not a bug in this
package).** `POST /v1/remember` on the hosted InfoLang runtime currently
ignores the `namespace` field in the request body — every write lands in the
`default` namespace regardless of what is requested. A server-side fix is
pending. Until it ships:

- Namespace-level isolation between crews/agents (the `namespace=` /
  `namespace_resolver=` mechanisms above) does not actually separate data on
  the hosted runtime today — everything ends up in `default`.
- `Memory.recall()`'s server-side namespace filter is therefore also not
  useful for separating crews right now, since nothing outside `default`
  gets written.
- This package's `MemoryScope`-based scope filtering (client-side, from data
  embedded in the stored text) is **not** affected by this bug and works
  correctly today, even against the shared `default` namespace.
- `InfoLangStorage` still sends the `namespace` you configure on every call —
  no code changes will be needed here once the server-side fix lands.

**`remember_batch` is unsupported on the deployed runtime.** The `infolang`
SDK exposes `remember_batch()` (via `POST /v1/execute` with a
`remember_batch` op), but the currently deployed runtime rejects that op.
`InfoLangStorage.save()` therefore issues one `POST /v1/remember` per record,
not a batch call — this is deliberate, not an oversight, and will not change
until the runtime accepts the op.

**No native "get by id" or "filter by metadata" route.** InfoLang's REST
contract is `POST /v1/recall`, `POST /v1/remember`, `GET /v1/memories`,
`DELETE /v1/memories/{id}` — there is no query-by-id or server-side metadata
filter. `get_record()`, `update()`, `delete()` (beyond simple cases),
`list_records()`, `count()`, `get_scope_info()`, `list_scopes()`, and
`list_categories()` are all built on a full namespace scan
(`GET /v1/memories`, page size `list_page`) with local filtering — O(N) per
call, the same pattern the `infolang` SDK itself uses internally for
`reset_namespace()`. This is fine for the bounded memory footprint of a
crew/agent; it is not a substitute for a real secondary index at very large
scale.

**`InfoLangEmbedder` is not a real embedding.** See
["Why the custom embedder?"](#why-the-custom-embedder). Do not use it outside
this package, and do not mix it with a different embedder on the same
`Memory` instance.

**Metadata is embedded as text.** Because `/v1/remember` has no metadata
field, `InfoLangStorage` appends a compact JSON sidecar to the text it
stores, after a sentinel that keeps it out of the human-readable content.
InfoLang's server-side embedding sees that whole string, so very large
`metadata` dicts add non-semantic noise to what gets embedded — keep
metadata small.

## Testing

```bash
pip install -e ".[dev]"
pip install -e "../sdk-python"
ruff check .
mypy
pytest
```

Tests use `respx` to mock the InfoLang HTTP API — no network access and no
real API key are required to run the suite. Coverage gate: 90%
(`--cov-fail-under=90`, see `pyproject.toml`).

## Examples

See `examples/minimal_crew.py` for a runnable crew wired to `InfoLangStorage`.

## License

Apache-2.0
