"""InfoLangStorage: an InfoLang-backed StorageBackend for CrewAI's unified Memory.

Wire it in via::

    from crewai import Crew
    from crewai.memory.unified_memory import Memory
    from infolang_crewai import InfoLangEmbedder, InfoLangStorage

    memory = Memory(
        storage=InfoLangStorage(api_key="il_live_...", namespace="my-crew"),
        embedder=InfoLangEmbedder(),
    )
    crew = Crew(agents=[...], tasks=[...], memory=memory)

See the package README for namespace-scoping patterns (per-crew vs
per-agent) and the full "Known limitations" list, including why
``InfoLangEmbedder`` must always be paired with this class and why
``get_record()``/``update()`` require a single static ``namespace``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from crewai.memory.storage.backend import StorageBackend
from crewai.memory.types import MemoryRecord, ScopeInfo
from infolang import InfoLang
from infolang.errors import InfoLangError
from pydantic import ValidationError

from . import _codec
from .embedder import decode_embedding
from .errors import InfoLangCrewAIConfigError, InfoLangCrewAIResponseError

_log = logging.getLogger("infolang_crewai")

_DEFAULT_NAMESPACE = "default"
_DEFAULT_SEARCH_POOL = 50
_DEFAULT_LIST_PAGE = 500


class InfoLangStorage(StorageBackend):
    """CrewAI ``StorageBackend`` backed by the InfoLang hosted memory API.

    Namespace scoping:
        Pass ``namespace="my-crew"`` for one InfoLang bank per crew (the
        common case: build one ``InfoLangStorage``/``Memory`` per ``Crew``).
        For per-agent isolation *at the InfoLang bank level*, either
        construct a separate ``InfoLangStorage`` per agent (each with its
        own ``namespace``), or pass ``namespace_resolver`` -- a callable
        that maps a CrewAI scope path (e.g. ``"/agent/researcher"``) to an
        InfoLang namespace. ``namespace_resolver`` only affects calls that
        receive scope information (``save``, ``search``, ``list_records``,
        ``delete``, ``count``, ``get_scope_info``, ``list_scopes``,
        ``list_categories``, ``reset``); ``get_record()`` and ``update()``
        take no scope argument in the CrewAI protocol, so they always use
        the static ``namespace``.

        For finer-grained partitioning *within* one InfoLang namespace, use
        CrewAI's own ``MemoryScope(root_path=...)`` -- this class preserves
        the full scope path in every record and filters on it locally.

    Known limitation: the hosted InfoLang runtime currently ignores the
    ``namespace`` field on ``POST /v1/remember`` (writes land in
    ``default`` regardless of what is requested; server-side fix pending).
    Until that ships, namespace-level isolation between crews/agents does
    not actually separate data on the hosted runtime -- see the package
    README's "Known limitations" section. Scope-prefix filtering (the
    ``MemoryScope`` path above) is enforced client-side from data embedded
    in the stored text and is unaffected by that bug.
    """

    def __init__(
        self,
        *,
        client: InfoLang | None = None,
        api_key: str | None = None,
        dev_key: str | None = None,
        base_url: str | None = None,
        namespace: str = _DEFAULT_NAMESPACE,
        namespace_resolver: Callable[[str | None], str] | None = None,
        source: str | None = None,
        search_pool: int = _DEFAULT_SEARCH_POOL,
        list_page: int = _DEFAULT_LIST_PAGE,
    ) -> None:
        other_creds_given = api_key is not None or dev_key is not None or base_url is not None
        if client is not None and other_creds_given:
            raise InfoLangCrewAIConfigError(
                "InfoLangStorage: pass either client=<InfoLang instance> or "
                "api_key=/dev_key=/base_url=, not both. Received a client and "
                "at least one of api_key/dev_key/base_url."
            )
        if not isinstance(namespace, str) or not namespace.strip():
            raise InfoLangCrewAIConfigError(
                f"InfoLangStorage: namespace must be a non-empty string, got {namespace!r}."
            )
        if namespace_resolver is not None and not callable(namespace_resolver):
            raise InfoLangCrewAIConfigError(
                "InfoLangStorage: namespace_resolver must be callable "
                f"(str | None) -> str, got {type(namespace_resolver).__name__}."
            )
        if search_pool < 1:
            raise InfoLangCrewAIConfigError(
                f"InfoLangStorage: search_pool must be >= 1, got {search_pool}."
            )
        if list_page < 1:
            raise InfoLangCrewAIConfigError(
                f"InfoLangStorage: list_page must be >= 1, got {list_page}."
            )

        try:
            self._client = client or InfoLang(api_key=api_key, dev_key=dev_key, base_url=base_url)
        except InfoLangError as exc:
            raise InfoLangCrewAIConfigError(
                f"InfoLangStorage: failed to construct client: {exc}"
            ) from exc

        self._namespace = namespace
        self._namespace_resolver = namespace_resolver
        self._source = source
        self._search_pool = search_pool
        self._list_page = list_page

    # -- namespace resolution ------------------------------------------------

    def _resolve_namespace(self, scope: str | None) -> str:
        if self._namespace_resolver is not None:
            resolved = self._namespace_resolver(scope)
            if not isinstance(resolved, str) or not resolved.strip():
                raise InfoLangCrewAIConfigError(
                    "InfoLangStorage: namespace_resolver must return a non-empty "
                    f"string, got {resolved!r} for scope={scope!r}."
                )
            return resolved
        return self._namespace

    # -- save -----------------------------------------------------------------

    def save(self, records: list[MemoryRecord]) -> None:
        for record in records:
            self._save_one(record)

    async def asave(self, records: list[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    def _save_one(self, record: MemoryRecord) -> None:
        if not isinstance(record, MemoryRecord):
            raise InfoLangCrewAIConfigError(
                "InfoLangStorage.save() expects MemoryRecord instances, "
                f"got {type(record).__name__}."
            )
        text = _codec.encode(record)
        tags = ",".join(record.categories) if record.categories else None
        ns = self._resolve_namespace(record.scope)
        try:
            self._client.remember(
                text, namespace=ns, source=record.source or self._source, tags=tags
            )
        except InfoLangError as exc:
            raise InfoLangCrewAIResponseError(
                f"InfoLangStorage.save() failed for record {record.id!r}: {exc}"
            ) from exc

    # -- search ---------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        query_text = decode_embedding(query_embedding)
        if not query_text:
            return []
        ns = self._resolve_namespace(scope_prefix)
        top_k = max(limit, self._search_pool)
        try:
            result = self._client.recall(query_text, namespace=ns, top_k=top_k)
        except InfoLangError as exc:
            raise InfoLangCrewAIResponseError(f"InfoLangStorage.search() failed: {exc}") from exc
        except ValidationError as exc:
            # The runtime returned a shape the SDK's response model could not
            # validate (e.g. "hits" was a string, not a list -- the literal
            # crewAIInc/crewAI#3152 failure class). Never let a raw pydantic
            # error escape a StorageBackend method.
            raise InfoLangCrewAIResponseError(
                f"InfoLangStorage.search() got an unexpected response shape from InfoLang: {exc}"
            ) from exc

        matches: list[tuple[MemoryRecord, float]] = []
        for chunk in result.chunks:
            record = _codec.decode(chunk.text, fallback_id=chunk.id)
            if scope_prefix and not record.scope.startswith(scope_prefix):
                continue
            if categories and not (set(categories) & set(record.categories)):
                continue
            if metadata_filter and not _matches_metadata(record.metadata, metadata_filter):
                continue
            score = _clamp_unit(chunk.score if chunk.score is not None else 0.5)
            if score < min_score:
                continue
            matches.append((record, score))
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return matches[:limit]

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
        )

    # -- listing / scanning -----------------------------------------------------
    #
    # InfoLang's REST contract has no "get by id" or "filter by metadata" route
    # (only POST /v1/recall, POST /v1/remember, GET /v1/memories, DELETE
    # /v1/memories/{id}). delete/update/get_record/list_records/count/
    # get_scope_info/list_scopes/list_categories are all built on top of a full
    # namespace scan (GET /v1/memories) -- O(N) per call, same pattern the
    # infolang SDK itself uses for reset_namespace(). Fine for the bounded
    # memory footprint of a crew/agent; not a substitute for a real index at
    # very large scale. See README "Known limitations".

    def _list_all(self, namespace: str) -> list[tuple[str, MemoryRecord]]:
        try:
            raw = self._client.list_recent(namespace=namespace, n=self._list_page)
        except InfoLangError as exc:
            raise InfoLangCrewAIResponseError(
                f"InfoLangStorage: list_recent failed: {exc}"
            ) from exc

        if not isinstance(raw, list):
            # Defensive: the runtime is documented to return a list; never
            # trust that blindly (crewAIInc/crewAI#3152-class defense).
            _log.warning(
                "InfoLangStorage: list_recent returned %s, expected list; treating as empty.",
                type(raw).__name__,
            )
            return []

        out: list[tuple[str, MemoryRecord]] = []
        for item in raw:
            parsed = _extract_id_text(item)
            if parsed is None:
                continue
            mem_id, text = parsed
            out.append((mem_id, _codec.decode(text, fallback_id=mem_id)))
        return out

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        ns = self._resolve_namespace(scope_prefix)
        records = [r for _, r in self._list_all(ns)]
        if scope_prefix:
            records = [r for r in records if r.scope.startswith(scope_prefix)]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    def count(self, scope_prefix: str | None = None) -> int:
        ns = self._resolve_namespace(scope_prefix)
        records = [r for _, r in self._list_all(ns)]
        if scope_prefix:
            records = [r for r in records if r.scope.startswith(scope_prefix)]
        return len(records)

    def get_record(self, record_id: str) -> MemoryRecord | None:
        for _, record in self._list_all(self._namespace):
            if record.id == record_id:
                return record
        return None

    def update(self, record: MemoryRecord) -> None:
        ns = self._resolve_namespace(record.scope)
        for mem_id, existing in self._list_all(ns):
            if existing.id == record.id:
                try:
                    self._client.forget(mem_id, namespace=ns)
                except InfoLangError as exc:
                    raise InfoLangCrewAIResponseError(
                        f"InfoLangStorage.update() failed to remove the prior version "
                        f"of record {record.id!r}: {exc}"
                    ) from exc
                break
        self._save_one(record)

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        ns = self._resolve_namespace(scope_prefix)
        deleted = 0
        for mem_id, record in self._list_all(ns):
            if not _matches_delete_criteria(
                record,
                scope_prefix=scope_prefix,
                categories=categories,
                record_ids=record_ids,
                older_than=older_than,
                metadata_filter=metadata_filter,
            ):
                continue
            try:
                self._client.forget(mem_id, namespace=ns)
            except InfoLangError as exc:
                raise InfoLangCrewAIResponseError(
                    f"InfoLangStorage.delete() failed to remove record {record.id!r}: {exc}"
                ) from exc
            deleted += 1
        return deleted

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.delete, scope_prefix, categories, record_ids, older_than, metadata_filter
        )

    def reset(self, scope_prefix: str | None = None) -> None:
        self.delete(scope_prefix=scope_prefix)

    # -- scope / category introspection -----------------------------------------

    def get_scope_info(self, scope: str) -> ScopeInfo:
        records = self.list_records(scope_prefix=scope, limit=self._list_page)
        if not records:
            return ScopeInfo(path=scope)
        categories = sorted({c for r in records for c in r.categories})
        created = [r.created_at for r in records]
        return ScopeInfo(
            path=scope,
            record_count=len(records),
            categories=categories,
            oldest_record=min(created),
            newest_record=max(created),
            child_scopes=_immediate_children(scope, (r.scope for r in records)),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        prefix = None if parent in ("/", "") else parent
        records = self.list_records(scope_prefix=prefix, limit=self._list_page)
        return _immediate_children(parent, (r.scope for r in records))

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        records = self.list_records(scope_prefix=scope_prefix, limit=self._list_page)
        counts: dict[str, int] = {}
        for record in records:
            for category in record.categories:
                counts[category] = counts.get(category, 0) + 1
        return counts


def _extract_id_text(item: Any) -> tuple[str, str] | None:
    """Defensively pull ``(id, text)`` out of a ``GET /v1/memories`` item.

    The runtime's ``RecentResponse.memories`` schema is
    ``additionalProperties: true`` -- shape is not guaranteed. Mirrors the
    infolang SDK's own ``_memory_id`` helper: try several plausible keys,
    skip (never raise) on anything that does not have both an id and text.
    """

    if not isinstance(item, dict):
        _log.warning("InfoLangStorage: skipping non-dict list_recent item: %s", type(item).__name__)
        return None
    mem_id = next(
        (item[k] for k in ("id", "memory_id", "i") if isinstance(item.get(k), str) and item.get(k)),
        None,
    )
    text = next((item[k] for k in ("text", "t", "content") if isinstance(item.get(k), str)), None)
    if mem_id is None or text is None:
        _log.warning(
            "InfoLangStorage: skipping list_recent item missing id/text (keys=%s)",
            sorted(item.keys()),
        )
        return None
    return mem_id, text


def _matches_metadata(metadata: dict[str, Any], filt: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filt.items())


def _matches_delete_criteria(
    record: MemoryRecord,
    *,
    scope_prefix: str | None,
    categories: list[str] | None,
    record_ids: list[str] | None,
    older_than: datetime | None,
    metadata_filter: dict[str, Any] | None,
) -> bool:
    if scope_prefix and not record.scope.startswith(scope_prefix):
        return False
    if categories and not (set(categories) & set(record.categories)):
        return False
    if record_ids and record.id not in record_ids:
        return False
    if older_than and record.created_at >= older_than:
        return False
    return not (metadata_filter and not _matches_metadata(record.metadata, metadata_filter))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _immediate_children(parent: str, scopes: Any) -> list[str]:
    root = "" if parent in ("/", "") else parent.rstrip("/")
    children: set[str] = set()
    for scope in scopes:
        if not scope.startswith(root + "/") and scope != root:
            continue
        remainder = scope[len(root) :].lstrip("/")
        if not remainder:
            continue
        head = remainder.split("/", 1)[0]
        children.add(f"{root}/{head}" if root else f"/{head}")
    return sorted(children)
