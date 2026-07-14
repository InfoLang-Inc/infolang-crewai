"""Sidecar codec: round-trips full MemoryRecord fields through InfoLang's plain-text store.

InfoLang's REST contract (``POST /v1/remember``) only carries ``text``,
``namespace``, ``source``, and ``tags`` -- there is no metadata/scope/
importance/created_at field. CrewAI's ``StorageBackend`` protocol needs those
fields to survive a save -> search -> update -> delete round trip. This
module packs them into a small JSON envelope appended to the stored text
(sentinel-delimited) so InfoLang's server-side embedding still weights mostly
toward the human-readable content, and unpacks that envelope back into a
``MemoryRecord`` on read.

Text written by something other than this package (no sentinel, or malformed
JSON) is treated as opaque content with default field values -- this module
never raises on foreign or corrupted data, only degrades gracefully to
defaults. Read-path leniency plus write-path strictness (see
:mod:`infolang_crewai.storage`) is a deliberate split: never crash a
``search()``/``list_records()`` call because of one bad row.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from crewai.memory.types import MemoryRecord
from pydantic import ValidationError

_SENTINEL = "\x1e__infolang_crewai_v1__\x1e"


def encode(record: MemoryRecord) -> str:
    """Pack a MemoryRecord into InfoLang-storable text: content + JSON sidecar."""

    sidecar = {
        "id": record.id,
        "scope": record.scope,
        "categories": record.categories,
        "metadata": record.metadata,
        "importance": record.importance,
        "source": record.source,
        "private": record.private,
        "created_at": record.created_at.isoformat(),
        "last_accessed": record.last_accessed.isoformat(),
    }
    return f"{record.content}{_SENTINEL}{json.dumps(sidecar, ensure_ascii=False)}"


def decode(text: str, *, fallback_id: str, default_scope: str = "/") -> MemoryRecord:
    """Unpack InfoLang-stored text back into a MemoryRecord.

    Falls back to sane defaults for text this package did not write (no
    sentinel, a sidecar that fails to parse, or one whose values fail
    ``MemoryRecord`` validation) instead of raising.
    """

    if not isinstance(text, str) or _SENTINEL not in text:
        return MemoryRecord(id=fallback_id, content=text or "", scope=default_scope)

    content, _, raw_sidecar = text.partition(_SENTINEL)
    try:
        sidecar = json.loads(raw_sidecar)
        if not isinstance(sidecar, dict):
            raise ValueError("sidecar is not a JSON object")
    except (json.JSONDecodeError, ValueError):
        return MemoryRecord(id=fallback_id, content=content, scope=default_scope)

    kwargs: dict[str, Any] = {
        "id": sidecar.get("id") or fallback_id,
        "content": content,
        "scope": sidecar.get("scope") or default_scope,
        "categories": _as_str_list(sidecar.get("categories")),
        "metadata": sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {},
        "importance": _coerce_unit_float(sidecar.get("importance"), 0.5),
        "source": sidecar.get("source") if isinstance(sidecar.get("source"), str) else None,
        "private": bool(sidecar.get("private", False)),
    }
    created_at = _parse_dt(sidecar.get("created_at"))
    if created_at is not None:
        kwargs["created_at"] = created_at
    last_accessed = _parse_dt(sidecar.get("last_accessed"))
    if last_accessed is not None:
        kwargs["last_accessed"] = last_accessed

    try:
        return MemoryRecord(**kwargs)
    except ValidationError:
        # A hand-edited or otherwise-foreign sidecar can carry a value that
        # parses as JSON but still fails MemoryRecord's own field
        # validation (e.g. an id/scope of the wrong type). Degrade to a
        # bare record with the recovered content rather than raising out of
        # a search()/list_records() call.
        return MemoryRecord(id=fallback_id, content=content, scope=default_scope)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str)]


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _coerce_unit_float(value: Any, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if 0.0 <= f <= 1.0 else default
