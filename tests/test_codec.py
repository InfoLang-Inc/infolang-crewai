"""Unit tests for the sidecar codec: field round-tripping and graceful
degradation on foreign / corrupted text."""

from __future__ import annotations

from datetime import UTC, datetime

from crewai.memory.types import MemoryRecord

from infolang_crewai import _codec


def test_round_trips_all_fields() -> None:
    record = MemoryRecord(
        id="rec-123",
        content="remember this",
        scope="/agent/researcher",
        categories=["fact", "urgent"],
        metadata={"key": "value", "n": 3},
        importance=0.75,
        source="user-42",
        private=True,
    )

    decoded = _codec.decode(_codec.encode(record), fallback_id="unused")

    assert decoded.id == record.id
    assert decoded.content == record.content
    assert decoded.scope == record.scope
    assert decoded.categories == record.categories
    assert decoded.metadata == record.metadata
    assert decoded.importance == record.importance
    assert decoded.source == record.source
    assert decoded.private == record.private
    assert decoded.created_at == record.created_at
    assert decoded.last_accessed == record.last_accessed


def test_foreign_text_without_sentinel_becomes_bare_record() -> None:
    decoded = _codec.decode("plain text from somewhere else", fallback_id="fallback-id")

    assert decoded.id == "fallback-id"
    assert decoded.content == "plain text from somewhere else"
    assert decoded.scope == "/"
    assert decoded.categories == []
    assert decoded.metadata == {}


def test_malformed_json_sidecar_degrades_gracefully() -> None:
    text = "the content" + _codec._SENTINEL + "{not valid json"

    decoded = _codec.decode(text, fallback_id="fallback-id")

    assert decoded.id == "fallback-id"
    assert decoded.content == "the content"


def test_sidecar_that_is_not_a_json_object_degrades_gracefully() -> None:
    text = "the content" + _codec._SENTINEL + '["just", "a", "list"]'

    decoded = _codec.decode(text, fallback_id="fallback-id")

    assert decoded.id == "fallback-id"
    assert decoded.content == "the content"


def test_out_of_range_importance_falls_back_to_default() -> None:
    text = "content" + _codec._SENTINEL + '{"importance": 42}'

    decoded = _codec.decode(text, fallback_id="fallback-id")

    assert decoded.importance == 0.5


def test_non_numeric_importance_falls_back_to_default() -> None:
    text = "content" + _codec._SENTINEL + '{"importance": "not-a-number"}'

    decoded = _codec.decode(text, fallback_id="fallback-id")

    assert decoded.importance == 0.5


def test_non_list_categories_become_empty_list() -> None:
    text = "content" + _codec._SENTINEL + '{"categories": "not-a-list"}'

    decoded = _codec.decode(text, fallback_id="fallback-id")

    assert decoded.categories == []


def test_non_dict_metadata_becomes_empty_dict() -> None:
    text = "content" + _codec._SENTINEL + '{"metadata": "not-a-dict"}'

    decoded = _codec.decode(text, fallback_id="fallback-id")

    assert decoded.metadata == {}


def test_invalid_created_at_falls_back_to_record_default() -> None:
    text = "content" + _codec._SENTINEL + '{"created_at": "not-a-timestamp"}'

    decoded = _codec.decode(text, fallback_id="fallback-id")

    # Falls back to MemoryRecord's own default_factory (now), just check it
    # parsed as a real, recent datetime rather than raising.
    assert isinstance(decoded.created_at, datetime)
    assert decoded.created_at.tzinfo is None or decoded.created_at.tzinfo == UTC


def test_sidecar_value_that_fails_memory_record_validation_degrades() -> None:
    # scope must be a str; a nested object here fails MemoryRecord validation
    # even though it is valid JSON.
    text = "content" + _codec._SENTINEL + '{"scope": {"nested": "object"}}'

    decoded = _codec.decode(text, fallback_id="fallback-id")

    assert decoded.id == "fallback-id"
    assert decoded.content == "content"
    assert decoded.scope == "/"


def test_non_string_input_does_not_raise() -> None:
    decoded = _codec.decode(None, fallback_id="fallback-id")  # type: ignore[arg-type]

    assert decoded.id == "fallback-id"
    assert decoded.content == ""
