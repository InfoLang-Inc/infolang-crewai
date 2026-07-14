"""Malformed / unexpected response shape failure classes.

Covers the crewAIInc/crewAI#3152 ("Expected list but got str") and #2591
(``KeyError`` on ``result["memory"]``) failure classes: the runtime returns
something other than the documented shape, and this package must never let a
raw ``KeyError``/``pydantic.ValidationError``/``TypeError`` escape a
``StorageBackend`` method.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from crewai.memory.types import MemoryRecord

from infolang_crewai import InfoLangCrewAIResponseError, InfoLangStorage
from infolang_crewai.embedder import encode_text

BASE_URL = "https://mock.infolang.test"


def test_recall_hits_as_string_raises_clear_error(respx_mock: respx.MockRouter) -> None:
    # crewAIInc/crewAI#3152 class: server sends a string where a list is
    # documented ("hits" should be list[RecallHit]).
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json="oops, not even a dict")
    )
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")

    with pytest.raises(InfoLangCrewAIResponseError, match="unexpected response shape"):
        storage.search(encode_text("find me something"), limit=5)


def test_recall_missing_hits_key_returns_empty_not_error(respx_mock: respx.MockRouter) -> None:
    # No "hits" key at all -- degrade to empty results, don't crash.
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"namespace": "ns"})
    )
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")

    results = storage.search(encode_text("query"), limit=5)

    assert results == []


def test_recall_hit_missing_required_field_raises_clear_error(
    respx_mock: respx.MockRouter,
) -> None:
    # A hit missing its id is a shape violation of the documented RecallHit
    # schema -- must not surface as a raw pydantic error.
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": [{"id": None, "text": "partial"}]})
    )
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")

    with pytest.raises(InfoLangCrewAIResponseError, match="unexpected response shape"):
        storage.search(encode_text("query"), limit=5)


def test_list_recent_memories_as_string_degrades_to_empty(
    respx_mock: respx.MockRouter,
) -> None:
    # GET /v1/memories documents memories: array, but the schema is
    # additionalProperties: true -- a misbehaving server sending a string
    # instead must not crash list_records()/count()/delete().
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": "not-a-list"})
    )
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")

    assert storage.list_records() == []
    assert storage.count() == 0


def test_list_recent_items_missing_id_or_text_are_skipped_not_fatal(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(
            200,
            json={
                "memories": [
                    {"id": "good-1", "text": "keep me"},
                    {"text": "no id here"},
                    {"id": "no-text-here"},
                    "not-even-a-dict",
                    42,
                    {"i": "good-2", "t": "compact keys also work"},
                ]
            },
        )
    )
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")

    records = storage.list_records()

    contents = {r.content for r in records}
    assert "keep me" in contents
    assert "compact keys also work" in contents
    assert len(records) == 2


def test_remember_response_missing_id_key_does_not_raise(respx_mock: respx.MockRouter) -> None:
    # crewAIInc/crewAI#2591 class: the response has no "id"/"memory" key at
    # all. save() must not do a blind result["id"]-style lookup.
    respx_mock.post(f"{BASE_URL}/v1/remember").mock(return_value=httpx.Response(200, json={}))
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")
    record = MemoryRecord(content="a fact worth keeping", scope="/")

    storage.save([record])  # must not raise


def test_remember_non_json_body_does_not_raise(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(
            200, content=b"not json at all", headers={"content-type": "text/plain"}
        )
    )
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")
    record = MemoryRecord(content="a fact", scope="/")

    # SDK's parse_remember() treats a non-dict body as {}; RememberResult has
    # no required fields (extra="allow"), so this degrades, it never raises.
    storage.save([record])
