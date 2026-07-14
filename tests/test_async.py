"""Async StorageBackend methods (asave/asearch/adelete) wrap the sync path correctly."""

from __future__ import annotations

import httpx
import respx
from crewai.memory.types import MemoryRecord

from infolang_crewai import InfoLangStorage, _codec
from infolang_crewai.embedder import encode_text

BASE_URL = "https://mock.infolang.test"


def _storage() -> InfoLangStorage:
    return InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")


async def test_asave_persists_record(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "il-1"})
    )
    storage = _storage()
    record = MemoryRecord(id="rec-1", content="async fact", scope="/")

    await storage.asave([record])

    assert route.call_count == 1


async def test_asearch_returns_decoded_matches(respx_mock: respx.MockRouter) -> None:
    record = MemoryRecord(id="rec-1", content="async findable", scope="/")
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200, json={"hits": [{"id": "il-1", "text": _codec.encode(record), "similarity": 0.8}]}
        )
    )
    storage = _storage()

    results = await storage.asearch(encode_text("query"), limit=5)

    assert len(results) == 1
    assert results[0][0].content == "async findable"


async def test_adelete_removes_matching_record(respx_mock: respx.MockRouter) -> None:
    record = MemoryRecord(id="rec-1", content="to delete", scope="/")
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(
            200, json={"memories": [{"id": "il-1", "text": _codec.encode(record)}]}
        )
    )
    respx_mock.delete(f"{BASE_URL}/v1/memories/il-1").mock(
        return_value=httpx.Response(200, json={})
    )
    storage = _storage()

    deleted = await storage.adelete(record_ids=["rec-1"])

    assert deleted == 1
