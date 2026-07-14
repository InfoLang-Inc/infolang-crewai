"""Empty results must be handled gracefully everywhere, not treated as errors."""

from __future__ import annotations

import httpx
import respx

from infolang_crewai import InfoLangStorage
from infolang_crewai.embedder import encode_text

BASE_URL = "https://mock.infolang.test"


def _storage() -> InfoLangStorage:
    return InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")


def test_search_with_no_hits_returns_empty_list(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    storage = _storage()

    assert storage.search(encode_text("anything"), limit=5) == []


def test_search_with_empty_query_short_circuits_without_http_call(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    storage = _storage()

    assert storage.search([], limit=5) == []
    assert route.call_count == 0


def test_list_records_with_no_memories_returns_empty_list(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    assert storage.list_records() == []


def test_count_with_no_memories_is_zero(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    assert storage.count() == 0


def test_get_record_not_found_returns_none(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    assert storage.get_record("does-not-exist") is None


def test_delete_with_no_matches_returns_zero(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    assert storage.delete(record_ids=["nope"]) == 0


def test_reset_on_empty_namespace_does_not_raise(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    storage.reset()  # must not raise


def test_list_categories_on_empty_namespace_is_empty_dict(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    assert storage.list_categories() == {}


def test_get_scope_info_on_empty_scope_has_zero_count(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    info = storage.get_scope_info("/agent/researcher")

    assert info.record_count == 0
    assert info.oldest_record is None
    assert info.newest_record is None
    assert info.child_scopes == []


def test_list_scopes_on_empty_namespace_is_empty(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage()

    assert storage.list_scopes() == []
