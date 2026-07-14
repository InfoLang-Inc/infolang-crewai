"""Full StorageBackend protocol coverage against a mocked InfoLang API."""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import respx
from crewai.memory.types import MemoryRecord

from infolang_crewai import InfoLangStorage, _codec
from infolang_crewai.embedder import encode_text

BASE_URL = "https://mock.infolang.test"


def _storage(**kwargs: object) -> InfoLangStorage:
    kwargs.setdefault("api_key", "il_test")
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("namespace", "ns")
    return InfoLangStorage(**kwargs)  # type: ignore[arg-type]


def _hit(record: MemoryRecord, score: float = 0.9) -> dict:
    return {"id": f"il-{record.id}", "text": _codec.encode(record), "similarity": score}


def _memory_item(record: MemoryRecord) -> dict:
    return {"id": f"il-{record.id}", "text": _codec.encode(record)}


# -- save -----------------------------------------------------------------


def test_save_sends_expected_request_body(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "il-1"})
    )
    record = MemoryRecord(
        id="rec-1",
        content="remember this",
        scope="/",
        categories=["fact", "urgent"],
        source="user-1",
    )
    storage = _storage()

    storage.save([record])

    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body["namespace"] == "ns"
    assert body["source"] == "user-1"
    assert body["tags"] == "fact,urgent"
    assert body["text"].startswith("remember this")
    assert _codec._SENTINEL in body["text"]


def test_save_multiple_records_issues_one_remember_per_record(
    respx_mock: respx.MockRouter,
) -> None:
    # Deliberate: the hosted runtime rejects the remember_batch op, so this
    # package never calls it. See README "Known limitations".
    route = respx_mock.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "il-x"})
    )
    records = [MemoryRecord(content=f"fact {i}", scope="/") for i in range(3)]
    storage = _storage()

    storage.save(records)

    assert route.call_count == 3


def test_save_rejects_non_memory_record_input() -> None:
    import pytest

    from infolang_crewai import InfoLangCrewAIConfigError

    storage = _storage()
    with pytest.raises(InfoLangCrewAIConfigError, match="MemoryRecord"):
        storage.save(["not-a-memory-record"])  # type: ignore[list-item]


# -- search -----------------------------------------------------------------


def test_search_returns_decoded_records_sorted_by_score(respx_mock: respx.MockRouter) -> None:
    low = MemoryRecord(id="low", content="low score", scope="/")
    high = MemoryRecord(id="high", content="high score", scope="/")
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200, json={"hits": [_hit(low, 0.3), _hit(high, 0.95)]}
        )
    )
    storage = _storage()

    results = storage.search(encode_text("query"), limit=5)

    assert [r.id for r, _ in results] == ["high", "low"]


def test_search_respects_limit(respx_mock: respx.MockRouter) -> None:
    hits = [_hit(MemoryRecord(id=str(i), content=f"c{i}", scope="/"), 0.5) for i in range(10)]
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": hits})
    )
    storage = _storage()

    results = storage.search(encode_text("query"), limit=3)

    assert len(results) == 3


def test_search_filters_by_scope_prefix(respx_mock: respx.MockRouter) -> None:
    in_scope = MemoryRecord(id="in", content="in scope", scope="/agent/researcher")
    out_of_scope = MemoryRecord(id="out", content="out of scope", scope="/agent/writer")
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": [_hit(in_scope), _hit(out_of_scope)]})
    )
    storage = _storage()

    results = storage.search(encode_text("query"), scope_prefix="/agent/researcher", limit=5)

    assert [r.id for r, _ in results] == ["in"]


def test_search_filters_by_categories(respx_mock: respx.MockRouter) -> None:
    matching = MemoryRecord(id="match", content="c", scope="/", categories=["urgent"])
    non_matching = MemoryRecord(id="no-match", content="c", scope="/", categories=["low-priority"])
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": [_hit(matching), _hit(non_matching)]})
    )
    storage = _storage()

    results = storage.search(encode_text("query"), categories=["urgent"], limit=5)

    assert [r.id for r, _ in results] == ["match"]


def test_search_filters_by_metadata(respx_mock: respx.MockRouter) -> None:
    matching = MemoryRecord(id="match", content="c", scope="/", metadata={"team": "eng"})
    non_matching = MemoryRecord(id="no-match", content="c", scope="/", metadata={"team": "sales"})
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": [_hit(matching), _hit(non_matching)]})
    )
    storage = _storage()

    results = storage.search(encode_text("query"), metadata_filter={"team": "eng"}, limit=5)

    assert [r.id for r, _ in results] == ["match"]


def test_search_filters_by_min_score(respx_mock: respx.MockRouter) -> None:
    low = MemoryRecord(id="low", content="c", scope="/")
    high = MemoryRecord(id="high", content="c", scope="/")
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": [_hit(low, 0.1), _hit(high, 0.9)]})
    )
    storage = _storage()

    results = storage.search(encode_text("query"), min_score=0.5, limit=5)

    assert [r.id for r, _ in results] == ["high"]


def test_search_sends_oversampled_top_k(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    storage = _storage(search_pool=77)

    storage.search(encode_text("query"), limit=5)

    body = json.loads(route.calls[0].request.content)
    assert body["top_k"] == 77  # max(limit=5, search_pool=77)


# -- list_records / count ----------------------------------------------------


def test_list_records_sorted_newest_first(respx_mock: respx.MockRouter) -> None:
    old = MemoryRecord(id="old", content="old", scope="/", created_at=datetime(2020, 1, 1))
    new = MemoryRecord(id="new", content="new", scope="/", created_at=datetime(2024, 1, 1))
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": [_memory_item(old), _memory_item(new)]})
    )
    storage = _storage()

    records = storage.list_records()

    assert [r.id for r in records] == ["new", "old"]


def test_list_records_respects_limit_and_offset(respx_mock: respx.MockRouter) -> None:
    items = [
        _memory_item(
            MemoryRecord(
                id=str(i), content=f"c{i}", scope="/", created_at=datetime(2024, 1, i + 1)
            )
        )
        for i in range(5)
    ]
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": items})
    )
    storage = _storage()

    page = storage.list_records(limit=2, offset=1)

    assert len(page) == 2
    # newest-first order is [4, 3, 2, 1, 0]; offset=1, limit=2 -> [3, 2]
    assert page[0].id == "3"
    assert page[1].id == "2"


def test_count_matches_scope_filtered_list_length(respx_mock: respx.MockRouter) -> None:
    in_scope = MemoryRecord(id="in", content="c", scope="/agent/a")
    out_of_scope = MemoryRecord(id="out", content="c", scope="/agent/b")
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(
            200, json={"memories": [_memory_item(in_scope), _memory_item(out_of_scope)]}
        )
    )
    storage = _storage()

    assert storage.count(scope_prefix="/agent/a") == 1


# -- get_record / update -----------------------------------------------------


def test_get_record_finds_matching_id(respx_mock: respx.MockRouter) -> None:
    target = MemoryRecord(id="target", content="findme", scope="/")
    other = MemoryRecord(id="other", content="skip", scope="/")
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(
            200, json={"memories": [_memory_item(target), _memory_item(other)]}
        )
    )
    storage = _storage()

    found = storage.get_record("target")

    assert found is not None
    assert found.content == "findme"


def test_update_removes_old_then_saves_new(respx_mock: respx.MockRouter) -> None:
    existing = MemoryRecord(id="rec-1", content="old content", scope="/")
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": [_memory_item(existing)]})
    )
    delete_route = respx_mock.delete(f"{BASE_URL}/v1/memories/il-rec-1").mock(
        return_value=httpx.Response(200, json={})
    )
    remember_route = respx_mock.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "il-rec-1-v2"})
    )
    storage = _storage()
    updated = MemoryRecord(id="rec-1", content="new content", scope="/")

    storage.update(updated)

    assert delete_route.call_count == 1
    assert remember_route.call_count == 1
    body = json.loads(remember_route.calls[0].request.content)
    assert body["text"].startswith("new content")


# -- delete / reset -----------------------------------------------------------


def test_delete_by_record_ids(respx_mock: respx.MockRouter) -> None:
    keep = MemoryRecord(id="keep", content="c", scope="/")
    remove = MemoryRecord(id="remove", content="c", scope="/")
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(
            200, json={"memories": [_memory_item(keep), _memory_item(remove)]}
        )
    )
    delete_route = respx_mock.delete(f"{BASE_URL}/v1/memories/il-remove").mock(
        return_value=httpx.Response(200, json={})
    )
    storage = _storage()

    deleted = storage.delete(record_ids=["remove"])

    assert deleted == 1
    assert delete_route.call_count == 1


def test_delete_by_older_than(respx_mock: respx.MockRouter) -> None:
    old = MemoryRecord(id="old", content="c", scope="/", created_at=datetime(2020, 1, 1))
    new = MemoryRecord(id="new", content="c", scope="/", created_at=datetime(2030, 1, 1))
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": [_memory_item(old), _memory_item(new)]})
    )
    respx_mock.delete(f"{BASE_URL}/v1/memories/il-old").mock(
        return_value=httpx.Response(200, json={})
    )
    storage = _storage()

    deleted = storage.delete(older_than=datetime(2025, 1, 1))

    assert deleted == 1


def test_reset_deletes_everything_in_namespace(respx_mock: respx.MockRouter) -> None:
    records = [MemoryRecord(id=str(i), content=f"c{i}", scope="/") for i in range(3)]
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": [_memory_item(r) for r in records]})
    )
    delete_route = respx_mock.delete(url__regex=r"/v1/memories/il-\d").mock(
        return_value=httpx.Response(200, json={})
    )
    storage = _storage()

    storage.reset()

    assert delete_route.call_count == 3


# -- scope / category introspection -------------------------------------------


def test_get_scope_info_aggregates_categories_and_timestamps(respx_mock: respx.MockRouter) -> None:
    r1 = MemoryRecord(
        id="1", content="c", scope="/agent/a", categories=["x"], created_at=datetime(2020, 1, 1)
    )
    r2 = MemoryRecord(
        id="2", content="c", scope="/agent/a", categories=["y"], created_at=datetime(2024, 1, 1)
    )
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": [_memory_item(r1), _memory_item(r2)]})
    )
    storage = _storage()

    info = storage.get_scope_info("/agent/a")

    assert info.record_count == 2
    assert info.categories == ["x", "y"]
    assert info.oldest_record == datetime(2020, 1, 1)
    assert info.newest_record == datetime(2024, 1, 1)


def test_list_scopes_returns_immediate_children(respx_mock: respx.MockRouter) -> None:
    records = [
        MemoryRecord(id="1", content="c", scope="/agent/researcher/notes"),
        MemoryRecord(id="2", content="c", scope="/agent/writer"),
        MemoryRecord(id="3", content="c", scope="/agent/researcher/drafts"),
    ]
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": [_memory_item(r) for r in records]})
    )
    storage = _storage()

    scopes = storage.list_scopes("/agent")

    assert scopes == ["/agent/researcher", "/agent/writer"]


def test_list_categories_counts_occurrences(respx_mock: respx.MockRouter) -> None:
    records = [
        MemoryRecord(id="1", content="c", scope="/", categories=["fact", "urgent"]),
        MemoryRecord(id="2", content="c", scope="/", categories=["fact"]),
    ]
    respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": [_memory_item(r) for r in records]})
    )
    storage = _storage()

    counts = storage.list_categories()

    assert counts == {"fact": 2, "urgent": 1}


# -- namespace scoping ---------------------------------------------------------


def test_namespace_resolver_routes_save_to_derived_namespace(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "il-1"})
    )
    storage = _storage(
        namespace_resolver=lambda scope: f"crew-{(scope or 'shared').strip('/').replace('/', '-')}"
    )
    record = MemoryRecord(content="c", scope="/agent/researcher")

    storage.save([record])

    body = json.loads(route.calls[0].request.content)
    assert body["namespace"] == "crew-agent-researcher"


def test_get_record_uses_static_namespace_even_with_resolver(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{BASE_URL}/v1/memories").mock(
        return_value=httpx.Response(200, json={"memories": []})
    )
    storage = _storage(namespace="static-ns", namespace_resolver=lambda scope: "resolved-ns")

    storage.get_record("whatever")

    assert route.calls[0].request.url.params.get("namespace") == "static-ns"
