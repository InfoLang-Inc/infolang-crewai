"""Unicode and long-text inputs must round-trip exactly through both the
sidecar codec and the InfoLangEmbedder text<->vector channel."""

from __future__ import annotations

import httpx
import pytest
import respx
from crewai.memory.types import MemoryRecord

from infolang_crewai import InfoLangStorage, _codec
from infolang_crewai.embedder import decode_embedding, encode_text

BASE_URL = "https://mock.infolang.test"

UNICODE_SAMPLES = [
    "emoji party \U0001f389\U0001f9e0\U0001f680",
    "你好，世界",  # Chinese
    "こんにちは世界",  # Japanese
    "مرحبا بالعالم",  # Arabic (RTL)
    "Привет, мир!",  # Russian
    "café naïve résumé üñîçødé",
    "zero-width​joiner and ‌‍ marks",
]

LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 500  # ~23,000 chars


@pytest.mark.parametrize("text", UNICODE_SAMPLES)
def test_embedder_round_trips_unicode_exactly(text: str) -> None:
    assert decode_embedding(encode_text(text)) == text


def test_embedder_round_trips_long_text_exactly() -> None:
    assert decode_embedding(encode_text(LONG_TEXT)) == LONG_TEXT


def test_embedder_round_trips_empty_string() -> None:
    assert decode_embedding(encode_text("")) == ""


@pytest.mark.parametrize("text", UNICODE_SAMPLES)
def test_codec_round_trips_unicode_content(text: str) -> None:
    record = MemoryRecord(content=text, scope="/agent/researcher", categories=["notes"])
    decoded = _codec.decode(_codec.encode(record), fallback_id="fallback")
    assert decoded.content == text
    assert decoded.scope == "/agent/researcher"


def test_codec_round_trips_long_content() -> None:
    record = MemoryRecord(content=LONG_TEXT, scope="/")
    decoded = _codec.decode(_codec.encode(record), fallback_id="fallback")
    assert decoded.content == LONG_TEXT


def test_codec_round_trips_unicode_in_metadata_and_categories() -> None:
    record = MemoryRecord(
        content="short content",
        scope="/",
        categories=["日本語", "café"],
        metadata={"note": "ملاحظة \U0001f4dd"},
    )
    decoded = _codec.decode(_codec.encode(record), fallback_id="fallback")
    assert decoded.categories == record.categories
    assert decoded.metadata == record.metadata


def test_save_and_search_unicode_content_end_to_end(respx_mock: respx.MockRouter) -> None:
    text = "你好 emoji \U0001f680 café"
    record = MemoryRecord(id="rec-1", content=text, scope="/")
    stored_text = _codec.encode(record)

    remember_route = respx_mock.post(f"{BASE_URL}/v1/remember").mock(
        return_value=httpx.Response(200, json={"id": "il-1"})
    )
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200, json={"hits": [{"id": "il-1", "text": stored_text, "similarity": 0.95}]}
        )
    )

    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")
    storage.save([record])
    results = storage.search(encode_text(text), limit=5)

    assert remember_route.call_count == 1
    assert len(results) == 1
    assert results[0][0].content == text
    assert results[0][0].id == "rec-1"
