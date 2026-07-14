"""Provider/auth-switching failure class (crewAIInc/crewAI#2448 analog).

Mem0's #2448 is an ``AttributeError`` that only surfaces under one provider
configuration (Gemini) because code elsewhere assumed an attribute that only
some providers set. The two places this package could have an equivalent
bug are (1) switching between InfoLang auth modes (api_key vs dev_key) and
(2) a caller wiring in a real embedding model instead of ``InfoLangEmbedder``
on the ``Memory`` side. Both are exercised here.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from infolang_crewai import InfoLangCrewAIConfigError, InfoLangStorage
from infolang_crewai.embedder import encode_text

BASE_URL = "https://mock.infolang.test"


def test_api_key_and_dev_key_storage_produce_identical_search_results(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200, json={"hits": [{"id": "m1", "text": "shared fact", "similarity": 0.9}]}
        )
    )

    api_key_storage = InfoLangStorage(api_key="il_live_x", base_url=BASE_URL, namespace="ns")
    dev_key_storage = InfoLangStorage(dev_key="dev:ns", base_url=BASE_URL, namespace="ns")

    api_key_results = api_key_storage.search(encode_text("query"), limit=5)
    dev_key_results = dev_key_storage.search(encode_text("query"), limit=5)

    assert len(api_key_results) == len(dev_key_results) == 1
    assert api_key_results[0][0].content == dev_key_results[0][0].content


def test_wrong_embedder_output_raises_config_error_not_attribute_error() -> None:
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")
    # A plausible real embedding: small floats, some negative, not on a
    # k/255 grid -- what you'd get if a real model's embedder were wired in
    # instead of InfoLangEmbedder.
    real_looking_embedding = [0.123, -0.456, 0.789, -0.001, 0.5001]

    with pytest.raises(InfoLangCrewAIConfigError, match="InfoLangEmbedder"):
        storage.search(real_looking_embedding, limit=5)


def test_wrong_embedder_output_with_out_of_range_values_raises_config_error() -> None:
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")
    out_of_range_embedding = [1.5, -2.0, 3.3]

    with pytest.raises(InfoLangCrewAIConfigError, match="InfoLangEmbedder"):
        storage.search(out_of_range_embedding, limit=5)


def test_embedder_switch_mid_session_is_isolated_per_call(respx_mock: respx.MockRouter) -> None:
    """Swapping which encoder produced the embedding, call to call, never leaks state."""
    respx_mock.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"hits": []})
    )
    storage = InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="ns")

    # First call: a well-formed InfoLangEmbedder vector.
    assert storage.search(encode_text("first query"), limit=5) == []

    # Second call: a malformed vector -- must raise cleanly and not corrupt
    # state used by a subsequent well-formed call.
    with pytest.raises(InfoLangCrewAIConfigError):
        storage.search([2.0, 3.0], limit=5)

    # Third call: back to a well-formed vector -- must still work.
    assert storage.search(encode_text("third query"), limit=5) == []
