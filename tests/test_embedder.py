"""Unit tests for InfoLangEmbedder and the encode_text/decode_embedding codec."""

from __future__ import annotations

import pytest

from infolang_crewai import InfoLangCrewAIConfigError, InfoLangEmbedder
from infolang_crewai.embedder import decode_embedding, encode_text


def test_encode_decode_round_trip() -> None:
    assert decode_embedding(encode_text("hello world")) == "hello world"


def test_encode_produces_values_in_unit_range() -> None:
    vector = encode_text("hello")
    assert all(0.0 <= v <= 1.0 for v in vector)


def test_decode_empty_vector_is_empty_string() -> None:
    assert decode_embedding([]) == ""


def test_decode_rejects_out_of_range_values() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="InfoLangEmbedder"):
        decode_embedding([1.2, 0.5])


def test_decode_rejects_negative_values() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="InfoLangEmbedder"):
        decode_embedding([-0.1, 0.5])


def test_decode_rejects_off_grid_values() -> None:
    # Not close to any k/255 -- what a real embedding model would produce.
    with pytest.raises(InfoLangCrewAIConfigError, match="InfoLangEmbedder"):
        decode_embedding([0.334, 0.5])


def test_callable_interface_matches_crewai_embedder_contract() -> None:
    embedder = InfoLangEmbedder()
    vectors = embedder(["one", "two"])

    assert len(vectors) == 2
    assert decode_embedding(vectors[0]) == "one"
    assert decode_embedding(vectors[1]) == "two"


def test_callable_rejects_non_list_input() -> None:
    embedder = InfoLangEmbedder()
    with pytest.raises(InfoLangCrewAIConfigError):
        embedder("not-a-list")  # type: ignore[arg-type]


def test_callable_rejects_list_with_non_string_items() -> None:
    embedder = InfoLangEmbedder()
    with pytest.raises(InfoLangCrewAIConfigError):
        embedder(["ok", 123])  # type: ignore[list-item]
