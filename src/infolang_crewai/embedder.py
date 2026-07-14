"""InfoLangEmbedder: carries raw query text through CrewAI's embedding interface.

CrewAI's ``Memory`` embeds every ``remember()``/``recall()`` string with
``self.embedder`` *before* calling the storage backend -- ``StorageBackend``
methods only ever see a ``list[float]``, never the original string (see
``crewai.memory.storage.backend.StorageBackend.search``). InfoLang embeds
server-side from raw text, so an :class:`~infolang_crewai.storage.InfoLangStorage`
needs the original string back to call InfoLang's ``/v1/recall``.

``InfoLangEmbedder`` bridges the two: instead of computing a real embedding,
it reversibly encodes each input string as a vector of UTF-8 byte values
scaled to ``[0, 1]``. ``InfoLangStorage`` decodes that vector back into the
original text before calling InfoLang. This is not a real embedding -- pair
it only with ``InfoLangStorage``, and always pass both together::

    memory = Memory(
        storage=InfoLangStorage(...),
        embedder=InfoLangEmbedder(),
    )

Mixing ``InfoLangEmbedder`` with any other storage backend, or
``InfoLangStorage`` with any other embedder, will not raise at import time --
but ``InfoLangStorage.search()`` validates every incoming vector and raises
``InfoLangCrewAIConfigError`` with a clear message the first time it sees one
that was not produced here, rather than silently returning wrong or empty
results.
"""

from __future__ import annotations

from .errors import InfoLangCrewAIConfigError


def encode_text(text: str) -> list[float]:
    """Reversibly encode ``text`` as a vector of UTF-8 byte values in ``[0, 1]``."""

    return [b / 255.0 for b in text.encode("utf-8")]


def decode_embedding(embedding: list[float]) -> str:
    """Decode a vector produced by :func:`encode_text` back into text.

    Raises:
        InfoLangCrewAIConfigError: if ``embedding`` does not look like output
            from :func:`encode_text` -- e.g. a real embedding model's vector
            was wired in instead of ``InfoLangEmbedder``. This is a
            best-effort sanity check (real embeddings essentially never land
            exactly on a ``k/255`` grid), not a cryptographic guarantee.
    """

    if not embedding:
        return ""
    raw_bytes = bytearray()
    for value in embedding:
        scaled = value * 255.0
        rounded = round(scaled)
        if not (0.0 <= value <= 1.0) or abs(scaled - rounded) > 1e-6 or not (0 <= rounded <= 255):
            raise InfoLangCrewAIConfigError(
                "InfoLangStorage received an embedding that InfoLangEmbedder did not "
                "produce. Pass embedder=InfoLangEmbedder() to Memory(...) alongside "
                "storage=InfoLangStorage(...) -- InfoLang embeds server-side from raw "
                "text, so a real embedding model's output cannot be decoded here."
            )
        raw_bytes.append(rounded)
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InfoLangCrewAIConfigError(
            "InfoLangStorage could not decode the query embedding back into text. "
            "Pass embedder=InfoLangEmbedder() to Memory(...) alongside "
            "storage=InfoLangStorage(...)."
        ) from exc


class InfoLangEmbedder:
    """Callable embedder for ``Memory(embedder=...)`` that pairs with InfoLangStorage.

    CrewAI calls embedders as ``embedder(list[str]) -> list[vector]``
    (see ``crewai.memory.types.embed_text``/``embed_texts``); this class
    matches that calling convention.
    """

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise InfoLangCrewAIConfigError(
                f"InfoLangEmbedder expects a list[str], got {type(texts).__name__}."
            )
        return [encode_text(t) for t in texts]
