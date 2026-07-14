"""Typed errors for infolang-crewai.

Mem0's CrewAI integration has several open, reported breakages that all trace
back to the same root cause: a shape assumption made without validation.
- crewAIInc/crewAI#2591 -- ``KeyError`` on ``result["memory"]`` when the
  provider's response omits the key.
- crewAIInc/crewAI#2587 -- ``memory_config`` silently not applied; the
  integration falls back to defaults without telling the caller.
- crewAIInc/crewAI#3152 -- "Expected list but got str", a 400 from blindly
  forwarding a malformed payload shape.
- crewAIInc/crewAI#2448 -- ``AttributeError`` when switching providers
  (e.g. under Gemini), because code assumed an attribute only some
  providers set.

Every one of those classes is addressed in this package by validating
config and response shapes *before* they reach SDK/library internals, and
raising one of these typed errors with a message that says exactly what was
wrong -- instead of letting a raw ``KeyError``, ``AttributeError``, or
``pydantic.ValidationError`` escape from inside a ``save()``/``search()``
call.
"""

from __future__ import annotations


class InfoLangCrewAIError(Exception):
    """Base class for every error raised by infolang-crewai."""


class InfoLangCrewAIConfigError(InfoLangCrewAIError):
    """Storage misconfiguration: missing client, bad namespace, wrong embedder.

    Raised eagerly -- at construction time for constructor arguments, or on
    first use for cross-cutting mismatches (e.g. a non-InfoLangEmbedder
    vector reaching ``search()``) -- so misconfiguration never silently
    degrades into "memory looks empty" or "memory looks like it does
    nothing" (the class of bug in crewAIInc/crewAI#2587).
    """


class InfoLangCrewAIResponseError(InfoLangCrewAIError):
    """The InfoLang runtime returned a response shape this package cannot use.

    Wraps SDK errors (``InfoLangError``) and response-validation failures
    (``pydantic.ValidationError`` from a malformed hit/record shape) behind
    one exception type with a clear message, instead of letting those
    escape raw from inside a ``StorageBackend`` method
    (crewAIInc/crewAI#2591, #3152).
    """
