"""infolang-crewai: InfoLang-backed storage for CrewAI's unified Memory system.

Quickstart::

    from crewai import Crew
    from crewai.memory.unified_memory import Memory
    from infolang_crewai import InfoLangEmbedder, InfoLangStorage

    memory = Memory(
        storage=InfoLangStorage(api_key="il_live_...", namespace="my-crew"),
        embedder=InfoLangEmbedder(),
    )
    crew = Crew(agents=[...], tasks=[...], memory=memory)

See the README for namespace-scoping patterns (per-crew vs per-agent) and
known limitations of the hosted InfoLang runtime.
"""

from __future__ import annotations

from ._version import __version__
from .embedder import InfoLangEmbedder
from .errors import (
    InfoLangCrewAIConfigError,
    InfoLangCrewAIError,
    InfoLangCrewAIResponseError,
)
from .storage import InfoLangStorage

__all__ = [
    "__version__",
    "InfoLangStorage",
    "InfoLangEmbedder",
    "InfoLangCrewAIError",
    "InfoLangCrewAIConfigError",
    "InfoLangCrewAIResponseError",
]
