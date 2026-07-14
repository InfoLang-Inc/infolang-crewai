"""Minimal runnable Crew wired to InfoLangStorage.

Requires:
    pip install infolang-crewai crewai
    export INFOLANG_API_KEY=il_live_...   # https://infolang.ai console -> Setup
    export OPENAI_API_KEY=sk-...          # CrewAI's default agent LLM

Run:
    python examples/minimal_crew.py

This crew has one agent that researches a topic and is asked to remember
what it learns; a second kickoff() call demonstrates that a fresh crew
sharing the same InfoLang namespace recalls what the first one stored.
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, Task
from crewai.memory.unified_memory import Memory

from infolang_crewai import InfoLangEmbedder, InfoLangStorage


def build_crew(namespace: str) -> Crew:
    memory = Memory(
        storage=InfoLangStorage(
            api_key=os.environ["INFOLANG_API_KEY"],
            namespace=namespace,
        ),
        embedder=InfoLangEmbedder(),
    )

    researcher = Agent(
        role="Research Analyst",
        goal="Research topics thoroughly and remember useful facts for later",
        backstory=(
            "You are a meticulous analyst who always checks memory for prior "
            "findings before starting new research, and saves anything worth "
            "remembering."
        ),
        memory=True,
    )

    task = Task(
        description=(
            "Research the following topic and remember the single most useful "
            "fact you learn: {topic}"
        ),
        expected_output="A short summary of what you found and what you remembered.",
        agent=researcher,
    )

    return Crew(agents=[researcher], tasks=[task], memory=memory, verbose=True)


def main() -> None:
    namespace = "infolang-crewai-example"

    first_crew = build_crew(namespace)
    result = first_crew.kickoff(inputs={"topic": "the benefits of semantic memory for AI agents"})
    print("--- First run ---")
    print(result)

    # A second, independent Crew sharing the same InfoLang namespace can
    # recall what the first crew stored.
    second_crew = build_crew(namespace)
    result = second_crew.kickoff(
        inputs={"topic": "what did we already learn about AI agent memory?"}
    )
    print("--- Second run (should reference remembered facts) ---")
    print(result)


if __name__ == "__main__":
    main()
