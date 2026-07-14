"""Config-not-applied failure class (crewAIInc/crewAI#2587 analog).

Every InfoLangStorage constructor argument is validated eagerly, so a typo
or conflicting config raises a clear error immediately instead of silently
falling back to defaults and only surfacing as "memory looks empty" later.
"""

from __future__ import annotations

import pytest
from infolang import InfoLang

from infolang_crewai import InfoLangCrewAIConfigError, InfoLangStorage

BASE_URL = "https://mock.infolang.test"


def test_empty_namespace_rejected() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="namespace"):
        InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="")


def test_whitespace_namespace_rejected() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="namespace"):
        InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace="   ")


def test_non_string_namespace_rejected() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="namespace"):
        InfoLangStorage(api_key="il_test", base_url=BASE_URL, namespace=123)  # type: ignore[arg-type]


def test_non_callable_namespace_resolver_rejected() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="namespace_resolver"):
        InfoLangStorage(
            api_key="il_test",
            base_url=BASE_URL,
            namespace_resolver="not-callable",  # type: ignore[arg-type]
        )


def test_client_and_api_key_conflict_rejected() -> None:
    client = InfoLang(api_key="il_test", base_url=BASE_URL)
    with pytest.raises(InfoLangCrewAIConfigError, match="not both"):
        InfoLangStorage(client=client, api_key="il_other")


def test_client_and_base_url_conflict_rejected() -> None:
    client = InfoLang(api_key="il_test", base_url=BASE_URL)
    with pytest.raises(InfoLangCrewAIConfigError, match="not both"):
        InfoLangStorage(client=client, base_url=BASE_URL)


def test_search_pool_must_be_positive() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="search_pool"):
        InfoLangStorage(api_key="il_test", base_url=BASE_URL, search_pool=0)


def test_list_page_must_be_positive() -> None:
    with pytest.raises(InfoLangCrewAIConfigError, match="list_page"):
        InfoLangStorage(api_key="il_test", base_url=BASE_URL, list_page=-1)


def test_missing_credentials_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFOLANG_API_KEY", raising=False)
    monkeypatch.delenv("INFOLANG_DEV_KEY", raising=False)
    with pytest.raises(InfoLangCrewAIConfigError, match="failed to construct client"):
        InfoLangStorage(base_url=BASE_URL)


def test_valid_construction_with_explicit_client_succeeds() -> None:
    client = InfoLang(api_key="il_test", base_url=BASE_URL)
    storage = InfoLangStorage(client=client, namespace="crew-a")
    assert storage is not None


def test_valid_construction_with_dev_key_succeeds() -> None:
    storage = InfoLangStorage(dev_key="devkey:my-namespace", namespace="crew-a")
    assert storage is not None


def test_namespace_resolver_returning_empty_string_raises_at_call_time(
    storage: InfoLangStorage,
) -> None:
    bad_storage = InfoLangStorage(
        api_key="il_test",
        base_url=BASE_URL,
        namespace_resolver=lambda scope: "",
    )
    with pytest.raises(InfoLangCrewAIConfigError, match="namespace_resolver"):
        bad_storage.list_records(scope_prefix="/agent/x")


def test_namespace_resolver_returning_non_string_raises_at_call_time() -> None:
    bad_storage = InfoLangStorage(
        api_key="il_test",
        base_url=BASE_URL,
        namespace_resolver=lambda scope: 42,  # type: ignore[return-value]
    )
    with pytest.raises(InfoLangCrewAIConfigError, match="namespace_resolver"):
        bad_storage.list_records(scope_prefix="/agent/x")
