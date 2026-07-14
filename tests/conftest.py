from __future__ import annotations

import pytest

from infolang_crewai import InfoLangStorage

BASE_URL = "https://mock.infolang.test"


@pytest.fixture
def storage() -> InfoLangStorage:
    return InfoLangStorage(api_key="il_test_key", base_url=BASE_URL, namespace="test-ns")
