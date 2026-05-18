"""Common pytest fixtures for ms2-met tests."""
import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def sample_pfind_file():
    return os.path.join(FIXTURES_DIR, "sample_pfind.qry.res")


@pytest.fixture
def sample_pfind_dir():
    return os.path.join(FIXTURES_DIR, "sample_pfind_dir")
