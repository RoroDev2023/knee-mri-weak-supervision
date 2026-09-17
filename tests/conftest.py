"""Stub heavy notebook dependencies before any extracted cell is exec'd.

The notebooks import torch, transformers, pydicom, and langdetect; CI
machines do not have them and the exec'd code paths never call into them
(the GPU loops cannot run hermetically). Stubs are installed only when
the real package is absent, so a developer running the suite inside a
full environment still exercises the real modules where they exist.
"""

import sys
from pathlib import Path
from unittest import mock

import importlib.util
import pytest

TESTS_DIR = Path(__file__).resolve().parent

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# name -> extra submodule names that "from X.Y import Z" statements need
STUB_TARGETS = {
    "torch": [],
    "transformers": [],
    "pydicom": [],
    "tqdm": [],
    "langdetect": ["langdetect.lang_detect_exception"],
}


def _is_importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture(scope="session", autouse=True)
def stub_heavy_dependencies():
    for name, submodules in STUB_TARGETS.items():
        if _is_importable(name):
            continue
        sys.modules[name] = mock.MagicMock(name=f"stub:{name}")
        for submodule in submodules:
            sys.modules[submodule] = mock.MagicMock(name=f"stub:{submodule}")
    yield
