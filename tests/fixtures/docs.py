import sys
from pathlib import Path
from typing import Generator

import pytest

# Package source root
docs_folder = Path(__file__).parents[2] / "docs"


@pytest.fixture
def add_docs_to_pythonpath() -> Generator[None, None, None]:
    """Testing"""
    # Add docs to python path
    sys.path.insert(0, str(docs_folder))
    yield None
