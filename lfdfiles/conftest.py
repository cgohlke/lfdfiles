"""Pytest configuration."""

__all__ = []

import pathlib

import numpy
import pytest

collect_ignore = ['__init__.py']

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'tests' / 'data'
TEMP = HERE.parent / 'tests' / '_temp'
# TEMP.mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def set_printoptions() -> None:
    """Adjust numpy array print options for use with `# doctest: +NUMBER`."""
    numpy.set_printoptions(
        # precision=3,
        threshold=5,
        formatter={'float': lambda x: f'{x:.4g}'},  # remove whitespace
    )


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    doctest_namespace['DATA'] = DATA
    doctest_namespace['TEMP'] = TEMP


# mypy: ignore-errors
