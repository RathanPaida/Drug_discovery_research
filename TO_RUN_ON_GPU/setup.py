"""Shim for `pip install -e .` on older toolchains.

All metadata lives in pyproject.toml -- this file deliberately declares nothing, so there
is exactly one place where dependencies and the package list are defined. It used to
declare `name`/`version`/`find_packages()` with NO install_requires and no
python_requires, which meant `pip install .` produced an importable package with none of
its dependencies and no indication that anything was missing.
"""
from setuptools import setup

setup()
