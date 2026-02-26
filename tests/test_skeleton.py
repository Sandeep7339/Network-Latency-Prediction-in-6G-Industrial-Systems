"""Verify that the expected project folder structure exists."""

import os
from pathlib import Path

# Project root is two levels up from this test file
ROOT = Path(__file__).resolve().parent.parent


def test_top_level_dirs_exist():
    """All required top-level directories must be present."""
    expected = ["data", "scripts", "notebooks", "src", "models", "figures", "reports", "tests"]
    for d in expected:
        assert (ROOT / d).is_dir(), f"Missing top-level directory: {d}"


def test_src_subpackages_exist():
    """All required src sub-packages must be present."""
    subpackages = ["data", "features", "models", "eval", "predict"]
    for pkg in subpackages:
        pkg_dir = ROOT / "src" / pkg
        assert pkg_dir.is_dir(), f"Missing src subpackage directory: src/{pkg}"
        assert (pkg_dir / "__init__.py").is_file(), f"Missing __init__.py in src/{pkg}"


def test_src_init_exists():
    """src/ itself must be a Python package."""
    assert (ROOT / "src" / "__init__.py").is_file(), "Missing src/__init__.py"


def test_requirements_txt_exists():
    """requirements.txt must be present at project root."""
    assert (ROOT / "requirements.txt").is_file(), "Missing requirements.txt"


def test_readme_exists():
    """README.md must be present at project root."""
    assert (ROOT / "README.md").is_file(), "Missing README.md"


def test_data_directory_has_csv_files():
    """data/ directory should contain at least one CSV file."""
    csv_files = list((ROOT / "data").glob("*.csv"))
    assert len(csv_files) > 0, "No CSV files found in data/"
