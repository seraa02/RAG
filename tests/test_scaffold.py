"""
Sanity tests for the project scaffold itself.

These do not test pipeline logic (none exists yet) -- they confirm the
directory structure and package layout are correct so future tests have a
working foundation to build on.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_expected_top_level_dirs_exist():
    for name in ["config", "data", "src", "eval", "tests"]:
        assert (PROJECT_ROOT / name).is_dir(), f"missing directory: {name}"


def test_expected_data_subdirs_exist():
    for name in ["raw", "processed", "cache"]:
        assert (PROJECT_ROOT / "data" / name).is_dir(), f"missing data subdir: {name}"


def test_src_package_is_importable():
    import src  # noqa: F401
    import src.config  # noqa: F401
    import src.schema  # noqa: F401


def test_ingest_and_retrieval_packages_are_importable():
    import src.ingest.sources.sec  # noqa: F401
    import src.ingest.normalize  # noqa: F401
    import src.retrieval.router  # noqa: F401


def test_eval_placeholders_exist():
    for name in ["questions.json", "golden_dataset.json", "annotations.json", "run_benchmark.py"]:
        assert (PROJECT_ROOT / "eval" / name).is_file(), f"missing eval file: {name}"
