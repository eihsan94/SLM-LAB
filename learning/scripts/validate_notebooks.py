"""Validate the structure and optionally execute the complete notebook course."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
REQUIRED_SECTIONS = (
    "Why this matters",
    "Learning objectives",
    "Failure modes",
    "Deliberate practice",
    "Exit ticket",
)


def load_notebooks() -> list[Path]:
    paths = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    if len(paths) != 40:
        raise AssertionError(f"expected 40 numbered notebooks, found {len(paths)}")
    for number, path in enumerate(paths):
        if not path.name.startswith(f"{number:02d}_"):
            raise AssertionError(f"expected lesson {number:02d}, found {path.name}")
    return paths


def validate_structure(path: Path) -> None:
    notebook: dict[str, Any] = json.loads(path.read_text())
    if notebook.get("nbformat") != 4:
        raise AssertionError(f"{path.name}: expected notebook format 4")

    cells = notebook.get("cells", [])
    if len(cells) < 8:
        raise AssertionError(
            f"{path.name}: expected a comprehensive lesson, found {len(cells)} cells"
        )

    cell_ids = [cell.get("id") for cell in cells]
    if None in cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise AssertionError(f"{path.name}: cell IDs must be present and unique")

    expected_number = int(path.name[:2])
    first_source = "".join(cells[0]["source"])
    if not first_source.startswith(f"# {expected_number:02d} —"):
        raise AssertionError(f"{path.name}: first heading does not match lesson number")

    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    for section in REQUIRED_SECTIONS:
        if section not in markdown:
            raise AssertionError(f"{path.name}: missing section {section!r}")

    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")


def validate_syllabus_links(paths: list[Path]) -> None:
    syllabus = (ROOT / "SYLLABUS.md").read_text()
    linked = set(re.findall(r"\((notebooks/[^)]+\.ipynb)\)", syllabus))
    expected = {path.relative_to(ROOT).as_posix() for path in paths}
    if linked != expected:
        missing = sorted(expected - linked)
        stale = sorted(linked - expected)
        raise AssertionError(
            f"syllabus link mismatch; missing={missing}, stale={stale}"
        )


def execute(path: Path) -> None:
    with path.open() as handle:
        notebook = nbformat.read(handle, as_version=4)
    executor = ExecutePreprocessor(timeout=180, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="also run every notebook in a fresh kernel without changing source files",
    )
    args = parser.parse_args()

    paths = load_notebooks()
    validate_syllabus_links(paths)
    execution_failures: list[tuple[Path, str, str]] = []
    for path in paths:
        validate_structure(path)
        if args.execute:
            try:
                execute(path)
            except (
                Exception
            ) as error:  # continue so one lesson cannot hide later failures
                error_type = type(error).__name__
                execution_failures.append((path, error_type, str(error)))
                print(f"FAIL {path.name}: {error_type}")
                continue
        print(f"PASS {path.name}")

    if execution_failures:
        print("\nExecution failure details:")
        for path, error_type, message in execution_failures:
            print(f"\n--- {path.name}: {error_type} ---\n{message}")
        raise SystemExit(
            f"{len(execution_failures)} notebook(s) failed clean-kernel execution"
        )

    mode = "structure and clean-kernel execution" if args.execute else "structure"
    print(f"Validated {len(paths)} notebooks: {mode} passed")


if __name__ == "__main__":
    main()
