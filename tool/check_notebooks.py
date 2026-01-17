"""
Check Notebooks Tool

Verifies that notebooks have sufficient output coverage (at least OUTPUT_PCT_THRESHOLD% of cells should have outputs).
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple


OUTPUT_PCT_THRESHOLD = 50.0  # Minimum percentage of code cells that must have outputs

class NotebookStats(NamedTuple):
    path: Path
    total_cells: int
    code_cells: int
    code_cells_with_output: int

    @property
    def has_output_pct(self) -> float:
        """Returns the percentage of code cells without outputs."""
        if self.code_cells == 0:
            return 100.0
        return (self.code_cells_with_output / self.code_cells) * 100.0


def check_notebook(notebook_path: Path) -> NotebookStats:
    """
    Check a single notebook's output coverage.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook.get('cells', [])
    total_cells = len(cells)

    # Only count code cells (not markdown cells)
    code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
    code_cell_count = len(code_cells)

    # Count code cells with non-empty outputs
    code_cells_with_output = sum(
        1 for cell in code_cells
        if cell.get('outputs') and len(cell.get('outputs')) > 0
    )

    return NotebookStats(
        path=notebook_path,
        total_cells=total_cells,
        code_cells=code_cell_count,
        code_cells_with_output=code_cells_with_output
    )


def main() -> int:
    """Main entry point."""

    directories: list[Path] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists() or not path.is_dir():
            print(f"Not a valid directory: {arg}", file=sys.stderr)
            return 1
        directories.append(path)

    if not directories:
        print("Usage: check_notebooks.py <directory> [<directory> ...]", file=sys.stderr)
        return 1

    notebooks = {f for dir in directories for f in dir.rglob('*.ipynb') if '.ipynb_checkpoints' not in f.parts}
    if not notebooks:
        print(f"No notebooks found.")
        return 1

    print(f"Running static checks on {len(notebooks)} notebook(s).")

    failed_notebooks: list[NotebookStats] = []

    for notebook_path in sorted(notebooks):
        stats = check_notebook(notebook_path)
        if stats.has_output_pct < OUTPUT_PCT_THRESHOLD:
            failed_notebooks.append(stats)

    # Summary
    if failed_notebooks:
        print(f"Notebook checks failed!")
        for stats in failed_notebooks:
            print(f"  FAILED: {stats.path}")
            print(f"    Too few code cells have output: {stats.has_output_pct:.1f}% < {OUTPUT_PCT_THRESHOLD:.1f}%")
        return 1
    else:
        print("No issues found.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
