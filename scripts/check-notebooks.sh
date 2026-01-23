#!/bin/bash +e

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

echo "Running notebook checks ..."

# Find all notebooks and diff using `nbqa mdformat`.
echo "Checking markdown formatting ..."
nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells --nbqa-diff docs/release >> /dev/null
if [ $? == 0 ]; then
    echo "Notebooks are properly formatted."
else
    echo 'Notebooks are not properly formatted. Please run `make format` to format them.'
    exit 1
fi

echo "Checking code formatting ..."
nbqa "ruff format --check --line-length=74" --nbqa-dont-skip-bad-cells docs/release || exit 1
nbqa "ruff check --select I --line-length=74" --nbqa-dont-skip-bad-cells docs/release || exit 1

# Run custom notebook checks
echo "Running custom notebook checks ..."
python tool/check_notebooks.py docs/release
if [ $? != 0 ]; then
    # TODO: Change this to `exit 1` once notebook issues are fixed.
    echo "There were errors, but currently they will be ignored."
fi
