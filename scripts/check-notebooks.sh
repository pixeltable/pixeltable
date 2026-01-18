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

# Run custom notebook checks
echo "Running custom notebook checks ..."
python tool/check_notebooks.py docs/release
if [ $? != 0 ]; then
    exit 1
fi
