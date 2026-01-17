#!/bin/bash +e

SCRIPT_DIR="$(dirname "$0")"

if [ $# -eq 0 ]; then
    echo "Error: No command provided"
    echo "Usage: $(basename "$0") {check|format}"
    exit 1
fi

cd "$SCRIPT_DIR/.."

if [ "$1" == "check" ]; then
    echo "Running notebook checks ..."
    # Find all notebooks and diff using `nbqa mdformat`.
    nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells --nbqa-diff docs/release >> /dev/null
    if [ $? == 0 ]; then
        echo "Notebooks are properly formatted."
    else
        echo 'Notebooks are not properly formatted. Please run `make format` to format them.'
        exit 1
    fi

    # Check that notebooks have sufficient output coverage (>=75%)
    python tool/check_notebooks.py docs/release
    if [ $? != 0 ]; then
        exit 1
    fi

elif [ "$1" == "format" ]; then
    nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells docs/release

else
    echo "Error: Invalid command: $1"
    echo "Usage: $(basename "$0") {check|format}"
    exit 1

fi
