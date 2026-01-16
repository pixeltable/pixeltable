#!/bin/bash +e

if [ $# -eq 0 ]; then
    echo "Error: No command provided"
    echo "Usage: $(basename "$0") {check|format}"
    exit 1
fi

if [ "$1" == "check" ]; then
    echo "Running check..."
    # Find all notebooks and diff using `nbqa mdformat`.
    find ./docs/ -type f -name "*.ipynb" | grep -v .ipynb_checkpoints | xargs nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells --nbqa-diff >> /dev/null
    if [ $? == 0 ]; then
        echo "Notebooks are properly formatted."
    else
        echo 'Notebooks are not properly formatted. Please run `make format` to format them.'
        exit 1
    fi

elif [ "$1" == "format" ]; then
    find ./docs/ -type f -name "*.ipynb" | grep -v .ipynb_checkpoints | xargs nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells

else
    echo "Error: Invalid command '$1'"
    echo "Usage: $0 {check|format}"
    exit 1

fi
