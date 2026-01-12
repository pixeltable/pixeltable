#!/bin/bash +e

if [ $# -eq 0 ]; then
    echo "Error: No command provided"
    echo "Usage: $0 {check|format}"
    exit 1
fi

if [ "$1" = "check" ]; then
    echo "Running check..."
    # Find all notebooks, send them to nbqa mdformat to make a diff. Fail if the output has more than 3 lines
    # (otherwise that's just some nice message that nbqa prints when the diff is empty)
    echo "find..."
    find . -type f -name "*.ipynb" | grep -v .ipynb_checkpoints
    echo $?
    echo "nbqa..."
    nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells --nbqa-diff ./docs/release/howto/cookbooks/data/data-sampling.ipynb
    echo $?
    echo "awk..."
    echo 'hello' | awk 'END {exit 0}'
    echo $?
    echo "find + grep + nbqa..."
    find . -type f -name "*.ipynb" | grep -v .ipynb_checkpoints | xargs nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells --nbqa-diff 
    echo $?
    echo "putting it all together..."
    find . -type f -name "*.ipynb" | grep -v .ipynb_checkpoints | xargs nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells --nbqa-diff | awk 'END {exit (NR > 3 ? 1 : 0)}'
    echo $?

elif [ "$1" = "format" ]; then
    echo "Running format..."
    find . -type f -name "*.ipynb" | grep -v .ipynb_checkpoints | xargs nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells

else
    echo "Error: Invalid command '$1'"
    echo "Usage: $0 {check|format}"
    exit 1
fi
