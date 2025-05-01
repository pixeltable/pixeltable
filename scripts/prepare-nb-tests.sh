#!/bin/bash -e

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."
TEST_PATH="target/nb-tests"
mkdir -p "$TEST_PATH"

DO_PIP_INSTALL=true
if [ "$1" == "--no-pip" ]; then
    DO_PIP_INSTALL=false
    shift
fi

if [ -z "$1" ]; then
    echo "Usage: $0 [--no-pip] <notebook-paths>"
    exit 1
fi

echo "Notebook paths: $@"
if [[ $DO_PIP_INSTALL == false ]]; then
    echo "Skipping pip install commands in notebooks."
fi

# Copy notebooks to target directory
echo
echo "Copying notebooks to test folder ..."
for notebook in $(find "$@" -name '*.ipynb' | grep -v .ipynb_checkpoints); do
    target="$TEST_PATH/$(basename "$notebook")"
    echo "$notebook -> $target"
    if [[ $DO_PIP_INSTALL == true ]]; then
        # Just copy the notebook to the test directory
        cp "$notebook" "$target"
    else
        # 1. Scrub the %pip install lines from the notebook
        # 2. Replace any `execute-only-with-pip` tags with `skip-execution`
        sed -E 's/%pip install [^"]*//' "$notebook" | sed 's/execute-only-with-pip/skip-execution/g' > "$target"
    fi
done

# Remove together notebook (it's causing problems)
rm "$TEST_PATH/working-with-together.ipynb"

# Get a list of all API keys referenced in the notebooks
REF_API_KEYS=$(grep -hoE '[A-Z_]*_API_(KEY|TOKEN)' "$TEST_PATH"/*.ipynb | sort | uniq)
echo
echo "Checking for API keys: $(echo "$REF_API_KEYS" | tr '\n' ' ')"
for env in $REF_API_KEYS; do
    if [ -z "${!env}" ]; then
        # The given API key is not available. Delete all notebooks that require it.
        for notebook in $(grep -l "$env" $TEST_PATH/*.ipynb); do
            echo "Skipping $notebook because $env is not defined."
            rm "$notebook"
        done
    fi
done
