#!/bin/bash -e

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."
TEST_PATH="target/nb-tests"
mkdir -p "$TEST_PATH"

# Copy notebooks to target directory, replacing `%pip install` lines with empty strings
echo "Copying notebooks to test folder ..."
for notebook in $(find docs/release -name '*.ipynb' | grep -v .ipynb_checkpoints); do
    sed -E 's/%pip install [^"]*//' "$notebook" > "$TEST_PATH/$(basename "$notebook")"
done

for env in {OPENAI_API_KEY,TOGETHER_API_KEY,FIREWORKS_API_KEY,LABEL_STUDIO_API_KEY}; do
    if [ -z "${!env}" ]; then
        # The given API key is not available. Delete all notebooks that require it.
        for notebook in $(grep -l "$env" $TEST_PATH/*.ipynb); do
            echo "Skipping $notebook because $env is not defined."
            rm "$notebook"
        done
    fi
done
