#!/bin/bash -e

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."
TEST_PATH="target/nb-tests"
mkdir -p "$TEST_PATH"

DO_PIP_INSTALL=true
if [ -n "$1" ]; then
    if [ "$1" == "--no-pip" ]; then
        echo "Skipping pip install commands in notebooks."
        DO_PIP_INSTALL=false
    else
        echo "Usage: $0 [--no-pip]"
        exit 1
    fi
fi

# Copy notebooks to target directory
echo "Copying notebooks to test folder ..."
for notebook in $(find docs/release tests -name '*.ipynb' | grep -v .ipynb_checkpoints); do
    if [[ $DO_PIP_INSTALL == true ]]; then
        # Just copy the notebook to the test directory
        cp "$notebook" "$TEST_PATH"
    else
        # Scrub the %pip install lines from the notebook
        sed -E 's/%pip install [^"]*//' "$notebook" > "$TEST_PATH/$(basename "$notebook")"
    fi
done

for env in {ANTHROPIC_API_KEY,FIREWORKS_API_KEY,LABEL_STUDIO_API_KEY,MISTRAL_API_KEY,OPENAI_API_KEY,REPLICATE_API_TOKEN,TOGETHER_API_KEY}; do
    if [ -z "${!env}" ]; then
        # The given API key is not available. Delete all notebooks that require it.
        for notebook in $(grep -l "$env" $TEST_PATH/*.ipynb); do
            echo "Skipping $notebook because $env is not defined."
            rm "$notebook"
        done
    fi
done
