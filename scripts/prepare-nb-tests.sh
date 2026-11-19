#!/bin/bash -e

# Copies the notebooks to the target path for nbmake. The notebooks' test tiers, and `skip` for the ones CI never runs,
# are in NOTEBOOK_MARKERS in tests/notebook_markers.py.

# Notebooks that need a GPU or a different pip setup
SKIP_NOTEBOOKS=()

# Check if `nvidia-smi` returns success; if not, skip GPU notebooks too
if ! nvidia-smi > /dev/null 2>&1; then
    echo "nvidia-smi not found or not working; skipping GPU notebooks."
    SKIP_NOTEBOOKS+=(
        working-with-vllm           # vLLM requires a CUDA environment
    )
fi

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

DO_PIP_INSTALL=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-pip)
            DO_PIP_INSTALL=false
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-pip] target-path <notebook-paths>"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "$2" ]; then
    echo "Usage: $0 [--no-pip] target-path <notebook-paths>"
    exit 1
fi

TARGET_DIR="$1"
shift

if [[ $DO_PIP_INSTALL == true ]]; then
    # TODO(PXT-1427): the instrumentation package is available in the source checkout, but is not yet published
    # on PyPI.
    SKIP_NOTEBOOKS+=(observability)
else
    # whisperx is not in the dev env; this notebook gets it only from its %pip cell
    SKIP_NOTEBOOKS+=(audio-podcast-chapters)
fi

echo "Target path: $TARGET_DIR"
echo "Notebook paths: $@"
if [[ $DO_PIP_INSTALL == false ]]; then
    echo "Skipping pip install commands in notebooks."
fi

mkdir -p "$TARGET_DIR"

# Copy notebooks to target directory
echo
echo "Copying notebooks to test folder ..."
for notebook in $(find "$@" -name '*.ipynb' | grep -v .ipynb_checkpoints); do
    target="$TARGET_DIR/$(basename "$notebook")"
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

# Remove skipped notebooks
for nb in "${SKIP_NOTEBOOKS[@]}"; do
    echo "Skipping $TARGET_DIR/${nb}.ipynb because it is in SKIP_NOTEBOOKS."
    rm "$TARGET_DIR/${nb}.ipynb"
done

# Get a list of all API keys referenced in the notebooks
REF_API_KEYS=$(grep -hoE '[A-Z0-9_]*_(API|ACCESS)_(KEY|TOKEN|SECRET)(_[A-Z0-9_]*)?|HF_TOKEN' "$TARGET_DIR"/*.ipynb | sort | uniq)
echo
echo "Checking for API keys: $(echo "$REF_API_KEYS" | tr '\n' ' ')"
for env in $REF_API_KEYS; do
    if [ -z "${!env}" ]; then
        # The given API key is not available. Delete all notebooks that require it.
        for nb in $(grep -l "$env" $TARGET_DIR/*.ipynb); do
            echo "Skipping $nb because $env is not defined."
            rm "$nb"
        done
    fi
done
