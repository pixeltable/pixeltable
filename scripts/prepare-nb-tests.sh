#!/bin/bash -e

# Notebooks that are always skipped
SKIP_NOTEBOOKS=(
    working-with-gemini          # Temporary
    working-with-together        # Flaky
    working-with-twelvelabs      # [PXT-1040] Temporary (rate limiting issues)
    rag-operations               # Failing in CI for unknown reasons
    llm-tool-calling             # Flaky
    working-with-fabric          # Requires Microsoft Fabric environment
    working-with-fiftyone        # Voxel51 is currently omitted from our dev env for security reasons
    working-with-tigris          # Requires Tigris environment
    img-detection-vs-segmentation  # Segmentation models are crashing in CI (memory issue?)
)

# Notebooks that are skipped unless --include-expensive is passed
EXPENSIVE_NOTEBOOKS=(
    video-generate-ai            # Expensive
    img-image-to-image           # Expensive (downloads ~5GB model)
    working-with-bfl             # Expensive (paid API, insufficient credits in CI)
    working-with-runwayml        # Expensive (paid API)
)

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

DO_PIP_INSTALL=true
INCLUDE_EXPENSIVE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-pip)
            DO_PIP_INSTALL=false
            shift
            ;;
        --include-expensive)
            INCLUDE_EXPENSIVE=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-pip] [--include-expensive] target-path <notebook-paths>"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "$2" ]; then
    echo "Usage: $0 [--no-pip] [--include-expensive] target-path <notebook-paths>"
    exit 1
fi

TARGET_DIR="$1"
shift

echo "Target path: $TARGET_DIR"
echo "Notebook paths: $@"
if [[ $DO_PIP_INSTALL == false ]]; then
    echo "Skipping pip install commands in notebooks."
fi
if [[ $INCLUDE_EXPENSIVE == true ]]; then
    echo "Including expensive notebooks."
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
    rm "$TARGET_DIR/${nb}.ipynb"
done

# Remove expensive notebooks unless --include-expensive was passed
if [[ $INCLUDE_EXPENSIVE == false ]]; then
    for nb in "${EXPENSIVE_NOTEBOOKS[@]}"; do
        rm "$TARGET_DIR/${nb}.ipynb"
    done
fi

# Get a list of all API keys referenced in the notebooks
REF_API_KEYS=$(grep -hoE '[A-Z0-9_]*_(API|ACCESS)_(KEY|TOKEN)(_[A-Z0-9_]*)?' "$TARGET_DIR"/*.ipynb | sort | uniq)
echo
echo "Checking for API keys: $(echo "$REF_API_KEYS" | tr '\n' ' ')"
for env in $REF_API_KEYS; do
    if [ -z "${!env}" ]; then
        # The given API key is not available. Delete all notebooks that require it.
        for notebook in $(grep -l "$env" $TARGET_DIR/*.ipynb); do
            echo "Skipping $notebook because $env is not defined."
            rm "$notebook"
        done
    fi
done
