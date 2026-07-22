#!/bin/bash -e

# Notebooks that are always skipped
SKIP_NOTEBOOKS=(
    audio-podcast-chapters          # Uses whisperx, which is disabled in our dev env (hard-pins torch 2.8)
    llm-tool-calling                # Relies on the user separately running an MCP server
    working-with-bfl                # [PXT-1111] Out of credits
    working-with-fabric             # [PXT-1113] Requires Microsoft Fabric environment
    working-with-fiftyone           # [PXT-1117] Voxel51 is currently omitted from our dev env for security reasons
    working-with-replicate          # [PXT-1248] Out of credits
    working-with-tigris             # [PXT-1122] Hard-codes getpass() calls for credentials and bucket
    working-with-runwayml           # [PXT-1120] RunwayML integration is very broken
    working-with-twelvelabs         # [PXT-1119] Exceeds rate limit
)

# Check if `nvidia-smi` returns success; if not, skip GPU notebooks too
if ! nvidia-smi > /dev/null 2>&1; then
    echo "nvidia-smi not found or not working; skipping GPU notebooks."
    SKIP_NOTEBOOKS+=(
        working-with-vllm           # vLLM requires a CUDA environment
    )
fi

# Notebooks that are skipped unless --include-very-expensive is passed
VERY_EXPENSIVE_NOTEBOOKS=(
    img-detection-vs-segmentation   # Resource intensive
    video-generate-ai               # High dollar cost
    working-with-gemini             # High dollar cost
    working-with-fal                # [PXT-1233] fal.ai integration failing on CI
    working-with-reve
    working-with-together           # Poor reliability
)

# Notebooks that are skipped unless --include-expensive is passed: all notebooks that use HF models.
EXPENSIVE_NOTEBOOKS=(
    audio-transcriptions
    computed-columns
    data-import-huggingface
    doc-chunk-for-rag
    embedding-indexes
    img-image-to-image
    img-promptable-segmentation
    multimodal_backend
    queries-and-expressions
    rag-demo
    rag-operations
    search-semantic-text
    search-similar-images
    ten-minute-tour
    video-image-slideshow
    working-with-hugging-face
    working-with-llama-cpp
    working-with-ollama             # [PXT-1246] Unreliable
)

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

DO_PIP_INSTALL=true
INCLUDE_VERY_EXPENSIVE=false
INCLUDE_EXPENSIVE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-pip)
            DO_PIP_INSTALL=false
            shift
            ;;
        --include-very-expensive)
            INCLUDE_VERY_EXPENSIVE=true
            shift
            ;;
        --include-expensive)
            INCLUDE_EXPENSIVE=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-pip] [--include-very-expensive] [--include-expensive] target-path <notebook-paths>"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ -z "$2" ]; then
    echo "Usage: $0 [--no-pip] [--include-very-expensive] [--include-expensive] target-path <notebook-paths>"
    exit 1
fi

TARGET_DIR="$1"
shift

echo "Target path: $TARGET_DIR"
echo "Notebook paths: $@"
if [[ $DO_PIP_INSTALL == false ]]; then
    echo "Skipping pip install commands in notebooks."
fi
if [[ $INCLUDE_VERY_EXPENSIVE == true ]]; then
    echo "Including very expensive notebooks."
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
    echo "Skipping $TARGET_DIR/${nb}.ipynb because it is in SKIP_NOTEBOOKS."
    rm "$TARGET_DIR/${nb}.ipynb"
done

# Remove very expensive notebooks unless --include-very-expensive was passed
if [[ $INCLUDE_VERY_EXPENSIVE == false ]]; then
    for nb in "${VERY_EXPENSIVE_NOTEBOOKS[@]}"; do
        echo "Skipping $TARGET_DIR/${nb}.ipynb because it is in VERY_EXPENSIVE_NOTEBOOKS."
        rm "$TARGET_DIR/${nb}.ipynb"
    done
fi

# Remove expensive notebooks unless --include-expensive was passed
if [[ $INCLUDE_EXPENSIVE == false ]]; then
    for nb in "${EXPENSIVE_NOTEBOOKS[@]}"; do
        echo "Skipping $TARGET_DIR/${nb}.ipynb because it is in EXPENSIVE_NOTEBOOKS."
        rm "$TARGET_DIR/${nb}.ipynb"
    done
fi

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
