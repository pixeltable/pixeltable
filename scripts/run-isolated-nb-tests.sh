#!/bin/bash -e

IFS=$'\n'
SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(dirname "$0")"

CLEAN_CACHES=0
if [ "$1" == "--clean-caches" ]; then
    CLEAN_CACHES=1
    shift
fi

if [ -z "$2" ]; then
    echo "Usage: $SCRIPT_NAME [--clean-caches] <python-version> <notebook-path> ..."
    echo ""
    echo "Notebook paths can be .ipynb files or directories containing .ipynb files, which will be collected recursively."
    echo "--clean-caches: If specified, will clean Hugging Face caches between each run."
    exit 1
fi

PY_VERSION="$1"
shift

# For isolated NB tests, use a directory that is *not* controlled by conftest.py
# (The whole point is to run independently of our dev environment.)
TEST_PATH="$SCRIPT_DIR/../target/nb-tests"

rm -rf "$TEST_PATH"
"$SCRIPT_DIR/prepare-nb-tests.sh" "$TEST_PATH" "$@"
rm -f "$TEST_PATH"/audio-transcriptions.ipynb  # temporary workaround

cd "$SCRIPT_DIR/.."

# Initialize conda in this subshell
eval "$(conda shell.bash hook)"

# Use a separate Pixeltable DB for these tests
export PIXELTABLE_HOME=~/.pixeltable
export PIXELTABLE_DB="isolatednbtests"
NB_CONDA_ENV=nb-test-env
FAILURES=0

for nb in "$TEST_PATH"/*.ipynb; do
    echo "Testing notebook: $nb"

    echo "Creating conda environment $NB_CONDA_ENV ..."
    conda create -q -y --name "$NB_CONDA_ENV" python="$PY_VERSION"

    echo "Activating conda environment ..."
    conda activate "$NB_CONDA_ENV"
    conda info

    echo "Installing ffmpeg ..."
    conda install -q -y -c conda-forge libiconv 'ffmpeg==6.1.1=gpl*'

    echo "Installing pytest ..."
    pip install -qU pip
    pip install -q pytest nbmake

    ATTEMPT=0
    while [[ $ATTEMPT < 3 ]]; do
        (( ATTEMPT++ ))
        echo "Running notebook (attempt $ATTEMPT): $nb"
        pytest -v -m '' --nbmake --nbmake-timeout=1800 "$nb"
        RESULT=$?
        if [ $RESULT == 0 ]; then
            break
        fi
    done
    if [ $RESULT == 0 ]; then
        echo "Notebook succeeded: $nb"
    else
        echo "Notebook failed: $nb"
        (( FAILURES++ ))
    fi

    echo "Cleaning $PIXELTABLE_DB postgres DB ..."
    POSTGRES_BIN_PATH=$(python -c 'import pixeltable_pgserver; import sys; sys.stdout.write(str(pixeltable_pgserver._commands.POSTGRES_BIN_PATH))')
    PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
    "$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "DROP DATABASE IF EXISTS $PIXELTABLE_DB;"

    if [ $CLEAN_CACHES ]; then
        echo "Cleaning Hugging Face cache ..."
        rm -rf ~/.cache/huggingface
    fi

    echo "Deactivating conda environment ..."
    conda deactivate

    echo "Removing conda environment ..."
    conda remove -y --name "$NB_CONDA_ENV" --all

    echo "Done!"
done

if [[ $FAILURES > 0 ]]; then
    echo "There were $FAILURES failed notebook(s)."
    exit 1
else
    echo "All notebooks succeeded."
fi
