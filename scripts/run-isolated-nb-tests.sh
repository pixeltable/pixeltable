#!/bin/bash -e

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."
PY_VERSION="$1"
TEST_PATH="tests/target/nb-tests"

if [ -z "$PY_VERSION" ]; then
    echo "Usage: run-isolated-nb-tests.sh <python-version>"
    exit 1
fi

# Initialize conda in this subshell
eval "$(conda shell.bash hook)"

# Use a separate Pixeltable DB for these tests
export PIXELTABLE_HOME=~/.pixeltable
export PIXELTABLE_DB="isolatednbtests"

"$SCRIPT_DIR/prepare-nb-tests.sh" docs/notebooks
rm -f "$TEST_PATH"/audio-transcriptions.ipynb  # temporary workaround

NB_CONDA_ENV=nb-test-env
FAILURES=0

for nb in "$TEST_PATH"/*.ipynb; do
    echo "Testing notebook: $nb"

    echo "Creating conda environment $NB_CONDA_ENV ..."
    conda create -y --name "$NB_CONDA_ENV" python="$PY_VERSION"

    echo "Activating conda environment ..."
    conda activate "$NB_CONDA_ENV"
    conda info

    echo "Installing pytest ..."
    pip install -qU pip
    pip install -q pytest nbmake

    echo "Running notebook $nb ..."
    pytest -v -m '' --nbmake --nbmake-timeout=1800 "$nb" || (( FAILURES++ )) || true

    echo "Cleaning $PIXELTABLE_DB postgres DB ..."
    POSTGRES_BIN_PATH=$(python -c 'import pixeltable_pgserver; import sys; sys.stdout.write(str(pixeltable_pgserver._commands.POSTGRES_BIN_PATH))')
    PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
    "$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "DROP DATABASE IF EXISTS $PIXELTABLE_DB;"

    echo "Cleaning Hugging Face cache ..."
    rm -rf ~/.cache/huggingface

    echo "Deactivating conda environment ..."
    conda deactivate

    echo "Removing conda environment ..."
    conda remove -y --name "$NB_CONDA_ENV" --all

    echo "Done!"
done

if [[ "$FAILURES" > 0 ]]; then
    echo "There were $FAILURES failed notebook(s)."
    exit 1
else
    echo "All notebooks succeeded."
fi
