#!/bin/bash -e

IFS=$'\n'
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."
TEST_PATH="target/nb-tests"

# Initialize conda in this subshell
eval "$(conda shell.bash hook)"

# Use a separate Pixeltable DB for these tests
export PIXELTABLE_HOME=~/.pixeltable
export PIXELTABLE_DB="isolatednbtests"

"$SCRIPT_DIR/prepare-nb-tests.sh" docs/notebooks
rm -f target/nb-tests/audio-transcriptions.ipynb  # temporary workaround

FAILURES=0

for nb in "$TEST_PATH"/*.ipynb; do
    echo "Testing notebook: $nb"
    echo "Creating conda environment ..."
    conda create -y --name nb-test-env python=3.9
    echo "Activating conda environment ..."
    conda activate nb-test-env
    conda info
    echo "Installing pytest ..."
    pip install -qU pip
    pip install -q pytest nbmake
    echo "Running notebook $nb ..."
    pytest -v -m '' --nbmake --nbmake-timeout=1800 "$nb" || ((FAILURES++))
    echo "Cleaning $PIXELTABLE_DB postgres DB ..."
    POSTGRES_BIN_PATH=$(python -c 'import pixeltable_pgserver; import sys; sys.stdout.write(str(pixeltable_pgserver._commands.POSTGRES_BIN_PATH))')
    PIXELTABLE_URL="postgresql://postgres:@/postgres?host=$PIXELTABLE_HOME/pgdata"
    "$POSTGRES_BIN_PATH/psql" "$PIXELTABLE_URL" -U postgres -c "DROP DATABASE $PIXELTABLE_DB;"
    echo "Deactivating conda environment ..."
    conda deactivate
    echo "Done!"
done

if [[ "$FAILURES" > 0 ]]; then
    echo "There were $FAILURES failed notebook(s)."
    exit 1
else
    echo "All notebooks succeeded."
fi
