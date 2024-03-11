#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"

"$SCRIPT_DIR/analyze-changed-files.sh" pylint "ruff check" "ruff format --check"
