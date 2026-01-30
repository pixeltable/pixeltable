#!/bin/bash +e

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells docs/release
nbqa "ruff format --line-length=74" --nbqa-dont-skip-bad-cells docs/release
# nbqa "ruff check --select I --fix --line-length=74" --nbqa-dont-skip-bad-cells docs/release
