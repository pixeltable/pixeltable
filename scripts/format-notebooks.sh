#!/bin/bash +e

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

nbqa mdformat --nbqa-md --nbqa-dont-skip-bad-cells docs/release
