#!/bin/bash

PY_FILES="$(git diff --name-only --diff-filter=ACMRTUXB origin/master | grep -E '.py$')"

if [ -z "$PY_FILES" ]; then
  exit 0
fi

ruff format $PY_FILES
