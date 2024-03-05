#!/bin/bash

BLACK_OPTIONS="-t py38 -t py39 -t py310 -t py311 --line-length=120 --skip-string-normalization"

PY_FILES="$(git diff --name-only --diff-filter=ACMRTUXB origin/master | grep -E '.py$')"

if [ -z "$PY_FILES" ]; then
  exit 0
fi

black $BLACK_OPTIONS $PY_FILES "$@"
