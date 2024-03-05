#!/bin/bash

PYLINT_ARGS="--max-line-length 120"

FLAKE8_ARGS="--max-line-length 120"

# Disable various Pylint messages that seem pointless or annoying.
# If Pylint is bugging you about something that seems like it should be excluded,
# propose it as a new exclusion by adding it to this list as part of the PR.
# C0114: Missing module docstring (missing-module-docstring)
# C0116: Missing function or method docstring (missing-function-docstring)
# E1121: Too many positional arguments for method call (too-many-function-args)
# R0401: Cyclic import
# R0801: Similar lines in 2 files
# R0902: Too many instance attributes
# R0913: Too many arguments
# R0914: Too many local variables
# W0511: TODO
DISABLED_PYLINT_MESSAGES="C0114,C0116,E1121,R0401,R0801,R0902,R0913,R0914,W0511"

# This will lint ONLY the files that differ from master. If run against a PR as
# part of CI, it will lint just the files affected by the PR.
PY_FILES="$(git diff --name-only --diff-filter=ACMRTUXB origin/master | grep -E '.py$')"

SCRIPT_DIR="$(dirname "$0")"

echo -e "\n============= Running pixeltable linting script."

if [ -z "$PY_FILES" ]; then
  echo -e "\n============= No files changed; nothing to do.\n"
  exit 0
fi

echo -e '\n============= The following files differ from `master` and will be linted:\n'
echo "$PY_FILES"

echo -e "\n============= Running pylint on changed files.\n"

pylint $PYLINT_ARGS --disable "$DISABLED_PYLINT_MESSAGES" $PY_FILES

echo -e "\n============= Running flake8 on changed files.\n"

flake8 $FLAKE8_ARGS $PY_FILES

echo -e "\n============= Running black --check.\n"

"$SCRIPT_DIR"/black-changed-files.sh --check
