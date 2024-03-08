#!/bin/bash

# This script will lint the .py files that differ from master. It is intended to be
# run against a PR in CI, linting just the files affected by the PR. It can also be
# run in a dev environment in order to replicate the linting behavior in CI.

PY_FILES="$(git diff --name-only --diff-filter=ACMRTUXB origin/master | grep -E '.py$')"

echo -e "\n============= Running pixeltable linting script."

if [ -z "$PY_FILES" ]; then
  echo -e "\n============= No files changed; nothing to do.\n"
  exit 0
fi

echo -e '\n============= The following files differ from `master` and will be linted:\n'

echo "$PY_FILES"

echo -e '\n============= Running `mypy` on changed files.\n'

mypy "$PY_FILES"

echo -e '\n============= Running `pylint` on changed files.\n'

pylint "$PY_FILES"

echo -e '\n============= Running `ruff check` on changed files.\n'

ruff check "$PY_FILES"

echo -e '\n============= Running `ruff format --check` on changed files.\n'

ruff format --check "$PY_FILES"
