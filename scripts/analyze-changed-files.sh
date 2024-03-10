#!/bin/bash

# This script will analyze the .py files that differ from master. It is intended to be
# run against a PR in CI, analyzing just the files affected by the PR. It can also be
# run in a dev environment in order to replicate the behavior of CI.

if [ -z "$1" ]; then
  echo 'Usage: analyze-changed-files.sh <analyzers>'
  echo 'Example: analyze-changed-files.sh "ruff check" "ruff format --check"'
  exit 1
fi

PY_FILES="$(git diff --name-only --diff-filter=ACMRTUXB origin/master | grep -E '.py$')"

echo -e "\n============= Running pixeltable analyzer script against tools:\n"

for analyzer in "$@"; do
  echo "$analyzer"
done

if [ -z "$PY_FILES" ]; then
  echo -e "\n============= No files changed; nothing to do.\n"
  exit 0
fi

echo -e '\n============= The following files differ from `master` and will be analyzed:\n'

echo "$PY_FILES"

# This will ensure that filenames with spaces are handled properly if they appear in $PY_FILES.
IFS=$'\n'

for analyzer in "$@"; do

  echo -e "\n============= Running \`$analyzer\` on changed files.\n"
  # Convert spaces to \n's in analyzer command to accommodate IFS=$'\n'.
  $(echo "$analyzer" | tr ' ' '\n') $PY_FILES

done
