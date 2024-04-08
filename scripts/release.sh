#!/bin/bash

# Pixeltable release script
# This MUST be run in the home repo (pixeltable/pixeltable), not a fork!

set -e

if [ -z "$1" ]; then
  echo "Usage: release.sh <pypi-api-key>"
  exit 1
fi

if [ "$(git remote get-url origin)" != 'https://github.com/pixeltable/pixeltable' ]; then
  echo "This MUST be run in the home repo (pixeltable/pixeltable), not a fork!"
  exit 1
fi

PYPI_API_KEY="$1"
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$SCRIPT_DIR/.."
VERSION="$(grep -o '^version = "[^"]*' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "//')"

if [ -z "$VERSION" ]; then
  echo "Could not extract version from pyproject.toml."
  exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "This will publish version $VERSION. Enter to confirm; Ctrl-C to abort."
read

cd "$PROJECT_ROOT"
poetry build
poetry publish --username __token__ --password "$PYPI_API_KEY"
git tag "v$VERSION"
git push "v$VERSION"
