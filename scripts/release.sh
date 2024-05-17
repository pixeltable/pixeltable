#!/bin/bash

# Pixeltable release script
# This MUST be run in the home repo (pixeltable/pixeltable), not a fork!

set -e

if [ -z "$2" ]; then
  echo "Usage: release.sh <version> <pypi-api-key>"
  echo "Example: release.sh 0.2.6 api-key"
  exit 1
fi

if [ "$(git remote get-url origin)" != 'https://github.com/pixeltable/pixeltable' ]; then
  echo "This MUST be run in the home repo (pixeltable/pixeltable), not a fork!"
  exit 1
fi

VERSION="$1"
PYPI_API_KEY="$2"
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$SCRIPT_DIR/.."

# VERSION="$(grep -o '^version = "[^"]*' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "//')"
# if [ -z "$VERSION" ]; then
#   echo "Could not extract version from pyproject.toml."
#   exit 1
# fi

echo "Project root: $PROJECT_ROOT"
echo "This will publish version $VERSION. Enter to confirm; Ctrl-C to abort."
read

cd "$PROJECT_ROOT"
git tag "v$VERSION"
git push origin "v$VERSION"

echo "v$VERSION tag created and pushed to home repo."
echo "Enter to proceed; Ctrl-C to abort."
read

make clean
poetry build
poetry publish --username __token__ --password "$PYPI_API_KEY"

echo "Updating mkdocs ..."
mkdocs build
mkdocs gh-deploy

echo "Creating release on github ..."
gh release create v$VERSION --generate-notes
open https://github.com/pixeltable/pixeltable/releases/tag/v$VERSION
