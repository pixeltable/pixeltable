#!/bin/bash -e

# Pixeltable release script

SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(readlink -f "$SCRIPT_DIR/..")"

if [ -f 'pyproject.toml' ]; then
  PROJECT_NAME=$(python -c "import toml; y = toml.load('pyproject.toml'); print(y['project']['name'])")
  VERSION=$(python -c "import toml; y = toml.load('pyproject.toml'); print(y['project'].get('version'))")
else
  echo "No pyproject.toml found. This script must be run from the root of a Pixeltable project."
  exit 1
fi

echo -e "\n============= Running release script for project $PROJECT_NAME.\n"

if [[ ! $(gh --version) ]]; then
  echo "You must have Github commandline utilities installed to run this script. See: https://cli.github.com/"
  exit 1
fi

if [ "$(git remote get-url home)" != "https://github.com/pixeltable/$PROJECT_NAME" ]; then
  echo "Unexpected home repo: $(git remote get-url home)"
  exit 1
fi

git diff > /dev/null  # Needed to ensure correct behavior of subsequent `git diff-index` call

if [[ -n "$(git diff-index HEAD)" || -n "$(git diff-files)" || -n "$(git ls-files --exclude-standard --others)" ]]; then
  echo "The release script must be run from a clean git repo."
  exit 1
fi

if [ -z "$PYPI_API_KEY" ]; then
  echo "PYPI_API_KEY not defined in environment; searching in ~/.pixeltable/config.toml."
  PYPI_API_KEY=$(
    python -c "import toml,sys; y = toml.load(sys.stdin); print(y['pypi']['api_key'])" < ~/.pixeltable/config.toml
  )
fi

git fetch home
git checkout home/main

if [ "$VERSION" == 'None' ]; then
  echo "Project uses dynamic versioning."
  echo -n "Enter version number for release: "
  read VERSION
fi

echo ""
echo "This will publish version v$VERSION to PyPI. Enter to confirm; Ctrl-C to abort."
read

git tag "v$VERSION"
git push home "v$VERSION"

echo "v$VERSION tag created and pushed to home repo."
echo "Enter to proceed; Ctrl-C to abort."
read

git tag -d release
git push --delete home release
git tag release
git push home release

if [ -f 'Makefile' ]; then
  echo "Running make clean ..."
  make clean
fi

poetry build
poetry publish --username __token__ --password "$PYPI_API_KEY"

echo "Creating release on github ..."
gh release create "v$VERSION" --generate-notes --repo "pixeltable/$PROJECT_NAME"
open "https://github.com/pixeltable/$PROJECT_NAME/releases/tag/v$VERSION"
