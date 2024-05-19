#!/bin/bash -e

# Pixeltable release script

SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(readlink -f "$SCRIPT_DIR/..")"

echo -e "\n============= Running pixeltable release script.\n"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

if [ "$(git remote get-url home)" != 'https://github.com/pixeltable/pixeltable' ]; then
  echo "Unexpected home repo: $(git remote get-url home)"
  exit 1
fi

if [[ -n "$(git diff-index HEAD)" || -n "$(git diff-files)" || -n "$(git ls-files --exclude-standard --others)" ]]; then
  echo "The release script must be run from a clean git repo."
  exit 1
fi

if [ -z "$PYPI_API_KEY" ]; then
  echo "PYPI_API_KEY not defined in environment; searching in ~/.pixeltable/config.yaml."
  PYPI_API_KEY=$(
    python -c "import yaml,sys; y = yaml.safe_load(sys.stdin); print(y['pypi']['api_key'])" < ~/.pixeltable/config.yaml
  )
fi

echo -n "Enter version number for release: "
read VERSION

echo ""
echo "This will publish version v$VERSION to PyPI. Enter to confirm; Ctrl-C to abort."
read

git tag "v$VERSION"
git push home "v$VERSION"

echo "v$VERSION tag created and pushed to home repo."
echo "Enter to proceed; Ctrl-C to abort."
read

make clean
poetry build
poetry publish --username __token__ --password "$PYPI_API_KEY"

echo "Creating release on github ..."
gh release create "v$VERSION" --generate-notes
open "https://github.com/pixeltable/pixeltable/releases/tag/v$VERSION"
