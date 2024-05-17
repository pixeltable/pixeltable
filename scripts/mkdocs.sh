#!/bin/bash -e

# Pixeltable mkdocs publish script
# This MUST be run in the home repo (pixeltable/pixeltable), not a fork!

if [ "$(git remote get-url origin)" != 'https://github.com/pixeltable/pixeltable' ]; then
  echo "This MUST be run in the home repo (pixeltable/pixeltable), not a fork!"
  exit 1
fi

mkdocs build
mkdocs gh-deploy
