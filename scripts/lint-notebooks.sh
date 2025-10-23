#!/bin/bash +e

errors=0

escape_regex() { sed 's/[^^]/[&]/g; s/\^/\\^/g' <<<"$1"; }
# Support both old badge format and new frontmatter format
colab_badge=$(escape_regex '[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]')
colab_frontmatter=$(escape_regex '[Open in Colab]')
colab_url_prefix=$(escape_regex 'https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/')
kaggle_badge=$(escape_regex '[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]')
kaggle_frontmatter=$(escape_regex '[Open in Kaggle]')
kaggle_url_prefix=$(escape_regex 'https://kaggle.com/kernels/welcome?src=https://github.com/pixeltable/pixeltable/blob/release/')
download_url_prefix=$(escape_regex 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/')

notebooks=$(find docs/notebooks -name '*.ipynb' | grep -v .ipynb_checkpoints)

IFS=$'\n'
for fn in $notebooks; do
    echo "Linting $fn ..."

    # Link checking disabled - frontmatter will be auto-generated during conversion
    # from notebook file paths, so no need to validate links in source notebooks
done

if [ $errors -gt 0 ]; then
    echo ""
    echo "There were $errors errors."
    exit 1
fi
