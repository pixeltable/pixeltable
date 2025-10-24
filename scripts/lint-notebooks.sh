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

    colab_line=$(grep "colab.research.google.com" "$fn")
    # Try badge format first, then frontmatter format
    colab_slug=$(echo "$colab_line" | sed -E "s/^.*$colab_badge\($colab_url_prefix(.*)\).*$/\1/")
    if [ -z "$colab_slug" ] || [ "$colab_slug" = "$colab_line" ]; then
        # Try frontmatter format: [Open in Colab](url) - match non-greedy up to first )
        colab_slug=$(echo "$colab_line" | sed -E "s/^.*$colab_frontmatter\($colab_url_prefix([^)]+)\).*$/\1/")
    fi
    if [ -n "$colab_line" ] && ([ -z "$colab_slug" ] || [ "$colab_slug" = "$colab_line" ]); then
        echo "[ERROR] Colab link is improperly formatted:"
        echo "[ERROR] $colab_line"
        errors=$((errors + 1))
    elif [ -n "$colab_line" ] && [[ "$colab_slug" != "$fn" ]]; then
        echo "[ERROR] Colab path does not match notebook path: $colab_slug"
        errors=$((errors + 1))
    fi

    kaggle_line=$(grep "kaggle.com/kernels/welcome" "$fn")
    # Try badge format first, then frontmatter format
    kaggle_slug=$(echo "$kaggle_line" | sed -E "s/^.*$kaggle_badge\($kaggle_url_prefix(.*)\).*$/\1/")
    if [ -z "$kaggle_slug" ] || [ "$kaggle_slug" = "$kaggle_line" ]; then
        # Try frontmatter format: [Open in Kaggle](url) - match non-greedy up to first )
        kaggle_slug=$(echo "$kaggle_line" | sed -E "s/^.*$kaggle_frontmatter\($kaggle_url_prefix([^)]+)\).*$/\1/")
    fi
    if [ -n "$kaggle_line" ] && ([ -z "$kaggle_slug" ] || [ "$kaggle_slug" = "$kaggle_line" ]); then
        echo "[ERROR] Kaggle link is improperly formatted:"
        echo "[ERROR] $kaggle_line"
        errors=$((errors + 1))
    elif [ -n "$kaggle_line" ] && [[ "$kaggle_slug" != "$fn" ]]; then
        echo "[ERROR] Kaggle path does not match notebook path: $kaggle_slug"
        errors=$((errors + 1))
    fi

    download_line=$(grep "Download Notebook" "$fn")
    download_slug=$(echo "$download_line" | sed -E "s/^.*\"$download_url_prefix([^\"]*).\".*$/\1/")
    if [ -n "$download_line" ] && [ -z "$download_slug" ]; then
        echo "[ERROR] Download Notebook link is improperly formatted:"
        echo "[ERROR] $download_line"
        errors=$((errors + 1))
    elif [ -n "$download_line" ] && [[ "$download_slug" != "$fn" ]]; then
        echo "[ERROR] Download Notebook path does not match notebook path: $download_slug"
        errors=$((errors + 1))
    fi
done

if [ $errors -gt 0 ]; then
    echo ""
    echo "There were $errors errors."
    exit 1
fi
