SHELL := /bin/bash
KERNEL_NAME := $(shell basename `pwd`)
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo "You must be in a conda environment to install the Pixeltable dev environment."
	@echo "See: https://www.notion.so/Setting-up-a-dev-environment-83a1ca32de034f94bd7fee0ddb46fed8"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install the development environment"
	@echo "  test          Run pytest"
	@echo "  lint          Run linting tools against changed files"
	@echo "  format        Format changed files with ruff (updates .py files in place)"
	@echo "  notebooks     Execute notebooks (updates .ipynb files in place)"
	@echo "  release       Create a pypi release and post to github"
	@echo "  release-docs  Build and deploy API documentation"
	@echo "  clean         Remove generated files and temp files"
	@echo "  *.ipynb       Run the notebook/notebooks (updates output cells in place)"

.PHONY: check-conda
check-conda:
ifdef CONDA_DEFAULT_ENV
ifeq ($(CONDA_DEFAULT_ENV),base)
	$(error Pixeltable must be installed from a conda environment (not `base`))
endif
else
	$(error Pixeltable must be installed from a conda environment)
endif

YOLOX_OK := $(shell python -c "import sys; sys.stdout.write(str(sys.version_info[1] <= 10))")

# Use the placeholder `.make-install` to track whether the installation is up-to-date
.make-install: poetry.lock
	@echo "Installing poetry ..."
	@python -m pip install --upgrade pip
	@python -m pip install poetry==1.8.2
	@poetry self add "poetry-dynamic-versioning[plugin]"
	@echo "Installing dependencies from poetry ..."
	@poetry install --with dev
ifeq ($(YOLOX_OK), True)
	# YOLOX only works on python <= 3.10 and cannot be installed via poetry
	@echo "Installing YOLOX ..."
	@python -m pip install -q git+https://github.com/Megvii-BaseDetection/YOLOX@ac58e0a
else
	@echo "Python version is >= 3.11; skipping YOLOX installation."
endif
	@echo "Installing Jupyter kernel ..."
	@python -m ipykernel install --user --name=$(KERNEL_NAME)
	@touch .make-install

.PHONY: install
install: check-conda .make-install

.PHONY: test
test: install
	@echo "Running pytest ..."
	@ulimit -n 4000; pytest -v

.PHONY: lint
lint: install
	@scripts/lint-changed-files.sh

.PHONY: format
format: install
	@scripts/format-changed-files.sh

NB_CELL_TIMEOUT := 3600
# for non-interactive running and rendering of notebooks
# we ensure the tqdm progress bar is updated exactly once per cell execution by setting the refresh rate to the timeout
# if it is executed more than once, every update gets its own line (due to ignored \r characters)
# see: https://github.com/tqdm/tqdm?tab=readme-ov-file#faq-and-known-issues
%.ipynb: export TQDM_MININTERVAL=$(NB_CELL_TIMEOUT)
%.ipynb: install
	@echo "Running and over-writing notebook $@ ..."
	@pytest --overwrite --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) $@

.PHONY: notebooks
notebooks: docs/release/**/*.ipynb

.PHONY: release
release: install
	@scripts/release.sh

.PHONY: release-docs
release-docs: install
	@mkdocs gh-deploy

.PHONY: clean
clean:
	@rm -f *.mp4 docs/source/tutorials/*.mp4 || true
	@rm -f .make-install || true
	@rm -rf site || true
	@rm -rf target || true
