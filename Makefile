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
	@echo "  install    Install the development environment"
	@echo "  test       Run pytest"
	@echo "  lint       Run linting tools against changed files"
	@echo "  format     Format changed files with ruff (updates .py files in place)"
	@echo "  notebooks  Execute notebooks (updates .ipynb files in place)"
	@echo "  mkdocs     Build and deploy mkdocs documentation"
	@echo "  clean      Remove generated files and temp files"

.PHONY: install
install: check-conda .make-install

.PHONY: check-conda
check-conda:
ifdef CONDA_DEFAULT_ENV
ifeq ($(CONDA_DEFAULT_ENV),base)
	$(error Pixeltable must be installed from a conda environment (not `base`))
endif
else
	$(error Pixeltable must be installed from a conda environment)
endif

# Use the placeholder `.make-install` to track whether the installation is up-to-date
.make-install: poetry.lock
	@echo "Installing poetry ..."
	@python -m pip install --upgrade pip
	@python -m pip install poetry==1.8.2
	@poetry self add "poetry-dynamic-versioning[plugin]"
	@echo "Installing dependencies from poetry ..."
	@poetry install --with dev
	# YOLOX cannot be installed via poetry, sadly
	@echo "Installing YOLOX ..."
	@python -m pip install git+https://github.com/Megvii-BaseDetection/YOLOX@ac58e0a
	@echo "Installing Jupyter kernel ..."
	@python -m ipykernel install --user --name=$(KERNEL_NAME)
	@touch .make-install

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

.PHONY: notebooks
notebooks: install
	@echo "Running notebooks and overwriting outputs ..."
	@pytest --overwrite --nbmake --nbmake-kernel=$(KERNEL_NAME) docs/release/tutorials/*.ipynb

.PHONY: mkdocs
mkdocs: install
	@scripts/mkdocs.sh

.PHONY: release
release: install
	@scripts/release.sh

.PHONY: clean
clean:
	@rm -f *.mp4 docs/source/tutorials/*.mp4 || true
	@rm -f .make-install || true
	@rm -rf site || true
	@rm -rf target || true
