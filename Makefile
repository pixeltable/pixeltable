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
	@echo "  nbtest        Run notebook tests"
	@echo "  typecheck     Run mypy"
	@echo "  lint          Run linting tools against changed files"
	@echo "  format        Format changed files with ruff (updates .py files in place)"
	@echo "  release       Create a pypi release and post to github"
	@echo "  release-docs  Build and deploy API documentation"
	@echo "  clean         Remove generated files and temp files"

.PHONY: setup-install
setup-install:
	@mkdir -p .make-install
ifdef CONDA_DEFAULT_ENV
ifeq ($(CONDA_DEFAULT_ENV),base)
	$(error Pixeltable must be installed from a conda environment (not `base`))
endif
else
	$(error Pixeltable must be installed from a conda environment)
endif

YOLOX_OK := $(shell python -c "import sys; sys.stdout.write(str(sys.version_info[1] <= 10))")

.make-install/poetry:
	@echo "Installing poetry ..."
	@python -m pip install -qU pip
	@python -m pip install -q poetry==1.8.2
	@poetry self add "poetry-dynamic-versioning[plugin]"
	@touch .make-install/poetry

.make-install/deps: poetry.lock
	@echo "Installing dependencies from poetry ..."
	@poetry install --with dev
	@touch .make-install/deps

.make-install/others:
ifeq ($(YOLOX_OK), True)
	# YOLOX only works on python <= 3.10 and cannot be installed via poetry
	@echo "Installing YOLOX ..."
	# We have to include protobuf in the `pip install` or else YOLOX will downgrade it
	@python -m pip install -q git+https://github.com/Megvii-BaseDetection/YOLOX@ac58e0a protobuf==5.27.0
else
	@echo "Python version is >= 3.11; skipping YOLOX installation."
endif
	@echo "Installing Jupyter kernel ..."
	@python -m ipykernel install --user --name=$(KERNEL_NAME)
	@touch .make-install/others

.PHONY: install
install: setup-install .make-install/poetry .make-install/deps .make-install/others

.PHONY: test
test: install
	@echo "Running pytest ..."
	@ulimit -n 4000; pytest -v

NB_CELL_TIMEOUT := 3600
# We ensure the TQDM progress bar is updated exactly once per cell execution, by setting the refresh rate equal
# to the timeout. This ensures it will pretty-print if --overwrite is set on nbmake.
# see: https://github.com/tqdm/tqdm?tab=readme-ov-file#faq-and-known-issues
.PHONY: nbtest
nbtest: install
	@export TQDM_MININTERVAL=$(NB_CELL_TIMEOUT)
	@echo "Running pytest on notebooks ..."
	@scripts/prepare-nb-tests.sh --no-pip
	@ulimit -n 4000; pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) target/nb-tests/*.ipynb

.PHONY: typecheck
typecheck: install
	@mypy pixeltable/*.py pixeltable/catalog pixeltable/ext pixeltable/functions

.PHONY: lint
lint: install
	@scripts/lint-changed-files.sh

.PHONY: format
format: install
	@scripts/format-changed-files.sh

.PHONY: release
release: install
	@scripts/release.sh

.PHONY: release-docs
release-docs: install
	@mkdocs gh-deploy

.PHONY: clean
clean:
	@rm -f *.mp4 docs/source/tutorials/*.mp4 || true
	@rm -rf .make-install || true
	@rm -rf site || true
	@rm -rf target || true
