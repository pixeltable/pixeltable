# Detect OS and set shell accordingly
ifeq ($(OS),Windows_NT)
    SHELL := pwsh.exe
    # PowerShell command to get directory name
    # Define Windows-specific commands
    SHELL_PREFIX := pwsh.exe
    MKDIR := powershell -Command New-Item -ItemType Directory -Path
    TOUCH := powershell -Command New-Item -ItemType File -Path -Force
    RM := powershell -Command Remove-Item -Force
    RMDIR := powershell -Command Remove-Item -Force -Recurse
    SET_ENV := set
    KERNEL_NAME := $(shell powershell -Command "(Get-Item .).Name")
    ULIMIT_CMD :=
else
    SHELL := /bin/bash
    # Define Unix-specific commands
    SHELL_PREFIX :=
    MKDIR := mkdir -p
    TOUCH := touch
    RM := rm -f
    RMDIR := rm -rf
    SET_ENV := export
    KERNEL_NAME := $(shell basename `pwd`)
    ULIMIT_CMD := ulimit -n 4000;
endif

# Common test parameters
PYTEST_COMMON_ARGS := -v -n auto --dist loadgroup --maxprocesses 6 tests

# We ensure the TQDM progress bar is updated exactly once per cell execution, by setting the refresh rate equal to the timeout
NB_CELL_TIMEOUT := 3600

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo 'Usage: make <target>'
	@echo 'You must be in a conda environment to install the Pixeltable dev environment.'
	@echo 'See: https://github.com/pixeltable/pixeltable/blob/main/CONTRIBUTING.md'
	@echo ''
	@echo 'Targets:'
	@echo '  install       Install the development environment'
	@echo '  test          Run pytest and check'
	@echo '  fulltest      Run fullpytest, nbtest, and check'
	@echo '  check		   Run typecheck, docscheck, lint, and formatcheck'
	@echo '  format        Run `ruff format` (updates .py files in place)'
	@echo '  release       Create a pypi release and post to github'
	@echo '  release-docs  Build and deploy API documentation (must be run from home repo, not a fork)'
	@echo ''
	@echo 'Individual test targets:'
	@echo '  clean         Remove generated files and temp files'
	@echo '  pytest        Run `pytest`'
	@echo '  fullpytest    Run `pytest`, including expensive tests'
	@echo '  nbtest        Run `pytest` on notebooks'
	@echo '  typecheck     Run `mypy`'
	@echo '  docscheck     Run `mkdocs build --strict`'
	@echo '  lint          Run `ruff check`'
	@echo '  formatcheck   Run `ruff format --check` (check only, do not modify files)

.PHONY: setup-install
setup-install:
ifeq ($(OS),Windows_NT)
	@powershell -Command "if (-not (Test-Path '.make-install')) { New-Item -ItemType Directory -Path '.make-install' }"
else
	@$(MKDIR) .make-install
endif
ifdef CONDA_DEFAULT_ENV
ifeq ($(CONDA_DEFAULT_ENV),base)
	$(error Pixeltable must be installed from a conda environment (not `base`))
endif
else
	$(error Pixeltable must be installed from a conda environment)
endif

.make-install/poetry:
	@echo 'Installing poetry ...'
	@python -m pip install -qU pip
	@python -m pip install -q poetry==2.1.1
	@poetry self add 'poetry-dynamic-versioning[plugin]==1.7.1'
	@$(TOUCH) .make-install/poetry

.make-install/deps: poetry.lock
	@echo 'Installing dependencies from poetry ...'
	@$(SET_ENV) CMAKE_ARGS='-DLLAVA_BUILD=OFF'
	@poetry install --with dev
	@$(TOUCH) .make-install/deps

.make-install/others:
	@echo 'Installing Jupyter kernel ...'
	@python -m ipykernel install --user --name=$(KERNEL_NAME)
	@$(TOUCH) .make-install/others

.PHONY: install
install: setup-install .make-install/poetry .make-install/deps .make-install/others

.PHONY: test
test: pytest check
	@echo 'All tests passed.'

.PHONY: fulltest
fulltest: fullpytest nbtest check
	@echo 'All tests passed.'

.PHONY: check
check: typecheck docscheck lint formatcheck
	@echo 'All static checks passed.'

.PHONY: pytest
pytest: install
	@echo 'Running `pytest` ...'
	@$(ULIMIT_CMD) pytest $(PYTEST_COMMON_ARGS)

.PHONY: fullpytest
fullpytest: install
	@echo 'Running `pytest`, including expensive tests ...'
	@$(ULIMIT_CMD) pytest -m '' $(PYTEST_COMMON_ARGS)

.PHONY: nbtest
nbtest: install
	@$(SET_ENV) TQDM_MININTERVAL=$(NB_CELL_TIMEOUT)
	@echo 'Running `pytest` on notebooks ...'
	@$(SHELL_PREFIX) scripts/prepare-nb-tests.sh --no-pip docs/notebooks tests
	@$(ULIMIT_CMD) pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) target/nb-tests/*.ipynb

.PHONY: typecheck
typecheck: install
	@echo 'Running `mypy` ...'
	@mypy pixeltable tests tool

.PHONY: docscheck
docscheck: install
	@echo 'Running `mkdocs build --strict` ...'
	@mkdocs build --strict

.PHONY: lint
lint: install
	@echo 'Running `ruff check` ...'
	@ruff check pixeltable tool

.PHONY: formatcheck
formatcheck: install
	@echo 'Running `ruff format --check` ...'
	@ruff format --check pixeltable tests tool
	@echo 'Running `ruff check --select I` ...'
	@ruff check --select I pixeltable tests tool

.PHONY: format
format: install
	@echo 'Running `ruff format` ...'
	@ruff format pixeltable tests tool
	@echo 'Running `ruff check --select I --fix` ...'
	@ruff check --select I --fix pixeltable tests tool

.PHONY: release
release: install
	@$(SHELL_PREFIX) scripts/release.sh

.PHONY: release-docs
release-docs: install
	@mkdocs gh-deploy

.PHONY: clean
clean:
	@$(RM) *.mp4 docs/source/tutorials/*.mp4 || true
	@$(RMDIR) .make-install || true
	@$(RMDIR) site || true
	@$(RMDIR) target || true
