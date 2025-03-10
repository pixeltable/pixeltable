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
	@echo "Usage: make <target>"
	@echo "You must be in a conda environment to install the Pixeltable dev environment."
	@echo "See: https://github.com/pixeltable/pixeltable/blob/main/CONTRIBUTING.md"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install the development environment"
	@echo "  test          Run pytest, typecheck, and docstest"
	@echo "  fulltest      Run pytest, nbtest, typecheck, and docstest, including expensive tests"
	@echo "  release       Create a pypi release and post to github"
	@echo "  release-docs  Build and deploy API documentation (must be run from home repo, not a fork)"
	@echo ""
	@echo "Individual test targets:"
	@echo "  clean         Remove generated files and temp files"
	@echo "  pytest        Run pytest"
	@echo "  fullpytest    Run pytest, including expensive tests"
	@echo "  typecheck     Run mypy"
	@echo "  docstest      Run mkdocs build --strict"
	@echo "  formattest    Run the formatter (check only)"
	@echo "  format        Run the formatter (update .py files in place)"
	@echo "  nbtest        Run notebook tests"
	@echo "  lint          Run linting tools against changed files"

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

YOLOX_OK := $(shell python -c "import sys; sys.stdout.write(str(sys.version_info[1] <= 10))")

.make-install/poetry:
	@echo "Installing poetry ..."
	@python -m pip install -qU pip
	@python -m pip install -q poetry==2.1.1
	@poetry self add "poetry-dynamic-versioning[plugin]==1.7.1"
	@$(TOUCH) .make-install/poetry

.make-install/deps: poetry.lock
	@echo "Installing dependencies from poetry ..."
	@$(SET_ENV) CMAKE_ARGS='-DLLAVA_BUILD=OFF'
	@poetry install --with dev
	@$(TOUCH) .make-install/deps

.make-install/others:
ifeq ($(YOLOX_OK), True)
	# YOLOX only works on python <= 3.10 and cannot be installed via poetry
	@echo "Installing YOLOX ..."
	# We have to include protobuf in the `pip install` or else YOLOX will downgrade it
	@python -m pip install -q git+https://github.com/Megvii-BaseDetection/YOLOX@ac58e0a protobuf==5.28.3
else
	@echo "Python version is >= 3.11; skipping YOLOX installation."
endif
	@echo "Installing Jupyter kernel ..."
	@python -m ipykernel install --user --name=$(KERNEL_NAME)
	@$(TOUCH) .make-install/others

.PHONY: install
install: setup-install .make-install/poetry .make-install/deps .make-install/others

.PHONY: test
test: pytest typecheck docstest formattest
	@echo "All tests passed!"

.PHONY: fulltest
fulltest: fullpytest nbtest typecheck docstest formattest
	@echo "All tests passed!"

.PHONY: pytest
pytest: install
	@echo "Running pytest ..."
	@$(ULIMIT_CMD) pytest $(PYTEST_COMMON_ARGS)

.PHONY: fullpytest
fullpytest: install
	@echo "Running pytest, including expensive tests ..."
	@$(ULIMIT_CMD) pytest -m '' $(PYTEST_COMMON_ARGS)

.PHONY: nbtest
nbtest: install
	@$(SET_ENV) TQDM_MININTERVAL=$(NB_CELL_TIMEOUT)
	@echo "Running pytest on notebooks ..."
	@$(SHELL_PREFIX) scripts/prepare-nb-tests.sh --no-pip docs/notebooks tests
	@$(ULIMIT_CMD) pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) target/nb-tests/*.ipynb

.PHONY: typecheck
typecheck: install
	@echo "Running mypy ..."
	@mypy pixeltable tests tool

.PHONY: docstest
docstest: install
	@echo "Running mkdocs build --strict ..."
	@mkdocs build --strict

.PHONY: formattest
formattest: install
	@echo "Running ruff format --check ..."
	@ruff format --check
	@echo "Running ruff check --select I ..."
	@ruff check --select I

.PHONY: lint
lint: install
	@$(SHELL_PREFIX) scripts/lint-changed-files.sh

.PHONY: format
format: install
	@echo "Running ruff format ..."
	@ruff format
	@echo "Running ruff check --select I --fix ..."
	@ruff check --select I --fix

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