# Detect OS and set shell accordingly
ifeq ($(OS),Windows_NT)
    SHELL := pwsh.exe
    # PowerShell command to get directory name
    # Define Windows-specific commands
    SHELL_PREFIX := pwsh.exe
    MKDIR := powershell -Command New-Item -ItemType Directory -Path
    TOUCH := powershell -Command New-Item -ItemType File -Path
    RM := powershell -Command Remove-Item -Force
    RMDIR := powershell -Command Remove-Item -Force -Recurse
    SET_ENV := set
    DIR_SEP := \\
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
    DIR_SEP := /
    KERNEL_NAME := $(shell basename `pwd`)
    ULIMIT_CMD := ulimit -n 4000;
endif

# Common test parameters
PYTEST_COMMON_ARGS := -v -n auto --dist loadgroup --maxprocesses 6 tests
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
	@echo "  nbtest        Run notebook tests"
	@echo "  lint          Run linting tools against changed files"
	@echo "  format        Format changed files with ruff (updates .py files in place)"

.PHONY: setup-install
setup-install:
ifeq ($(OS),Windows_NT)
	@powershell -Command "if (-not (Test-Path '.make-install')) { New-Item -ItemType Directory -Path '.make-install' }"
else
	@if [ ! -d ".make-install" ]; then $(MKDIR) .make-install; fi
endif
ifdef CONDA_DEFAULT_ENV
ifeq ($(CONDA_DEFAULT_ENV),base)
	$(error Pixeltable must be installed from a conda environment (not `base`))
endif
else
	$(error Pixeltable must be installed from a conda environment)
endif

YOLOX_OK := $(shell python -c "import sys; sys.stdout.write(str(sys.version_info[1] <= 10))")
WHISPERX_OK := $(shell python -c "import sys; sys.stdout.write(str(sys.version_info[1] <= 12))")

.make-install/poetry:
	@echo "Installing poetry ..."
	@python -m pip install -qU pip
	@python -m pip install -q poetry==1.8.4
	@poetry self add "poetry-dynamic-versioning[plugin]"
	@$(TOUCH) .make-install$(DIR_SEP)poetry -Force

.make-install/deps: poetry.lock
	@echo "Installing dependencies from poetry ..."
	@$(SET_ENV) CMAKE_ARGS='-DLLAVA_BUILD=OFF'
	@poetry install --with dev
	@$(TOUCH) .make-install$(DIR_SEP)deps -Force

.make-install/others:
ifeq ($(YOLOX_OK), True)
	# YOLOX only works on python <= 3.10 and cannot be installed via poetry
	@echo "Installing YOLOX ..."
	# We have to include protobuf in the `pip install` or else YOLOX will downgrade it
	@python -m pip install -q git+https://github.com/Megvii-BaseDetection/YOLOX@ac58e0a protobuf==5.28.3
else
	@echo "Python version is >= 3.11; skipping YOLOX installation."
endif
ifeq ($(WHISPERX_OK), True)
	# WhisperX only works on python <= 3.12 and has overly specific version requirements
	# that make it difficult to use with poetry
	@echo "Installing WhisperX ..."
	@python -m pip install -q git+https://github.com/m-bain/whisperX.git@f2da2f8 typer==0.9.0
else
	@echo "Python version is >= 3.13; skipping WhisperX installation."
endif
	@echo "Installing Jupyter kernel ..."
	@python -m ipykernel install --user --name=$(KERNEL_NAME)
	@$(TOUCH) .make-install$(DIR_SEP)others -Force

.PHONY: install
install: setup-install .make-install/poetry .make-install/deps .make-install/others

.PHONY: test
test: pytest typecheck docstest
	@echo "All tests passed!"

.PHONY: fulltest
fulltest: fullpytest nbtest typecheck docstest
	@echo "All tests passed!"

.PHONY: pytest
pytest: install
	@echo "Running pytest ..."
	@$(ULIMIT_CMD) pytest $(PYTEST_COMMON_ARGS)

.PHONY: fullpytest
fullpytest: install
	@echo "Running pytest, including expensive tests ..."
	@$(ULIMIT_CMD) pytest -v -m '' -n auto --dist loadgroup --maxprocesses 6 tests

.PHONY: nbtest
nbtest: install
	@$(SET_ENV) TQDM_MININTERVAL=$(NB_CELL_TIMEOUT)
	@echo "Running pytest on notebooks ..."
	@$(SHELL_PREFIX) scripts/prepare-nb-tests.sh --no-pip docs/notebooks tests
	@$(ULIMIT_CMD) pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) target/nb-tests/*.ipynb

.PHONY: typecheck
typecheck: install
	@echo "Running mypy ..."
	@mypy pixeltable

.PHONY: docstest
docstest: install
	@echo "Running mkdocs build --strict ..."
	@mkdocs build --strict

.PHONY: lint
lint: install
	@$(SHELL_PREFIX) scripts/lint-changed-files.sh

.PHONY: format
format: install
	@$(SHELL_PREFIX) scripts/format-changed-files.sh

.PHONY: release
release: install
	@$(SHELL_PREFIX) scripts/release.sh

.PHONY: release-docs
release-docs: install
	@mkdocs gh-deploy

.PHONY: clean
clean:
	@$(RM) *.mp4 docs$(DIR_SEP)source$(DIR_SEP)tutorials$(DIR_SEP)*.mp4 || true
	@$(RMDIR) .make-install || true
	@$(RMDIR) site || true
	@$(RMDIR) target || true