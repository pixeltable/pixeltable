# Detect OS and set shell accordingly
ifeq ($(OS),Windows_NT)
    SHELL := pwsh.exe
    # PowerShell command to get directory name
    KERNEL_NAME := $(shell (Get-Item .).Name)
else
    SHELL := /bin/bash
    KERNEL_NAME := $(shell basename `pwd`)
endif

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
	@if (-not (Test-Path .make-install)) { New-Item -ItemType Directory -Path .make-install }
else
	@mkdir -p .make-install
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
ifeq ($(OS),Windows_NT)
	@New-Item -ItemType File -Path .make-install/poetry -Force
else
	@touch .make-install/poetry
endif

.make-install/deps: poetry.lock
	@echo "Installing dependencies from poetry ..."
	@set CMAKE_ARGS='-DLLAVA_BUILD=OFF'
	@poetry install --with dev
ifeq ($(OS),Windows_NT)
	@New-Item -ItemType File -Path .make-install/deps -Force
else
	@touch .make-install/deps
endif

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
ifeq ($(OS),Windows_NT)
	@New-Item -ItemType File -Path .make-install/others -Force
else
	@touch .make-install/others
endif

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
ifeq ($(OS),Windows_NT)
	@pytest -v -n auto --dist loadgroup --maxprocesses 6 tests
else
	@ulimit -n 4000; pytest -v -n auto --dist loadgroup --maxprocesses 6 tests
endif

.PHONY: fullpytest
fullpytest: install
	@echo "Running pytest, including expensive tests ..."
ifeq ($(OS),Windows_NT)
	@pytest -v -m '' -n auto --dist loadgroup --maxprocesses 6 tests
else
	@ulimit -n 4000; pytest -v -m '' -n auto --dist loadgroup --maxprocesses 6 tests
endif

NB_CELL_TIMEOUT := 3600
# We ensure the TQDM progress bar is updated exactly once per cell execution, by setting the refresh rate equal
# to the timeout. This ensures it will pretty-print if --overwrite is set on nbmake.
# see: https://github.com/tqdm/tqdm?tab=readme-ov-file#faq-and-known-issues
.PHONY: nbtest
nbtest: install
ifeq ($(OS),Windows_NT)
	@set TQDM_MININTERVAL=$(NB_CELL_TIMEOUT)
else
	@export TQDM_MININTERVAL=$(NB_CELL_TIMEOUT)
endif
	@echo "Running pytest on notebooks ..."
ifeq ($(OS),Windows_NT)
	@pwsh.exe scripts/prepare-nb-tests.sh --no-pip docs/notebooks tests
	@pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) target/nb-tests/*.ipynb
else
	@scripts/prepare-nb-tests.sh --no-pip docs/notebooks tests
	@ulimit -n 4000; pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) target/nb-tests/*.ipynb
endif

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
ifeq ($(OS),Windows_NT)
	@pwsh.exe scripts/lint-changed-files.sh
else
	@scripts/lint-changed-files.sh
endif

.PHONY: format
format: install
ifeq ($(OS),Windows_NT)
	@pwsh.exe scripts/format-changed-files.sh
else
	@scripts/format-changed-files.sh
endif

.PHONY: release
release: install
ifeq ($(OS),Windows_NT)
	@pwsh.exe scripts/release.sh
else
	@scripts/release.sh
endif

.PHONY: release-docs
release-docs: install
	@mkdocs gh-deploy

.PHONY: clean
clean:
ifeq ($(OS),Windows_NT)
	@if exist "*.mp4" del /Q "*.mp4"
	@if exist "docs\source\tutorials\*.mp4" del /Q "docs\source\tutorials\*.mp4"
	@if exist ".make-install" rmdir /S /Q ".make-install"
	@if exist "site" rmdir /S /Q "site"
	@if exist "target" rmdir /S /Q "target"
else
	@rm -f *.mp4 docs/source/tutorials/*.mp4 || true
	@rm -rf .make-install || true
	@rm -rf site || true
	@rm -rf target || true
endif