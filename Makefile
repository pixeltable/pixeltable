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

UV_ARGS := --group extra-dev

# Common test parameters
PYTEST_COMMON_ARGS := -v -n auto --dist loadgroup --maxprocesses 6 --reruns 2 \
	--only-rerun 'That Pixeltable operation could not be completed because it conflicted with'

# We ensure the TQDM progress bar is updated exactly once per cell execution, by setting the refresh rate equal to the timeout
NB_CELL_TIMEOUT := 3600
TQDM_MININTERVAL := $(NB_CELL_TIMEOUT)

# Needed for LLaMA build to work correctly on some Linux systems
CMAKE_ARGS := -DLLAVA_BUILD=OFF

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo 'Usage: make <target>'
	@echo 'You must be in a conda environment to install the Pixeltable dev environment.'
	@echo 'See: https://github.com/pixeltable/pixeltable/blob/main/CONTRIBUTING.md'
	@echo ''
	@echo 'Targets:'
	@echo '  install       Install the development environment'
	@echo '  test          Run pytest, stresstest, and check'
	@echo '  fulltest      Run fullpytest, nbtest, stresstest, and check'
	@echo '  slimtest      Run a slimpytest and check'
	@echo '  check         Run typecheck, docscheck, lint, and formatcheck'
	@echo '  format        Run `ruff format` (updates .py files in place)'
	@echo '  release       Create a pypi release and post to github'
	@echo '  docs-local    Build documentation for local preview (auto-updates doctools)'
	@echo '  docs-dev      Deploy versioned docs to dev with errors visible (auto-updates doctools)'
	@echo '  docs-stage    Deploy versioned documentation to staging (auto-updates doctools)'
	@echo '  docs-prod     Deploy documentation from staging to production (auto-updates doctools)'
	@echo ''
	@echo 'Individual test targets:'
	@echo '  clean         Remove generated files and temp files'
	@echo '  pytest        Run `pytest`'
	@echo '  fullpytest    Run `pytest`, including expensive tests'
	@echo '  slimpytest    Run `pytest` with a minimal set of tests'
	@echo '  nbtest        Run `pytest` on notebooks'
	@echo '  stresstest    Run stress tests such as random-tbl-ops'
	@echo '  typecheck     Run `mypy`'
	@echo '  docscheck     Run `mkdocs build --strict`'
	@echo '  lint          Run `ruff check`'
	@echo '  formatcheck   Run `ruff format --check` (check only, do not modify files)'
	@echo ''
	@echo 'Parameters:'
	@echo '  UV_ARGS       Additional arguments to pass to `uv sync`'
	@echo '  VERSION       Version to use for docs deployment'
	@echo ''
	@echo 'Example commands:'
	@echo '  make install UV_ARGS=--no-dev   Switch to minimal Pixeltable installation (no dev packages)'
	@echo '  make test UV_ARGS=--no-dev      Run tests with minimal Pixeltable installation'

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

# Environment installation, prior to running `uv sync`
.make-install/env:
	@echo 'Installing uv ...'
	@python -m pip install -qU pip
	@python -m pip install -q uv==0.9.3
	@echo 'Installing conda packages ...'
	@conda install -q -y -c conda-forge libiconv 'ffmpeg==6.1.1=gpl*' quarto nodejs
	@echo 'Installing mintlify ...'
	@npm install --silent -g @mintlify/cli
	@echo 'Fixing quarto conda packaging bugs ...'
	@mkdir -p $(CONDA_PREFIX)/bin/tools/aarch64 2>/dev/null || true
	@ln -sf $(CONDA_PREFIX)/bin/deno $(CONDA_PREFIX)/bin/tools/aarch64/deno 2>/dev/null || true
	@ln -sf $(CONDA_PREFIX)/bin/pandoc $(CONDA_PREFIX)/bin/tools/aarch64/pandoc 2>/dev/null || true
	@for dir in $(CONDA_PREFIX)/share/quarto/*/; do \
		target=$$(basename $$dir); \
		ln -sf $$dir $(CONDA_PREFIX)/share/$$target 2>/dev/null || true; \
	done
	@$(TOUCH) .make-install/env

.PHONY: install-deps
install-deps:
	@echo 'Installing dependencies from uv ...'
	@$(SET_ENV) VIRTUAL_ENV="$(CONDA_PREFIX)"; uv sync --active $(UV_ARGS)

# After running `uv sync`
.make-install/others:
	@echo 'Installing Jupyter kernel ...'
	@python -m ipykernel install --user --name=$(KERNEL_NAME)
	@$(TOUCH) .make-install/others

.PHONY: install
install: setup-install .make-install/env install-deps .make-install/others

.PHONY: test
test: pytest check
	@echo 'All tests passed.'

.PHONY: fulltest
fulltest: fullpytest nbtest check
	@echo 'All tests passed.'

.PHONY: slimtest
slimtest: slimpytest check
	@echo 'All tests passed.'

.PHONY: check
check: typecheck docscheck lint formatcheck
	@echo 'All static checks passed.'

.PHONY: pytest
pytest: install
	@echo 'Running `pytest` ...'
	@$(ULIMIT_CMD) pytest $(PYTEST_COMMON_ARGS) tests

.PHONY: fullpytest
fullpytest: install
	@echo 'Running `pytest`, including expensive tests ...'
	@$(ULIMIT_CMD) pytest $(PYTEST_COMMON_ARGS) -m '' tests

.PHONY: slimpytest
slimpytest: install
	@echo 'Running `pytest` on a slim configuration ...'
	@$(ULIMIT_CMD) pytest $(PYTEST_COMMON_ARGS) tests/test_{catalog,dirs,env,exprs,function,index,snapshot,table,view}.py

.PHONY: nbtest
nbtest: install
	@echo 'Running `pytest` on notebooks ...'
	@$(SHELL_PREFIX) scripts/prepare-nb-tests.sh --no-pip tests/target/nb-tests docs/notebooks tests
	@$(ULIMIT_CMD) pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) tests/target/nb-tests/*.ipynb

.PHONY: stresstest
stresstest: install
	@$(SHELL_PREFIX) scripts/stress-tests.sh

.PHONY: typecheck
typecheck: install
	@echo 'Running `mypy` ...'
	@mypy pixeltable tests tool

.PHONY: docscheck
docscheck: install
	@echo 'Running `mkdocs build --strict` ...'
	@python -W ignore::DeprecationWarning -m mkdocs build --strict
	@echo 'Running `pydoclint` ...'
	@pydoclint -q pixeltable tests tool

.PHONY: lint
lint: install
	@echo 'Running `ruff check` ...'
	@ruff check pixeltable tests tool

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

# Shared target to update doctools (with force-reinstall to bypass pip/git caches)
.PHONY: update-doctools
update-doctools:
	@echo 'Updating pixeltable-doctools...'
	@python -m pip uninstall -y -q pixeltable-doctools 2>/dev/null || true
	@python -m pip install -q --upgrade --no-cache-dir --force-reinstall --no-deps git+https://github.com/pixeltable/pixeltable-doctools.git

.PHONY: docs-local
docs-local: install update-doctools
	@echo 'Building documentation for local preview...'
	@python -m doctools.build_mintlify.build_mintlify
	@echo ''
	@echo 'Documentation built successfully!'
	@echo 'To preview, run: cd $(CURDIR)/docs/target && npx mintlify dev'

.PHONY: docs-dev
docs-dev: install update-doctools
	@echo 'Building and deploying documentation to dev for pre-release validation (with errors visible)...'
	@python -m doctools.deploy.deploy_docs_dev

.PHONY: docs-stage
docs-stage: install update-doctools
	@test -n "$(VERSION)" || (echo "ERROR: VERSION required. Usage: make docs-stage VERSION=0.4.17" && exit 1)
	@echo 'Building and deploying documentation for $(VERSION) to staging...'
	@python -m doctools.deploy.deploy_docs_stage --version=$(VERSION)

.PHONY: docs-prod
docs-prod: install update-doctools
	@echo 'Deploying documentation from stage to production...'
	@echo 'This will completely replace production with staging content.'
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ] || (echo "Deployment cancelled." && exit 1)
	@python -m doctools.deploy.deploy_docs_prod

.PHONY: clean
clean:
	@$(RM) *.mp4 docs/source/tutorials/*.mp4 || true
	@$(RMDIR) .make-install || true
	@$(RMDIR) site || true
	@$(RMDIR) target || true
	@$(RMDIR) tests/target || true
	@$(RMDIR) docs/target || true
