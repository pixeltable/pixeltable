# Parameter defaults
DURATION := 120
UV_ARGS := --group extra-dev
WORKERS := 12

# Common test args
PYTEST_COMMON_ARGS := -v -n auto --dist loadgroup --maxprocesses 6 --reruns 2 \
	--only-rerun 'That Pixeltable operation could not be completed because it conflicted with'

# Needed for LLaMA build to work correctly on some Linux systems
CMAKE_ARGS := -DLLAVA_BUILD=OFF
NB_CELL_TIMEOUT := 3600
# We ensure the TQDM progress bar is updated exactly once per cell execution, by setting the refresh rate equal to the timeout
TQDM_MININTERVAL := $(NB_CELL_TIMEOUT)
ULIMIT_CMD := ulimit -n 4000;

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
	@echo '  check         Run typecheck, lint, and formatcheck'
	@echo '  format        Run `ruff format` (updates .py files in place)'
	@echo '  release       Create a pypi release and post to github'
	@echo '  docs          Build mintlify documentation'
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
	@echo '  stresstest    Run stress tests such as random-ops'
	@echo '  typecheck     Run `mypy`'
	@echo '  docscheck     Run `mkdocs build --strict`'
	@echo '  lint          Run `ruff check`'
	@echo '  formatcheck   Run `ruff format --check` (check only, do not modify files)'
	@echo ''
	@echo 'Global parameters:'
	@echo '  UV_ARGS       Additional arguments to pass to `uv sync` (default = '\''--group extra-dev'\'')'
	@echo ''
	@echo '`make stresstest` parameters:'
	@echo '  DURATION      Duration in seconds for stress tests (default = 120)'
	@echo '  WORKERS       Number of workers for stress tests (default = 12)'
	@echo ''
	@echo '`make docs-deploy` parameters:'
	@echo "  TARGET        Deployment target ('dev', 'stage', or 'prod')"
	@echo ''
	@echo 'Example command lines:'
	@echo '  make install UV_ARGS=--no-dev   Switch to minimal Pixeltable installation (no dev packages)'
	@echo '  make test UV_ARGS=--no-dev      Run tests with minimal Pixeltable installation'
	@echo '  make stresstest DURATION=7200   Run stress tests for 2 hours'
	@echo '  make docs-deploy TARGET=dev     Deploy docs to dev environment'

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

# Environment installation, prior to running `uv sync`
.make-install/env:
	@echo 'Installing uv ...'
	@python -m pip install -qU pip
	@python -m pip install -q uv==0.9.3
	@echo 'Installing conda packages ...'
	@if ! which mamba >/dev/null 2>&1; then conda install -q -y -c conda-forge mamba; fi
	@mamba install -q -y -c conda-forge libiconv 'ffmpeg==6.1.1=gpl*' quarto nodejs lychee
	@echo 'Installing mintlify ...'
	@if ! which mint >/dev/null 2>&1; then npm install --silent -g @mintlify/cli; fi
	@echo 'Fixing quarto conda packaging bugs ...'
	@mkdir -p $(CONDA_PREFIX)/bin/tools/aarch64 2>/dev/null || true
	@ln -sf $(CONDA_PREFIX)/bin/deno $(CONDA_PREFIX)/bin/tools/aarch64/deno 2>/dev/null || true
	@ln -sf $(CONDA_PREFIX)/bin/pandoc $(CONDA_PREFIX)/bin/tools/aarch64/pandoc 2>/dev/null || true
	@for dir in $(CONDA_PREFIX)/share/quarto/*/; do \
		target=$$(basename $$dir); \
		ln -sf $$dir $(CONDA_PREFIX)/share/$$target 2>/dev/null || true; \
	done
	@touch .make-install/env

.PHONY: install-deps
install-deps:
	@echo 'Installing dependencies from uv ...'
	@touch pyproject.toml
	@VIRTUAL_ENV="$(CONDA_PREFIX)" uv sync --active $(UV_ARGS)

# After running `uv sync`
.make-install/others:
	@echo 'Installing Jupyter kernel ...'
	@python -m ipykernel install --user --name=pixeltable
	@touch .make-install/others

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
check: typecheck lint formatcheck
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
	@$(ULIMIT_CMD) pytest $(PYTEST_COMMON_ARGS) tests/test_{catalog,dirs,env,exprs,function,index,snapshot,table,view}.py tests/share/test_packager.py

.PHONY: nbtest
nbtest: install
	@echo 'Running `pytest` on notebooks ...'
	@scripts/prepare-nb-tests.sh --no-pip tests/target/nb-tests docs/release tests
	@$(ULIMIT_CMD) pytest -v --nbmake --nbmake-timeout=$(NB_CELL_TIMEOUT) --nbmake-kernel=$(KERNEL_NAME) tests/target/nb-tests/*.ipynb

.PHONY: stresstest
stresstest: install
	@scripts/stress-tests.sh $(WORKERS) $(DURATION)

.PHONY: typecheck
typecheck: install
	@echo 'Running `mypy` ...'
	@mypy pixeltable tests tool

.PHONY: docscheck
docscheck: install
	@echo 'Running `mkdocs build --strict` ...'
	@python -W ignore::DeprecationWarning -m mkdocs build --strict

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
	@scripts/release.sh

.PHONY: docs
docs: install
	VIRTUAL_ENV="$(CONDA_PREFIX)" uv sync --active $(UV_ARGS) --upgrade-package pixeltable-doctools
	@python -m pixeltable_doctools.build
	@cd target/docs && mintlify broken-links || true

.PHONY: docs-serve
docs-serve: docs
	@cd target/docs && mintlify dev

.PHONY: docs-deploy
docs-deploy: docs
ifdef TARGET
	@git fetch https://github.com/pixeltable/pixeltable --tags --force
	@python -m pixeltable_doctools.deploy $(TARGET)
else
	$(error Usage: make docs-deploy TARGET=<dev|stage|prod>)
endif

# TODO: incorporate this into a new/expanded docscheck
.PHONY: linkscheck
linkscheck: docs
	lychee target/docs/ --exclude-path target/docs/changelog/ --max-concurrency 3 --exclude 'file://*' --exclude-loopback -q

.PHONY: clean
clean:
	@rm -f *.mp4 docs/source/tutorials/*.mp4 || true
	@rm -rf .make-install || true
	@rm -rf site || true
	@rm -rf target || true
	@rm -rf tests/target || true
