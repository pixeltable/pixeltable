SHELL := /bin/bash
KERNEL_NAME := $(shell basename `pwd`)
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  install    Install the development environment"
	@echo "  test       Run pytests"
	@echo "  notebook   Run notebooks (updates output cells in place)"
	@echo "  typecheck  Run type checks"
	@echo "  clean      Clean up re-generatable files, as well as temp files"

poetry.lock: pyproject.toml
	@poetry lock --no-update

# use a file to track whether the install has been run
# avoid re-running the install if install has been done and deps 
# haven't changed
.make-install: poetry.lock
	@echo "Installing development environment..."
	@poetry install -E s3 -E openai -E torch -E nos --with=dev
	@poetry run python -m ipykernel install --user --name=$(KERNEL_NAME)
	@touch .make-install

.PHONY: install
install: .make-install

.PHONY: test
test: install
	@echo "Running pytests..."
# on my Mac at least, the default ulimit is too low for the number of open files
	@ulimit -n 4000; poetry run pytest

.PHONY: notebooks
notebooks: install
	@echo "Running notebooks and overwriting outputs..."
	@poetry run pytest --overwrite --nbmake --nbmake-kernel=$(KERNEL_NAME) docs/source/tutorials/*.ipynb

.PHONY: typecheck
typecheck: install
	@echo "Running type checks..."
	@poetry run mypy --config-file mypy.ini --package pixeltable

# does not remove the poetry.lock
.PHONY: clean
clean:
	@rm -f *.mp4 docs/source/tutorials/*.mp4 || true
	@rm -f .make-install || true
	@rm -rf site || true

.PHONY: build-docs
build-docs: install
	@echo "Building docs..."
	@poetry run mkdocs build  

.PHONY: deploy-docs
deploy-docs: install
	@echo "Builds and publishes docs to origin/gh-pages"
	@poetry run mkdocs gh-deploy --force

