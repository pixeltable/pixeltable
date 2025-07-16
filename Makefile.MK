PROJECT_DIR := $(shell pwd)
POETRY ?= $(shell command -v poetry 2>/dev/null)

PYTHON_VERSION := 3.12

# If poetry is not installed, install it
ifeq (, $(POETRY))
$(warning Poetry not found. Installing Poetry...)
install-poetry:
	curl -sSL https://install.python-poetry.org | python3 -
else
install-poetry:
	@echo "Poetry is already installed."
endif

.PHONY: install
install: install-poetry
	poetry env use python$(PYTHON_VERSION)
	poetry install

.PHONY: format
format:
	poetry run black .

.PHONY: lint
lint:
	poetry run flake8 .

.PHONY: clean
clean:
	rm -rf .venv
	poetry env remove python$(PYTHON_VERSION) || true