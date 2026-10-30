.DEFAULT_GOAL := check
.PHONY: install fmt lint types imports test check

install:  ## Sync the dev environment and activate the repo's git hooks
	uv sync --locked --extra dev
	git config core.hooksPath .githooks

fmt:  ## Auto-format and apply safe lint fixes
	uv run --locked ruff format .
	uv run --locked ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run --locked ruff check .
	uv run --locked ruff format --check .

types:  ## Type-check all production modules and scripts
	uv run --locked ty check

imports:  ## Check domain dependency contracts
	uv run --locked lint-imports

test:  ## Run offline unit and integration tests
	uv run --locked python -m pytest

check: lint types imports test  ## The local and CI gate
