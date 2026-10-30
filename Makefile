.DEFAULT_GOAL := check
.PHONY: install fmt lint types imports test check verify

install:  ## Sync the dev environment
	uv sync --locked --extra dev

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

test:  ## Run the fast offline CPU tests
	uv run --locked python -m pytest -n 4

check: lint types imports test  ## The fast local and CI gate

verify:  ## Run the slow offline tests (statistical fits)
	uv run --locked python -m pytest -m slow
