.DEFAULT_GOAL := check
.PHONY: install fix lint types imports test check slow verify

install:  ## Sync the dev environment
	uv sync --locked --extra dev

fix:  ## Apply every auto-fixer: ruff format and safe lint fixes
	uv run --locked ruff format .
	uv run --locked ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run --locked ruff check .
	uv run --locked ruff format --check .

types:  ## Strict type-check src, scripts, and tests
	uv run --locked basedpyright

imports:  ## Check domain dependency contracts
	uv run --locked lint-imports

test:  ## Run the fast offline CPU tests
	uv run --locked python -m pytest -n 4

check:  ## The fast local and CI gate: lint, types, imports, and tests in parallel
	@$(MAKE) -j4 lint types imports test

slow:  ## Run the slow offline tests (statistical fits)
	uv run --locked python -m pytest -m slow

verify:  ## Drive the real training and launch entrypoints offline (ARGS=all|list|control|<drive>)
	.agents/skills/verify-esme-pretrain/verify $(ARGS)
