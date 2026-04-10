.PHONY: install test lint format typecheck

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ --cov=graph_diffusion --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/graph_diffusion
