.PHONY: install test build clean

install:
	python -m pip install -e ".[dev,convert]"

test:
	python -m pytest

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
