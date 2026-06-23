# SHERLOC Pipeline Development Makefile
# ======================================
#
# Common development tasks for the SHERLOC pipeline.
#
# Usage:
#   make test      - Run all tests
#   make lint      - Run linting (ruff)
#   make clean     - Remove build artifacts and caches
#   make install   - Install package in development mode
#   make help      - Show this help message

.PHONY: test lint clean install help

# Default target
.DEFAULT_GOAL := help

# Run all tests
test:
	@echo "Running tests..."
	python -m pytest tests/

# Run tests with coverage
test-cov:
	@echo "Running tests with coverage..."
	python -m pytest tests/ --cov=sherloc_pipeline --cov-report=term-missing

# Run only unit tests (fast)
test-unit:
	@echo "Running unit tests..."
	python -m pytest tests/unit/ -m "not slow"

# Run only integration tests
test-integration:
	@echo "Running integration tests..."
	python -m pytest tests/integration/ -m integration

# Lint code with ruff (if available)
lint:
	@if command -v ruff >/dev/null 2>&1; then \
		echo "Running ruff..."; \
		ruff check src/ tests/; \
	else \
		echo "ruff not installed. Install with: pip install ruff"; \
	fi

# Format code with ruff (if available)
format:
	@if command -v ruff >/dev/null 2>&1; then \
		echo "Formatting with ruff..."; \
		ruff format src/ tests/; \
	else \
		echo "ruff not installed. Install with: pip install ruff"; \
	fi

# Remove build artifacts and caches
clean:
	@echo "Cleaning build artifacts..."
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf src/*.egg-info build/ dist/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Install package in development mode
install:
	@echo "Installing package in development mode..."
	pip install -e .

# Install with development dependencies
install-dev:
	@echo "Installing package with development dependencies..."
	pip install -e ".[dev]"
	pip install pytest pytest-cov ruff

# Show help
help:
	@echo "SHERLOC Pipeline Development Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make test           - Run all tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make test-unit      - Run only unit tests"
	@echo "  make test-integration - Run only integration tests"
	@echo "  make lint           - Run linting (ruff)"
	@echo "  make format         - Format code (ruff)"
	@echo "  make clean          - Remove build artifacts and caches"
	@echo "  make install        - Install package in development mode"
	@echo "  make install-dev    - Install with development dependencies"
	@echo "  make help           - Show this help message"

