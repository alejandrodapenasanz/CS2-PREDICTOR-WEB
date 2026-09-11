PYTHON ?= python3.13
LOCKFILE ?= requirements.lock.txt
WHEEL_DIR ?= wheels
PYTEST = $(PYTHON) -m pytest
RUFF = $(PYTHON) -m ruff
MYPY = $(PYTHON) -m mypy
LOCK_INSTALL = $(PYTHON) -m pip install --no-deps --only-binary=:all: --require-hashes --find-links "$(WHEEL_DIR)" -r "$(LOCKFILE)"

.PHONY: install verify-wheel lint format-check typecheck imports test smoke gates test-working test-cov test-unit test-integration test-fast test-offline test-slow clean test-one

# Install only the fully pinned, hashed lock under CPython 3.13.
install: verify-wheel
	$(LOCK_INSTALL)
	$(PYTHON) -m pip check

# Verify the one allowlisted local wheel without contacting a package index.
verify-wheel:
	$(PYTHON) scripts/verify_flasgger_wheel.py

# Fast, deterministic local/CI quality gates.
lint:
	$(RUFF) check .

format-check:
	$(RUFF) format --check .

typecheck:
	$(MYPY)

imports:
	$(PYTHON) scripts/check_import_coverage.py

smoke:
	$(PYTHON) scripts/smoke_boot.py

gates: lint format-check typecheck imports test-offline smoke

# Run all tests (unit + integration).
test:
	$(PYTEST) tests/ -v

# Run only unit tests (fast, with mocks).
test-unit:
	$(PYTEST) tests/test_routes.py -v

# Run only integration tests (slow, real HLTV connections).
test-integration:
	$(PYTEST) tests/test_integration_real.py -v

# Run only fast tests (exclude slow integration tests).
test-fast:
	$(PYTEST) -m "not slow and not integration" tests/ -v

# Deterministic alias used by the quality gate (never contacts HLTV).
test-offline: test-fast

# Run only slow tests (integration tests).
test-slow:
	$(PYTEST) -m "slow" tests/ -v

# Run tests with coverage report.
test-cov:
	$(PYTEST) --cov=routes --cov-report=html --cov-report=term tests/

# Clean cache and temporary test files.
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -name "*.pyc" -delete
	rm -rf htmlcov/

# Run one named endpoint test.
test-one:
	$(PYTEST) tests/test_routes.py::TestRoutesEndpoints::$(TEST) -v
