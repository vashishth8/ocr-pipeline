PYTHON ?= python3.11
SRC := src
TYPECHECK := $(SRC)/sttl/benchmark.py $(SRC)/sttl/gates.py $(SRC)/sttl/geometry.py $(SRC)/sttl/jsonio.py $(SRC)/sttl/metrics.py $(SRC)/sttl/metrics_cli.py $(SRC)/sttl/reporting.py $(SRC)/sttl/text.py $(SRC)/sttl/version.py sttl_compat.py

.PHONY: install test lint format typecheck coverage check

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install --no-deps --no-build-isolation -e .

test:
	PYTHONPATH=$(SRC) $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy $(TYPECHECK)

coverage:
	PYTHONPATH=$(SRC) $(PYTHON) -m coverage run -m pytest
	$(PYTHON) -m coverage report
	$(PYTHON) -m coverage xml

check: lint format typecheck test
