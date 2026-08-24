CXX ?= c++
CXXFLAGS = -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG

.PHONY: help ssffs verify lint format typecheck check test clean dev install pre-commit

help:
	@echo "Targets:"
	@echo "  ssffs        Build C++ reference (bit-identical flags)"
	@echo "  verify       SHA + help check"
	@echo "  dev          Install dev deps (pip -e .[dev] + pre-commit)"
	@echo "  lint         ruff check + format --check"
	@echo "  format       ruff format + ruff check --fix"
	@echo "  typecheck    mypy src"
	@echo "  check        lint + typecheck + test"
	@echo "  test         pytest (needs ssffs binary)"
	@echo "  pre-commit   install hooks"
	@echo "  clean        remove build artifacts"

ssffs: anc/ssffs/ssffs.cpp
	$(CXX) $(CXXFLAGS) -o ssffs anc/ssffs/ssffs.cpp

verify: ssffs
	@echo "sha256 reuters_apte.arff: $$(shasum -a 256 anc/data/reuters_apte.arff | cut -d' ' -f1)"
	@test "$$(shasum -a 256 anc/data/reuters_apte.arff | cut -d' ' -f1)" = "4b22e0e94f53993595f5fa80c7eca0b5dbda0ec80423ac2e31861e156ea1834a" || (echo "SHA MISMATCH" && false)
	./ssffs --help | head -n 5

dev:
	pip install -e ".[dev]"
	pre-commit install --install-hooks

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

check: lint typecheck test

test: ssffs
	pytest -v

pre-commit:
	pre-commit install

clean:
	rm -f ssffs /tmp/ssffs
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
