CXX ?= c++
PYTHON ?= python3
CXXFLAGS = -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG -Wall -Wextra -Wpedantic -Werror

.PHONY: verify lint typecheck test check

verify:
	@test "$$(shasum -a 256 anc/ssffs/ssffs.cpp | cut -d' ' -f1)" = "acf6d17e01238fead9af18dfbe2f502e000df6ec372272daeb3ed99c286a9a24"
	@test "$$(shasum -a 256 anc/data/reuters_apte.arff | cut -d' ' -f1)" = "4b22e0e94f53993595f5fa80c7eca0b5dbda0ec80423ac2e31861e156ea1834a"
	$(CXX) $(CXXFLAGS) -o /tmp/ssffs anc/ssffs/ssffs.cpp
	/tmp/ssffs --help >/dev/null

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

check: lint typecheck verify test
