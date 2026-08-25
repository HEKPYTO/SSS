CXX ?= c++
CXXFLAGS = -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG

.PHONY: verify lint typecheck test package check

verify:
	@test "$$(shasum -a 256 anc/ssffs/ssffs.cpp | cut -d' ' -f1)" = "acf6d17e01238fead9af18dfbe2f502e000df6ec372272daeb3ed99c286a9a24"
	@test "$$(shasum -a 256 anc/data/reuters_apte.arff | cut -d' ' -f1)" = "4b22e0e94f53993595f5fa80c7eca0b5dbda0ec80423ac2e31861e156ea1834a"
	$(CXX) $(CXXFLAGS) -o /tmp/ssffs anc/ssffs/ssffs.cpp
	/tmp/ssffs --help >/dev/null

lint:
	python3 -m ruff check src tests
	python3 -m ruff format --check src tests

typecheck:
	python3 -m mypy src

test:
	PYTHONPATH=src python3 -m pytest -q

package:
	python3 -m pip wheel . --no-deps --wheel-dir /tmp/sss-wheel
	python3 -m pytest tests/test_package.py -q

check: verify lint typecheck test package
