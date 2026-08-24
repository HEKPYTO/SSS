CXX ?= c++
CXXFLAGS = -std=c++17 -O2 -funroll-loops -ffp-contract=off -DNDEBUG

ssffs: anc/ssffs/ssffs.cpp
	$(CXX) $(CXXFLAGS) -o ssffs anc/ssffs/ssffs.cpp

verify: ssffs
	@echo "sha256 reuters_apte.arff: $$(shasum -a 256 anc/data/reuters_apte.arff | cut -d' ' -f1)"
	@test "$$(shasum -a 256 anc/data/reuters_apte.arff | cut -d' ' -f1)" = "4b22e0e94f53993595f5fa80c7eca0b5dbda0ec80423ac2e31861e156ea1834a" || (echo "SHA MISMATCH" && false)
	./ssffs --help | head -n 5

lint:
	ruff check
	ruff format --check

format:
	ruff format
	ruff check --fix

typecheck:
	mypy src

check: lint typecheck test

test: ssffs
	pytest -v

clean:
	rm -f ssffs /tmp/ssffs
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
