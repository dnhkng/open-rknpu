# Developer tasks. `make help` lists the targets.
PYTHON ?= python3
PYTHONPATH := src
RUFF ?= ruff
MAINTAINED := src tests examples research/verify_suites.py research/campaign_sweep.py \
              research/check_docs_links.py research/probe_conv_envelope.py research/build_mel_kws_suite.py

.DEFAULT_GOAL := help

.PHONY: help test lint test-one primitives baseline campaign docs-check wheel runtime clean

help:  ## list the targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test:  ## run the host test suite
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

lint:  ## ruff on the maintained paths
	$(RUFF) check $(MAINTAINED)

test-one:  ## run one test module, e.g. make test-one M=tests.test_walk
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest $(M)

primitives:  ## run every low-level op example (host only)
	@for script in examples/primitives/[0-9]*.py; do \
	  echo "== $$script"; PYTHONPATH=$(PYTHONPATH) $(PYTHON) $$script || exit 1; \
	done

baseline:  ## recompile all 2,244 suite models against the checked-in baseline
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research/verify_suites.py

campaign:  ## recompile the campaign suites and diff the published containers
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research/campaign_sweep.py

docs-check:  ## relative links and heading anchors across every markdown file
	$(PYTHON) research/check_docs_links.py

wheel:  ## build the wheel and sdist
	$(PYTHON) -m build --out-dir dist

runtime:  ## build the host runtime with the system compiler (sanity check only)
	$(MAKE) -C runtime

clean:  ## remove build outputs and caches
	rm -rf build dist .pytest_cache .ruff_cache `find . -name __pycache__ -type d`
	$(MAKE) -C runtime clean
