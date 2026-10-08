# Developer workflow. `make check` before every commit.

PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

.PHONY: help check validate compile smoke

help:
	@echo "make check     compile + validate (the pre-commit gate)"
	@echo "make validate  config consistency gate (scripts/validate_config.py)"
	@echo "make compile   syntax-check every .py under src/ and scripts/"
	@echo "make smoke     CLI wiring smoke (--help on both trainers)"

check: compile validate

validate:
	$(PYTHON) scripts/validate_config.py

compile:
	@find src scripts -name '*.py' -print0 | xargs -0 $(PYTHON) -m py_compile
	@echo "compile OK"

smoke:
	$(PYTHON) -m sota_model.training.pretrain --help > /dev/null
	$(PYTHON) scripts/pipelines/03_pretrain.py --help > /dev/null
	@echo "smoke OK"
