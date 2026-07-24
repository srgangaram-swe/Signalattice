# Signalattice — developer convenience targets.
# Run `make help` for the list.

.DEFAULT_GOAL := help
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
CONFIG ?= configs/example.yaml

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: venv
venv: ## Create a virtual environment in .venv
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

.PHONY: install
install: ## Install the package (editable) with dev extras
	$(BIN)/python -m pip install -e ".[dev]"

.PHONY: install-all
install-all: ## Install every optional integration, including PyTorch
	$(BIN)/python -m pip install -e ".[dev,data,boost,torch,mlflow]"

.PHONY: lint
lint: ## Run ruff + black --check
	$(BIN)/ruff check src tests scripts
	$(BIN)/black --check src tests scripts

.PHONY: format
format: ## Auto-format with ruff --fix and black
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/black src tests scripts

.PHONY: typecheck
typecheck: ## Run mypy static type checks
	$(BIN)/mypy src

.PHONY: quality
quality: lint typecheck ## Run all static quality gates

.PHONY: test
test: ## Run the test suite (skips network tests)
	$(BIN)/pytest -m "not network"

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(BIN)/pytest -m "not network" --cov=quant_platform --cov-branch \
		--cov-report=term-missing --cov-report=xml --cov-fail-under=80

.PHONY: build
build: ## Build the source distribution and wheel
	$(BIN)/python -m build

.PHONY: ci
ci: quality test-cov build ## Reproduce the local CI quality, test, and package gates

.PHONY: ingest
ingest: ## Ingest market data using $(CONFIG)
	$(BIN)/signalattice ingest-data --config $(CONFIG)

.PHONY: features
features: ## Build features using $(CONFIG)
	$(BIN)/signalattice build-features --config $(CONFIG)

.PHONY: train
train: ## Train models using $(CONFIG)
	$(BIN)/signalattice train-model --config $(CONFIG)

.PHONY: backtest
backtest: ## Run backtest using $(CONFIG)
	$(BIN)/signalattice run-backtest --config $(CONFIG)

.PHONY: report
report: ## Generate report using $(CONFIG)
	$(BIN)/signalattice generate-report --config $(CONFIG)

.PHONY: pipeline
pipeline: ## Run the full end-to-end pipeline using $(CONFIG)
	$(BIN)/signalattice run-full-pipeline --config $(CONFIG)

.PHONY: demo
demo: ## Run a fully offline demo (synthetic data) end-to-end
	$(BIN)/signalattice run-full-pipeline --config configs/synthetic.yaml

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-data
clean-data: ## Remove generated data, reports, experiments (keeps .gitkeep)
	find data -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	rm -rf reports/figures/*.png reports/*.md reports/*.html experiments *.sqlite 2>/dev/null || true

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build --pull -t signalattice:latest .

.PHONY: docker-demo
docker-demo: ## Run the synthetic demo inside Docker
	docker compose run --rm platform run-full-pipeline --config configs/synthetic.yaml
