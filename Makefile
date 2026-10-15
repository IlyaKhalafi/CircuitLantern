.PHONY: install install-cuda dev backend web test test-ui build format check

TORCH_BACKEND ?= auto

install:
	cd backend && uv sync --group dev
	cd web && npm ci

# Select a wheel using the installed driver, or set TORCH_BACKEND=cu128 explicitly.
install-cuda: install
	cd backend && uv pip install --python .venv/bin/python --upgrade --torch-backend=$(TORCH_BACKEND) "torch>=2.6,<3"
	cd backend && .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA device found after installation"'

dev:
	bash scripts/dev.sh

backend:
	cd backend && uv run --no-sync uvicorn circuit_lantern.main:app --host 127.0.0.1 --port 8000

web:
	cd web && npm run dev

test:
	cd backend && uv run --no-sync pytest -q

test-ui:
	bash scripts/test-ui.sh

check:
	cd backend && uv run --no-sync ruff check circuit_lantern tests
	cd backend && uv run --no-sync ruff format --check circuit_lantern tests
	cd web && npm run typecheck
	cd web && npm run format:check

format:
	cd backend && uv run --no-sync ruff format circuit_lantern tests
	cd web && npm run format

build:
	cd web && npm run build
