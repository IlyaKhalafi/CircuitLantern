.PHONY: install dev backend web test build format check

install:
	cd backend && uv sync --group dev
	cd web && npm ci

dev:
	bash scripts/dev.sh

backend:
	cd backend && uv run uvicorn circuit_lantern.main:app --host 127.0.0.1 --port 8000

web:
	cd web && npm run dev

test:
	cd backend && uv run pytest -q

check:
	cd backend && uv run ruff check circuit_lantern tests
	cd backend && uv run ruff format --check circuit_lantern tests
	cd web && npm run typecheck

format:
	cd backend && uv run ruff format circuit_lantern tests
	cd web && npm run format

build:
	cd web && npm run build
