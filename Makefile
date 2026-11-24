# Makefile for PAWS Buddy

# Include .env file if it exists
-include .env
export

# Port determination logic:
# Requirement: Environment variable (PORT or APP_PORT) has highest priority.
# If an environment variable is exported in the shell or .env, use it over command-line override.
# Otherwise, allow overriding via command-line (PORT=...), or fallback to default 8000.
ENV_PORT ?= $(PORT)
FINAL_PORT = $(shell python3 -c 'import os; print(os.getenv("PORT") or "$(PORT)" or "8000")')
HOST ?= 127.0.0.1

.PHONY: run tunnel docker-build docker-up docker-down docker-logs help

help:
	@echo "Usage:"
	@echo "  make run             Run app on default port (8000) or PORT from environment/.env"
	@echo "  make run PORT=9000   Run app overriding port (if not set in env)"
	@echo "  PORT=7000 make run   Run app using env variable (highest priority)"
	@echo "  make docker-up       Start container with volume mounting and live reload"
	@echo "  make docker-down     Stop and remove Docker containers"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-logs     Follow container logs"

run:
	uv run uvicorn app:app --host $(HOST) --port $(FINAL_PORT) --reload

docker-build:
	docker compose build

docker-up:
	PORT=$(FINAL_PORT) docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

tunnel:
	cloudflared tunnel --url http://localhost:$(PORT)
