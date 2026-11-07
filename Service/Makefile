.PHONY: help install run test train docker-build docker-run benchmark

VENV ?= ../.venv/bin
PYTHON ?= $(VENV)/python3
PIP ?= $(VENV)/pip

help:
	@echo "Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make train         Run training and export artifacts"
	@echo "  make test          Run pytest test suite"
	@echo "  make run           Run FastAPI service locally with uvicorn"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make benchmark     Run latency and RPS benchmark against live service"

install:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

train:
	PYTHONPATH=. $(PYTHON) src/train.py --data-path ../Research/ctr_task_dataset.csv --output-dir artifacts

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

run:
	PYTHONPATH=. $(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build --network=host -t ctr-rtb-service:latest .

docker-run:
	docker run -d --name ctr_service -p 8000:8000 ctr-rtb-service:latest

benchmark:
	PYTHONPATH=. $(PYTHON) scripts/benchmark.py --requests 300

