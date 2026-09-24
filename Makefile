.PHONY: install lint test fetch train monitor serve up down clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest --cov --cov-report=term-missing

fetch:
	mlplatform fetch

train:
	mlplatform train

monitor:
	mlplatform monitor

serve:
	mlplatform serve --reload

up:
	docker compose up --build

down:
	docker compose down -v

clean:
	rm -rf artifacts data/predictions.parquet .pytest_cache .ruff_cache
