.PHONY: install run fast test lint clean

install:
	pip install -e ".[dev]"

run:            ## full pipeline: clean -> validate -> train -> evaluate -> reports
	python -m hotel_ml.pipeline

fast:           ## smoke run with a tiny search grid
	python -m hotel_ml.pipeline --fast

test:
	pytest -q

lint:
	ruff check src tests

clean:
	rm -rf data/processed/* models/* .pytest_cache
