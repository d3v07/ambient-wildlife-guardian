.PHONY: install run test destroy clean

install:
	uv sync

run:
	uv run uvicorn app.server:app --port 8080 --reload

test:
	uv run pytest tests/

destroy: clean
	rm -rf .venv

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .venv uv.lock
