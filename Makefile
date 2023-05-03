format:
	black . -l 79
	isort .

lint:
	flake8 --exclude '*venv'

tests:
	$(PYTHON) -m pytest tests/
