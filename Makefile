format:
	black . -l 79
	isort .

lint:
	flake8 --exclude '*venv,models/bert.py'

tests:
	$(PYTHON) -m pytest tests/
