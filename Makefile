install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

lint:
	flake8 mini_project_1.py

test:
	python -m pytest -vv --cov=hello test_mini_project_1.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test
