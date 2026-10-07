install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv test_*.py

format:	
	black *.py src/*.py

lint:
	pylint --disable=R,C app.py --ignore-patterns=test_.*?py *.py src/*.py

all: install lint test format