help:
	@echo "available commands"
	@echo "-------------------------------------------------"
	@echo "install       : installs all dependencies"
	@echo "build         : build project"
	@echo "clean         : cleans up artifacts in project"
	@echo "run           : run gunicorn service"
	@echo "run           : test"
	@echo "-------------------------------------------------"

install: requirements.txt
	pip install -r requirements.txt --no-cache

build:
	# python setup.py build
	# mv BKTree.cpython-36m-x86_64-linux-gnu.so qa/tools/bktree/
	
clean:
	rm -rf __pycache__
	rm -rf build

run:
	bash ./scripts/start.sh

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/ --junitxml="tests.xml"