help:
	@echo "available commands"
	@echo "-------------------------------------------------"
	@echo "install       : installs all dependencies"
	@echo "build         : build project"
	@echo "clean         : cleans up artifacts in project"
	@echo "run           : run gunicorn service"
	@echo "-------------------------------------------------"

install: requirements.txt
	pip install -r requirements.txt

build:
	python setup.py build
	mv BKTree.cpython-36m-x86_64-linux-gnu.so qa/tools/bktree/
clean:
	rm -rf __pycache__
	rm -rf build

run:
	gunicorn -c conf.py app:app