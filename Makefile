help:
	@echo "available commands"
	@echo "-------------------------------------------------"
	@echo "install       : installs all dependencies"
	@echo "clean         : cleans up artifacts in project"
	@echo "-------------------------------------------------"

install: requirements.txt
	pip install -r requirements.txt

clean:
	rm -r models/*.tar.gz
	rm -rf __pycache__