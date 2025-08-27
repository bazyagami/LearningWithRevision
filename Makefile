ENV_NAME ?= pdd
ENV_FILE ?= environment.yml

.PHONY: help env smoke clean

help:
	@echo "make env    - create/update conda env '$(ENV_NAME)' from $(ENV_FILE)"
	@echo "make clean  - remove __pycache__ and *.pyc"

env:
	conda env create -n $(ENV_NAME) -f $(ENV_FILE) || \
	conda env update -n $(ENV_NAME) -f $(ENV_FILE)
	@echo "Activate with: conda activate $(ENV_NAME)"

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + \
	 -o -name "*.pyc" -delete
