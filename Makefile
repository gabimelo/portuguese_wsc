.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 tests

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = portuguese_wsc
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

install_flake8:
	python3 -m pip install flake8

config_githooks:
	git config core.hooksPath githooks
	chmod +x githooks/pre-commit

dev_init: download_dev_data config_githooks install_flake8

## Code Testing
tests:
	pytest --cov=src tests/

## Run Code
train:
	python -m src.main

## Make Dataset
corpus_dictionary:
	python -m src.scripts.make_corpus_dictionary_pickle

processed_data:
	python -m src.scripts.make_processed_dataset

interim_data:
	python -m src.scripts.make_interim_dataset

interim_data_without_splits:
	python -m src.scripts.make_interim_dataset --split False

download_dev_data: wikidump

wikidump:
	./get_wikidump.sh

nilc_corpora:
	./get_nilc_corpora.sh

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint: lint_src lint_tests

lint_src:
	flake8 src

lint_tests:
	flake8 tests
