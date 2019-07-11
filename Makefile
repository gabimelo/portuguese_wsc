.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 tests

install_flake8:
	python3 -m pip install flake8

config_githooks:
	git config core.hooksPath githooks
	chmod +x githooks/pre-commit

download_nltk_data:
	python -c "import nltk; nltk.download('punkt')"

create_required_dirs:
	mkdir -p models/english-wikitext-2/trained_models
	mkdir -p models/trained_models

dev_init: config_githooks install_flake8 download_nltk_data create_required_dirs processed_data

## Code Testing

tests:
	pytest --cov=src tests/

## Run Code

train:
	python -m src.main --training

winograd_test:
	python -m src.main

generate:
	python -m src.main --generating

## Make Dataset

corpus: corpus_dictionary
	python -m src.datasets_manipulation_scripts.make_corpus_pickle

corpus_dictionary: processed_data
	python -m src.datasets_manipulation_scripts.make_corpus_dictionary_pickle

processed_data: interim_data
	python -m src.datasets_manipulation_scripts.make_processed_dataset

recuced_processed_data: processed_data
	./src/datasets_manipulation_scripts/make_reduced_processed_dataset.sh

interim_data: download_dev_data
	python -m src.datasets_manipulation_scripts.make_interim_dataset

interim_data_without_splits: download_dev_data
	python -m src.datasets_manipulation_scripts.make_interim_dataset --split False

download_dev_data: wikidump

wikidump:
	./get_wikidump.sh

nilc_corpora:
	./get_nilc_corpora.sh

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

clean_ipynb_checkpoints:
	find . -type d -name ".ipynb_checkpoints" -exec rm -rv {} +

## Lint using flake8
lint: lint_src lint_tests

lint_src:
	flake8 src

lint_tests:
	flake8 tests
