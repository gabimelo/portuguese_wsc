.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 tests

install-flake8:
	python3 -m pip install flake8

config-githooks:
	git config core.hooksPath githooks

download-nltk-data:
	python -c "import nltk; nltk.download('punkt')"

create-required-dirs:
	mkdir -p models/english-wikitext-2/trained_models
	mkdir -p models/trained_models

dev-init: config-githooks install-flake8 download-nltk-data create-required-dirs processed-data

docker-build:
	docker build -t wsc_port .

docker-run = docker run -it wsc_port

## Code Testing

tests:
	pytest --cov=src tests/

docker-tests: docker-build
	$(docker-run) pytest --cov=src tests/

## Run Code

train: docker-build
	$(docker-run) python -m src.main --training

winograd-test: docker-build
	$(docker-run) python -m src.main

generate: docker-build
	$(docker-run) python -m src.main --generating

## Make Dataset

corpus: corpus-dictionary
	python -m src.datasets_manipulation_scripts.make_corpus_pickle

corpus-dictionary: processed-data
	python -m src.datasets_manipulation_scripts.make_corpus_dictionary_pickle

processed-data: interim-data
	python -m src.datasets_manipulation_scripts.make_processed_dataset

recuced-processed-data: processed-data
	./src/datasets_manipulation_scripts/make_reduced_processed_dataset.sh

interim-data: download-dev-data
	python -m src.datasets_manipulation_scripts.make_interim_dataset

interim-data-without-splits: download-dev-data
	python -m src.datasets_manipulation_scripts.make_interim_dataset --split False

download-dev-data: wikidump

wikidump:
	./get_wikidump.sh

nilc_corpora:
	./get_nilc_corpora.sh

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

clean-ipynb-checkpoints:
	find . -type d -name ".ipynb_checkpoints" -exec rm -rv {} +

## Lint using flake8
lint: docker-build lint-src lint-tests

lint-src:
	# flake8 src
	$(docker-run) flake8 src

lint-tests:
	# flake8 tests
	$(docker-run) flake8 tests
