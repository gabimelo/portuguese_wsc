# -*- coding: utf-8 -*-
from src.datasets_manipulation.corpus import Corpus
from src.helpers.logger import Logger

logger = Logger()


def main():
    corpus = Corpus()
    corpus.add_corpus_data()
    logger.info('Finished generating Corpus pickle')


if __name__ == '__main__':
    main()
