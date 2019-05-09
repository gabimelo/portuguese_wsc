# -*- coding: utf-8 -*-
from src.data.text_manipulations import Corpus
from src.consts import (
    CORPUS_DICTIONARY_FILE_NAME
)
from src.logger import Logger

logger = Logger()


def main():
    Corpus()
    logger.info('Finished generating Corpus Dictionary pickle')


if __name__ == '__main__':
    main()
