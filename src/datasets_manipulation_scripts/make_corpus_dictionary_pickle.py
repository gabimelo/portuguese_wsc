# -*- coding: utf-8 -*-
from src.data.text_manipulations import Corpus
from src.helpers.logger import Logger

logger = Logger()


def main():
    Corpus()
    logger.info('Finished generating Corpus Dictionary pickle')


if __name__ == '__main__':
    main()
