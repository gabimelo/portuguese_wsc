import os

from src.datasets_manipulation.dictionary import Dictionary
from src.helpers.logger import Logger
from src.helpers.consts import CORPUS_DICTIONARY_FILE_NAME

logger = Logger()


def main():
    if not os.path.exists(CORPUS_DICTIONARY_FILE_NAME):
        dictionary = Dictionary()
        dictionary.generate_full_dir_dictionary()
    logger.info('Finished generating Corpus Dictionary pickle')


if __name__ == '__main__':
    main()
