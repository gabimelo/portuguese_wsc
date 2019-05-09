# -*- coding: utf-8 -*-
import shutil

from src.logger import Logger
from src.consts import (
    WIKI_PT_TXT_FILE_NAME, WIKI_PT_TXT_DIR_NAME, TEST_SET_FILE_NAME, TRAIN_SET_FILE_NAME, VAL_SET_FILE_NAME
)

logger = Logger()


def main():
    shutil.copy(WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_NAME + '00.txt', TRAIN_SET_FILE_NAME)
    shutil.copy(WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_NAME + '01.txt', VAL_SET_FILE_NAME)
    shutil.copy(WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_NAME + '02.txt', TEST_SET_FILE_NAME)

    logger.info('Finished generating processed dataset')


if __name__ == '__main__':
    main()
