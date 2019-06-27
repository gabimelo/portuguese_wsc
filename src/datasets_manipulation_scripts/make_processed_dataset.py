# -*- coding: utf-8 -*-
import shutil
import os

from src.helpers.logger import Logger
from src.consts import (
    WIKI_PT_TXT_FILE_BASE_NAME, WIKI_PT_TXT_DIR_NAME, TEST_SET_FILE_NAME, TRAIN_SET_FILE_NAME, VAL_SET_FILE_NAME,
    PROCESSED_DATA_DIR_NAME
)

logger = Logger()


def main():
    if TRAIN_SET_FILE_NAME.split('/')[-1] in os.listdir(PROCESSED_DATA_DIR_NAME):
        logger.info("Training dataset file already in processed data directory, skipping this process.")

    else:
        shutil.copy(WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '00.txt', TRAIN_SET_FILE_NAME)
        shutil.copy(WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '01.txt', VAL_SET_FILE_NAME)
        shutil.copy(WIKI_PT_TXT_DIR_NAME + '/' + WIKI_PT_TXT_FILE_BASE_NAME + '02.txt', TEST_SET_FILE_NAME)

        logger.info('Finished generating processed dataset')


if __name__ == '__main__':
    main()
