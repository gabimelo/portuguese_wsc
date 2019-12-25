# -*- coding: utf-8 -*-
import os
import click

from src.helpers.logger import Logger
from src.helpers.consts import WIKI_PT_TXT_DIR_NAME, WIKI_PT_XML_FILE_NAME, INTERIM_DATA_DIR_NAME
from src.datasets_manipulation.wikidump import make_corpus_files

logger = Logger()


@click.command()
@click.option('--split', default=True)
def main(split):
    if len(os.listdir(WIKI_PT_TXT_DIR_NAME)) != 0:
        logger.info("Directory for interim data not empty, skipping this process.")
    else:
        if not os.path.exists(INTERIM_DATA_DIR_NAME):
            os.makedirs(INTERIM_DATA_DIR_NAME)
        if not os.path.exists(WIKI_PT_TXT_DIR_NAME):
            os.makedirs(WIKI_PT_TXT_DIR_NAME)

        make_corpus_files(WIKI_PT_XML_FILE_NAME, WIKI_PT_TXT_DIR_NAME, split=split, size=10000)

        logger.info('Finished generating interim dataset')


if __name__ == '__main__':
    main()
