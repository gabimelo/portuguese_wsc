# -*- coding: utf-8 -*-
import os
import click

from src.logger import Logger
from src.consts import WIKI_PT_TXT_DIR_NAME, WIKI_PT_XML_FILE_NAME
from src.wikidump import make_corpus_files

logger = Logger()


@click.command()
@click.option('--split', default=True)
def main(split):
    if not os.path.exists(WIKI_PT_TXT_DIR_NAME):
        os.makedirs(WIKI_PT_TXT_DIR_NAME)

    make_corpus_files(WIKI_PT_XML_FILE_NAME, WIKI_PT_TXT_DIR_NAME, split=split, size=10000)

    logger.info('Finished generating interim dataset')


if __name__ == '__main__':
    main()
