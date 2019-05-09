# -*- coding: utf-8 -*-
import os
import click

from gensim.corpora import WikiCorpus

from src.logger import Logger
from src.consts import WIKI_PT_TXT_FILE_NAME, WIKI_PT_TXT_DIR_NAME, WIKI_PT_XML_FILE_NAME

logger = Logger()


def next_output_file_name(output_dir, num):
    """Get the next filename to use for writing new articles."""
    output_file_name = output_dir + '/' + WIKI_PT_TXT_FILE_NAME + '{:>02d}'.format(num) + '.txt'
    return output_file_name


def make_corpus(input_file, output_dir, split=True, size=10000):
    """Convert Wikipedia xml dump file to text corpus"""
    wiki = WikiCorpus(input_file)

    count = num = 0
    output_file_name = next_output_file_name(output_dir, num)
    output_file = open(output_file_name, 'w')

    for text in wiki.get_texts():
        output_file.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        count += 1
        if count == size:
            num += 1
            if split:
                logger.info('%s Done.' % output_file_name)
                output_file.close()
                output_file_name = next_output_file_name(output_dir, num)
                output_file = open(output_file_name, 'w')
            else:
                logger.info('Processed ' + str(count*num) + ' articles')
            count = 0

    output_file.close()

    logger.info('Completed.')


@click.command()
@click.option('--split', default=True)
def main(split):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if not os.path.exists(WIKI_PT_TXT_DIR_NAME):
        os.makedirs(WIKI_PT_TXT_DIR_NAME)

    make_corpus(WIKI_PT_XML_FILE_NAME, WIKI_PT_TXT_DIR_NAME, split=split, size=10000)


if __name__ == '__main__':
    main()
