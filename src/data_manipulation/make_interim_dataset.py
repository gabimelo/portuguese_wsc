# -*- coding: utf-8 -*-
import os
import click
import sys

from gensim.corpora import WikiCorpus

from src.logger import Logger
from src.consts import WIKI_PT_TXT_FILE_NAME, WIKI_PT_TXT_DIR_NAME, WIKI_PT_XML_FILE_NAME

logger = Logger()


def next_output_file_name(output_dir, num):
    """Get the next filename to use for writing new articles."""
    output_file_name = output_dir + '/' + WIKI_PT_TXT_FILE_NAME + '{:>02d}'.format(num) + '.txt'
    return output_file_name


if sys.version_info[0] >= 3:
    unicode = str


def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.

    Returns
    -------
    str
        Unicode version of `text`.

    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15


def tokenize(content, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
    # override original method in wikicorpus.py
    return [to_unicode(token) for token in content.split()
            if token_min_len <= len(token) <= token_max_len and not token.startswith('_')]


def make_corpus(input_file, output_dir, split=True, size=10000):
    """Convert Wikipedia xml dump file to text corpus"""
    wiki = WikiCorpus(input_file, tokenizer_func=tokenize)

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
                logger.info('Processed ' + str(count * num) + ' articles')
            count = 0

    output_file.close()

    logger.info('Completed.')


@click.command()
@click.option('--split', default=True)
def main(split):
    if not os.path.exists(WIKI_PT_TXT_DIR_NAME):
        os.makedirs(WIKI_PT_TXT_DIR_NAME)

    make_corpus(WIKI_PT_XML_FILE_NAME, WIKI_PT_TXT_DIR_NAME, split=split, size=10000)

    logger.info('Finished generating interim dataset')


if __name__ == '__main__':
    main()
