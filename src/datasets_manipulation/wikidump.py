import re

from gensim.corpora import WikiCorpus
from nltk.tokenize import word_tokenize

from src.helpers.logger import Logger
from src.helpers.consts import WIKI_PT_TXT_FILE_BASE_NAME

logger = Logger()


def filter_good_and_featured_articles(elem, text, *args, **kwargs):
    regex_pt_good = re.compile(r'.*\{\{(artigo bom.*?)\}\}[\s]*', flags=re.DOTALL)
    regex_pt_featured = re.compile(r'.*\{\{(artigo destacado.*?)\}\}[\s]*', flags=re.DOTALL)

    if text is None:
        return False
    if regex_pt_good.match(text) or regex_pt_featured.match(text):
        return True
    else:
        return False


def get_output_file_name(output_dir, num):
    """Get the filename to use for writing new articles."""
    output_file_name = f'{output_dir}/{WIKI_PT_TXT_FILE_BASE_NAME}{num:>02d}.txt'
    return output_file_name


def tokenize(content, token_min_len=2, token_max_len=15, lower=True):
    # override original method in wikicorpus.py
    return word_tokenize(content, language='portuguese')


def make_corpus_files(input_file, output_dir, split=True, size=10000):
    """Convert Wikipedia xml dump file to text corpus"""
    wiki = WikiCorpus(
        input_file,
        tokenizer_func=tokenize,
        filter_articles=filter_good_and_featured_articles
    )

    count = num = 0
    output_file_name = get_output_file_name(output_dir, num)
    output_file = open(output_file_name, 'w')

    for text in wiki.get_texts():
        output_file.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        count += 1
        if count == size:
            num += 1
            if split:
                logger.info('%s Done.' % output_file_name)
                output_file.close()
                output_file_name = get_output_file_name(output_dir, num)
                output_file = open(output_file_name, 'w')
            else:
                logger.info('Processed ' + str(count * num) + ' articles')
            count = 0

    logger.info('Processed a total of' + str((size * num) + count) + ' articles')
    output_file.close()

    logger.info('Completed.')
