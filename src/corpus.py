import os
import json
import pickle
from io import open

import torch

from src.consts import (
    PROCESSED_DATA_DIR_NAME, FILE_TOKEN_COUNT_DICT_FILE_NAME, CORPUS_DICTIONARY_FILE_NAME, CORPUS_FILE_NAME
)
from src.dictionary import Dictionary


class Corpus(object):
    def __init__(self):
        self.train = self.valid = self.test = None
        if os.path.exists(CORPUS_DICTIONARY_FILE_NAME):
            self.dictionary = pickle.load(open(CORPUS_DICTIONARY_FILE_NAME, "rb"))
        else:
            self.dictionary = Dictionary()
            file_token_count_dict = self.dictionary.generate_full_dir_dictionary()
            self.dictionary.save_dictionary(file_token_count_dict)

    def add_corpus_data(self, path=PROCESSED_DATA_DIR_NAME):
        if 'english-wikitext-2' in CORPUS_FILE_NAME:
            self.train = self.tokenize(path + '/train.txt', 'train.txt')
            self.valid = self.tokenize(path + '/valid.txt', 'valid.txt')
            self.test = self.tokenize(path + '/test.txt', 'test.txt')
        else:
            self.train = self.tokenize(path + '/train.txt', 'wiki_pt00.txt')
            self.valid = self.tokenize(path + '/val.txt', 'wiki_pt01.txt')
            self.test = self.tokenize(path + '/test.txt', 'wiki_pt02.txt')
        self.save_corpus()

    def save_corpus(self):
        pickle.dump(self, open(CORPUS_FILE_NAME, "wb"))

    def tokenize(self, file_path, file_name):
        """
        Tokenizes a text file.
        file_path is actual file to be tokenized
        file_name is used only to grab token count from file_token_count_dict
        """
        with open(FILE_TOKEN_COUNT_DICT_FILE_NAME, 'r', encoding='utf-8') as fp:
            file_token_count_dict = json.load(fp)

        with open(file_path, 'r', encoding="utf8") as f:
            tokens = torch.LongTensor(file_token_count_dict[file_name])
            file_token_count = 0
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = line.strip().split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        tokens[file_token_count] = self.dictionary.word2idx[word]
                    else:
                        try:
                            tokens[file_token_count] = self.dictionary.word2idx['<unk>']
                        except KeyError:
                            tokens[file_token_count] = self.dictionary.word2idx['<UNK>']
                    file_token_count += 1

        return tokens
