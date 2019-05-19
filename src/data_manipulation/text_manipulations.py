import os
import json
import pickle
from io import open

import torch

from src.consts import WIKI_PT_TXT_DIR_NAME, FILE_TOKEN_COUNT_DICT_FILE_NAME, CORPUS_DICTIONARY_FILE_NAME


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        self.train = None
        self.valid = None
        self.test = None

        if os.path.exists(CORPUS_DICTIONARY_FILE_NAME):
            self.dictionary = pickle.load(open(CORPUS_DICTIONARY_FILE_NAME, "rb"))
        else:
            self.dictionary = Dictionary()
            file_token_count_dict = {}
            # TODO figure out if should be WIKI_PT_TXT_DIR_NAME or PROCESSED_DATA_DIR_NAME
            for file_name in os.listdir(WIKI_PT_TXT_DIR_NAME):
                file_token_count = self.generate_corpus_dictionary(WIKI_PT_TXT_DIR_NAME + '/' + file_name)
                file_token_count_dict[file_name] = file_token_count

            with open(FILE_TOKEN_COUNT_DICT_FILE_NAME, 'w') as outfile:
                json.dump(file_token_count_dict, outfile)

            pickle.dump(self.dictionary, open(CORPUS_DICTIONARY_FILE_NAME, "wb"))

    def add_corpus_data(self, path):
        # TODO hard coded for now, must fix this
        self.train = self.tokenize(path + '/train.txt', 'wiki_pt00.txt')
        self.valid = self.tokenize(path + '/val.txt', 'wiki_pt01.txt')
        self.test = self.tokenize(path + '/test.txt', 'wiki_pt02.txt')

    def generate_corpus_dictionary(self, file_name):
        print(file_name)
        assert os.path.exists(file_name)
        # Add words to the dictionary
        with open(file_name, 'r', encoding="utf8") as f:
            file_token_count = 0
            for line in f:
                words = line.split() + ['<eos>']
                file_token_count += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        return file_token_count

    def tokenize(self, file_path, file_name):
        """Tokenizes a text file."""
        with open(FILE_TOKEN_COUNT_DICT_FILE_NAME, 'r', encoding='utf-8') as fp:
            file_token_count_dict = json.load(fp)

        with open(file_path, 'r', encoding="utf8") as f:
            tokens = torch.LongTensor(file_token_count_dict[file_name])
            file_token_count = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    tokens[file_token_count] = self.dictionary.word2idx[word]
                    file_token_count += 1

        return tokens
