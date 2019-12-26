import os
import json
import pickle
from functools import partial

import torch

from src.helpers.consts import (
    TRAIN_SET_FILE_NAME, TEST_SET_FILE_NAME, VAL_SET_FILE_NAME, FILE_TOKEN_COUNT_DICT_FILE_NAME,
    CORPUS_DICTIONARY_FILE_NAME, CORPUS_FILE_NAME
)
from src.datasets_manipulation.dictionary import Dictionary


class Corpus(object):
    def __init__(self):
        self.train = self.valid = self.test = None
        if os.path.exists(CORPUS_DICTIONARY_FILE_NAME):
            self.dictionary = pickle.load(open(CORPUS_DICTIONARY_FILE_NAME, "rb"))
        else:
            self.dictionary = Dictionary()
            self.dictionary.generate_full_dir_dictionary()

    def add_corpus_data(self):
        with open(FILE_TOKEN_COUNT_DICT_FILE_NAME, 'r', encoding='utf-8') as fp:
            file_token_count_dict = json.load(fp)

        tokenize = partial(self.tokenize, file_token_count_dict)

        self.train = tokenize(TRAIN_SET_FILE_NAME)
        self.test = tokenize(TEST_SET_FILE_NAME)
        self.valid = tokenize(VAL_SET_FILE_NAME)
        self.save_corpus()

    def save_corpus(self):
        pickle.dump(self, open(CORPUS_FILE_NAME, "wb"))

    def tokenize(self, file_token_count_dict, file_path):
        '''
            Tokenizes a text file, that is: returns a LongTensor where each element represents the
            token index for the token that was present at that text sequence.
            The manner of reading from the file to generate word sequences must be the same that is
            utilized in the dictionary generating code.
        '''
        # TODO check if this really needs to be a torch tensor
        tokens = torch.LongTensor(file_token_count_dict[file_path])

        with open(file_path, 'r', encoding="utf8") as f:
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
