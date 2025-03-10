import json
import pickle

from src.helpers import consts


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.word_count = {'<unk>': 0}
        self.idx2word = ['<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def word2idx_from_idx2word(self):
        self.word2idx = dict(zip(self.idx2word, list(range(len(self.idx2word)))))

    def filter_words(self, min_appearances_for_word_in_vocab):
        del self.word2idx

        # we don't want to remove word at index 0, which is <UNK>, and for now has not been present in data
        for index in range(len(self.idx2word) - 1, 0, -1):
            if self.word_count[self.idx2word[index]] < min_appearances_for_word_in_vocab:
                self.word_count['<unk>'] += self.word_count[self.idx2word[index]]
                del self.word_count[self.idx2word[index]]
                del self.idx2word[index]

        self.word2idx_from_idx2word()

    def generate_full_dir_dictionary(self):
        file_token_count_dict = {}
        for file_name in [consts.TEST_SET_FILE_NAME, consts.TRAIN_SET_FILE_NAME, consts.VAL_SET_FILE_NAME]:
            file_token_count = self.generate_corpus_dictionary(file_name)
            file_token_count_dict[file_name] = file_token_count

        if consts.FILTER_WORDS > 0:
            self.filter_words(consts.FILTER_WORDS)

        self.save_dictionary(file_token_count_dict)

    def save_dictionary(self, file_token_count_dict):
        with open(consts.FILE_TOKEN_COUNT_DICT_FILE_NAME, 'w') as outfile:
            json.dump(file_token_count_dict, outfile)

        pickle.dump(self, open(consts.CORPUS_DICTIONARY_FILE_NAME, "wb"))

    def generate_corpus_dictionary(self, file_name):
        with open(file_name, 'r', encoding="utf8") as f:
            file_token_count = 0
            for line in f:
                if len(line.strip()) == 0:
                    continue
                words = line.strip().split() + ['<eos>']
                file_token_count += len(words)
                for word in words:
                    self.add_word(word)

        return file_token_count
