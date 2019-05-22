import datetime

WIKI_PT_XML_FILE_NAME = 'data/external/ptwiki-latest-pages-articles.xml.bz2'
WIKI_PT_TXT_FILE_NAME = 'wiki_pt'
WIKI_PT_TXT_DIR_NAME = 'data/interim/wiki_pt_splits'

PROCESSED_DATA_DIR_NAME = 'data/processed'
TEST_SET_FILE_NAME = 'data/processed/test.txt'
TRAIN_SET_FILE_NAME = 'data/processed/train.txt'
VAL_SET_FILE_NAME = 'data/processed/val.txt'

MODEL_FILE_NAME = 'models/trained_models/model-{}.pt'.format(datetime.datetime.now())
CORPUS_DICTIONARY_FILE_NAME = 'models/corpus_dictionary.pkl'
CORPUS_FILE_NAME = 'models/corpus.pkl'
FILE_TOKEN_COUNT_DICT_FILE_NAME = 'models/file_token_count_dict.json'

RANDOM_SEED = 42
USE_CUDA = True

BATCH_SIZE = 20
# TODO maybe should have TEST_BATCH_SIZE as well, and set it to 1
EVAL_BATCH_SIZE = 10
MODEL_TYPE = 'LSTM'  # other options: RNN_TANH, RNN_RELU, LSTM, GRU
EMBEDDINGS_SIZE = 200
HIDDEN_UNIT_COUNT = 200
LAYER_COUNT = 2
DROPOUT_PROB = 0.2
TIED = True  # tie the word embedding and softmax weights
SEQUENCE_LENGTH = 35
INITIAL_LEARNING_RATE = 20
EPOCHS = 40
GRADIENT_CLIPPING = 0.25
LOG_INTERVAL = 200
