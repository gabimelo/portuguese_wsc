PROCESSED_DATA_DIR_NAME = 'data/processed'
INTERIM_DATA_DIR_NAME = 'data/interim'
RAW_DATA_DIR_NAME = 'data/raw'
MODEL_DIR = 'models'

WIKI_PT_XML_FILE_NAME = 'data/external/ptwiki-latest-pages-articles.xml.bz2'
WIKI_PT_TXT_FILE_BASE_NAME = 'wiki_pt'
WIKI_PT_TXT_DIR_NAME = 'data/interim/wiki_pt_splits'

MISSING_TRANSLATION_INDEXES = [60, 61, 62, 63, 72, 73, 86, 87]
WINOGRAD_PT_HTML_SCHEMAS_FILE = f'{RAW_DATA_DIR_NAME}/portuguese_wsc.html'
WINOGRAD_ASSOCIATIVE_LABEL_FILE = f'{RAW_DATA_DIR_NAME}/WSC_associative_label.json'
WINOGRAD_SWITCHED_LABEL_FILE = f'{RAW_DATA_DIR_NAME}/WSC_switched_label.json'
MANUAL_PT_FIXES_FILE = f'{RAW_DATA_DIR_NAME}/manual_fixes_portuguese.json'
CAPITALIZED_WORD_LIST_FILE = f'{RAW_DATA_DIR_NAME}/capitalized_words.txt'

PORTUGUESE = True

if PORTUGUESE:
    WINOGRAD_SCHEMAS_ORIGINAL_FILE = ''
    WINOGRAD_SCHEMAS_FILE = f'{PROCESSED_DATA_DIR_NAME}/portuguese_wsc.json'
    FILTER_WORDS = 5
else:
    WINOGRAD_SCHEMAS_ORIGINAL_FILE = f'{RAW_DATA_DIR_NAME}/english_wsc.json'
    WINOGRAD_SCHEMAS_FILE = f'{PROCESSED_DATA_DIR_NAME}/english_wsc.json'
    PROCESSED_DATA_DIR_NAME = f'{PROCESSED_DATA_DIR_NAME}/english-wikitext-2'
    MODEL_DIR = f'{MODEL_DIR}/english-wikitext-2'
    FILTER_WORDS = 0

TRAINED_MODEL_DIR = f'{MODEL_DIR}/trained_models'
TEST_SET_FILE_NAME = f'{PROCESSED_DATA_DIR_NAME}/test.txt'
TRAIN_SET_FILE_NAME = f'{PROCESSED_DATA_DIR_NAME}/train.txt'
VAL_SET_FILE_NAME = f'{PROCESSED_DATA_DIR_NAME}/valid.txt'
MODEL_FILE_NAME = f'{TRAINED_MODEL_DIR}/model-{{}}.pt'
MODEL_RESULTS_FILE_NAME = f'{TRAINED_MODEL_DIR}/model-results-{{}}.txt'
CORPUS_DICTIONARY_FILE_NAME = f'{MODEL_DIR}/corpus_dictionary.pkl'
CORPUS_FILE_NAME = f'{MODEL_DIR}/corpus.pkl'
FILE_TOKEN_COUNT_DICT_FILE_NAME = f'{MODEL_DIR}/file_token_count_dict.json'

RANDOM_SEED = 1111
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

USE_DATA_PARALLELIZATION = False
MAIN_GPU_INDEX = 0

WORDS_TO_GENERATE = 1000
TEMPERATURE = 1.0  # higher will increase diversity. Has to be greater or equal 1e-3
