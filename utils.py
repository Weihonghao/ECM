import logging
import numpy as np
from os.path import join as pjoin
from tensorflow.python.platform import gfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def load_glove_embeddings(embed_path):
    logger.info("Loading glove embedding...")
    glove = np.load(embed_path)['glove']
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove

def add_paddings(sentence, max_length, n_features=1):
    mask = [True] * len(sentence)
    pad_len = max_length - len(sentence)
    if pad_len > 0:
        padded_sentence = sentence + [0] * pad_len
        mask += [False] * pad_len
    else:
        padded_sentence = sentence[:max_length]
        mask = mask[:max_length]
    return padded_sentence, mask

def preprocess_dataset(dataset, question_maxlen, context_maxlen):
    processed = []
    for q, q_len, c, c_len, ans in dataset:
        # add padding:
        q_padded, q_mask = add_paddings(q, question_maxlen)
        c_padded, c_mask = add_paddings(c, context_maxlen)
        processed.append([q_padded, q_mask, c_padded, c_mask, ans])
    return processed

def strip(x):
    return map(int, x.strip().split(" "))