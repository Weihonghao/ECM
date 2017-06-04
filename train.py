from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from os.path import join as pjoin

import preprocess_data
import utils
import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_float("keep_prob", 0.95, "Keep prob of output.")
tf.app.flags.DEFINE_integer("state_size", 256, "Size of encoder and decoder hidden layer.")
tf.app.flags.DEFINE_string("data_dir", "data/", "Data directory")
tf.app.flags.DEFINE_string("vocab_path", "data/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("embed_path", "",
                           "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("train_dir", "train/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 100,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_print", 1,
                            "How many training steps to print info.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("window_batch", 3, "window size / batch size")

tf.app.flags.DEFINE_string("retrain_embeddings", False, "Whether to retrain word embeddings")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

class DataConfig(object):
    """docstring for DataDir"""

    def __init__(self, data_dir):
        self.train_from = pjoin(data_dir, 'train.ids.from')
        self.train_to = pjoin(data_dir, 'train.ids.to')
        self.val_from = self.train_from
        self.val_to = self.train_to

        self.val_from = pjoin(data_dir, 'val.ids.from')
        self.val_to = pjoin(data_dir, 'val.ids.to')
        self.test_from = pjoin(data_dir, 'test.ids.from')
        self.test_to = pjoin(data_dir, 'test.ids.to')


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.
    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(preprocess_data.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def initialize_model(session, model):
    """Create translation model and initialize or load parameters in session."""


    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    """Train a en->fr translation model using WMT data."""
    data_config = DataConfig(FLAGS.data_dir)
    logFile = open('data/log.txt', 'w')
    embed_path = FLAGS.embed_path or pjoin("data", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = utils.load_glove_embeddings(embed_path)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    non_emotion_size, vocab, rev_vocab = preprocess_data.initialize_vocabulary(vocab_path)
    FLAGS.vocab_size = len(vocab)
    print(embeddings.shape[0], len(vocab))
    assert embeddings.shape[0] == len(vocab)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        # Create model.
        with tf.device('/gpu:1'):
            print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = ECM_model.ECMModel(embeddings, vocab_label, emotion_label, rev_vocab, FLAGS)
            initialize_model(sess, model)
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
            toc = time.time()
            logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

            # Read data into buckets and compute their sizes.
            print("Reading development and training data (limit: %d)."
                  % FLAGS.max_train_data_size)
            dev_set = read_data(data_config.val_from, data_config.val_to)
            train_set = read_data(data_config.train_from, data_config.train_to, FLAGS.max_train_data_size)

            while True:
                model.train(sess, train_set)
                


def decode():
    embed_path = FLAGS.embed_path or pjoin("data", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = utils.load_glove_embeddings(embed_path)

    with tf.Session() as sess:
        # Create model and load parameters.
        # model = create_model(sess, embeddings, True)
        # model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
        non_emotion_size, en_vocab, rev_fr_vocab = preprocess_data.initialize_vocabulary(vocab_path)
        FLAGS.vocab_size = len(en_vocab)
        print("embeddings.shape[0]: " + str(embeddings.shape[0]))
        print("len(en_vocab):" + str(len(en_vocab)))
        assert embeddings.shape[0] == len(en_vocab)
        model = create_model(sess, embeddings, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = preprocess_data.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if preprocess_data.EOS_ID in outputs:
                outputs = outputs[:outputs.index(preprocess_data.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(_):
    elif FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
