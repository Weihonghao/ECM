from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import ECM_model

import numpy as np
import tensorflow as tf
from os.path import join as pjoin

import preprocess_data
import utils
import seq2seq_model
from tensorflow.python.platform import gfile

logging.basicConfig(level=logging.INFO)


tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 1, "Number of epochs to train.")
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

def strip(x):
    return map(int, x.strip().split(" "))



'''def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]'''

class DataConfig(object):
    """docstring for DataDir"""

    def __init__(self, data_dir):
        self.train_from = pjoin(data_dir, 'train.ids.from')
        self.train_to = pjoin(data_dir, 'train.ids.to')
        self.train_tag = pjoin(data_dir, 'train.tag')


        self.val_from = pjoin(data_dir, 'val.ids.from')
        self.val_to = pjoin(data_dir, 'val.ids.to')
        self.val_tag = pjoin(data_dir, 'val.tag')
        self.test_from = pjoin(data_dir, 'test.ids.from')
        self.test_to = pjoin(data_dir, 'test.ids.to')
        self.test_tag = pjoin(data_dir, 'test.tag')


def read_data(data_config):
    train = []
    max_q_len = 0
    max_a_len = 0
    print("Loading training data from %s ..." % data_config.train_from)
    with gfile.GFile(data_config.train_from, mode="rb") as q_file, \
         gfile.GFile(data_config.train_to, mode="rb") as a_file, \
         gfile.GFile(data_config.train_tag, mode="rb") as t_file:
            for (q, a, t) in zip(q_file, a_file, t_file):
                question = strip(q)
                answer = strip(a)
                tag = strip(t)
                
                sample = [question, len(question), answer, len(answer), tag]
                train.append(sample)
                max_q_len = max(max_q_len, len(question))
                max_a_len = max(max_a_len, len(answer))
                
    print("Finish loading %d train data." % len(train))
    val = []
    print("Loading training data from %s ..." %data_config.val_from)
    with gfile.GFile(data_config.val_from, mode="rb") as q_file, \
         gfile.GFile(data_config.val_to, mode="rb") as a_file, \
         gfile.GFile(data_config.val_tag, mode="rb") as t_file:
            for (q, a, t) in zip(q_file, a_file, t_file):
                question = strip(q)
                answer = strip(a)
                tag = strip(t)

                sample = [question, len(question), answer, len(answer), tag]
                val.append(sample)
                max_q_len = max(max_q_len, len(question))
                max_a_len = max(max_a_len, len(answer))

    print("Finish loading %d val data." % len(val))
    print("Max question length %d " % max_q_len)
    print("Max answer length %d " % max_a_len)

    return {"training": train, "validation": val, "question_maxlen": max_q_len, "answer_maxlen": max_a_len}


def initialize_model(session, model):
    """Create translation model and initialize or load parameters in session."""
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
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
    FLAGS.non_emotion_size = non_emotion_size
    FLAGS.encoder_state_size = 128
    FLAGS.decoder_state_size = 2 * FLAGS.encoder_state_size
    print(embeddings.shape[0], len(vocab))
    assert embeddings.shape[0] == len(vocab)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess: #log_device_placement=True
        # Create model.
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
        with tf.device('/gpu:1'):
            #print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = ECM_model.ECMModel(embeddings, rev_vocab, FLAGS)
            initialize_model(sess, model)

            '''tmpModel = initialize_model(sess, model)
            if tmpModel is not None:
                model = tmpModel'''
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
            toc = time.time()
            logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

            dataset = read_data(data_config)
            
            training_set = dataset['training'] # [question, len(question), answer, len(answer), tag]
            validation_set = dataset['validation']

            for epoch in range(FLAGS.epochs):
                logging.info("="* 10 + " Epoch %d out of %d " + "="* 10, epoch + 1, FLAGS.epochs)
                batch_num = len(training_set) / FLAGS.batch_size
                #prog = Progbar(target=batch_num)
                avg_loss = 0
                for i, batch in enumerate(utils.minibatches(training_set, FLAGS.batch_size, window_batch=FLAGS.window_batch)):
                    global_batch_num = batch_num * epoch + i
                    _, loss = model.train(sess, batch)
                    print('loss is: ', epoch,'  ',  i, '  ', loss)
                    avg_loss += loss
                avg_loss /= batch_num
                logging.info("Average training loss: {}".format(avg_loss))
                
                logging.info("-- validation --")
                batch_num = len(validation_set) / FLAGS.batch_size
                avg_loss = 0
                for i, batch in enumerate(utils.minibatches(validation_set, FLAGS.batch_size, window_batch=FLAGS.window_batch)):
                    global_batch_num = batch_num * epoch + i
                    loss = model.test(sess, batch)
                    print(loss)
                    #avg_loss += loss
                #avg_loss /= batch_num
                #logging.info("Average validation loss: {}".format(avg_loss))
                


def main(_):
    '''if FLAGS.decode:
        decode()
    else:'''
    train()


if __name__ == "__main__":
    tf.app.run()
