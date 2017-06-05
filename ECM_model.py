from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import inspect
import numpy as np
import tensorflow as tf
import copy
import logging
import preprocess_data
import tensorflow as tf
import numpy as np


class ECMModel(object):
    def __init__(self, embeddings, id2word, config):
        magic_number = 256
        assert  (magic_number%2 == 0)
        self.embeddings = tf.cast(embeddings, dtype=tf.float32)
        # self.vocab_label = vocab_label  # label for vocab
        # self.emotion_label = emotion_label  # label for emotion
        self.config = config
        self.batch_size = config.batch_size
        #print("batch size", self.batch_size)
        self.vocab_size = config.vocab_size
        self.non_emotion_size = config.non_emotion_size
        self.emotion_size = self.vocab_size - self.non_emotion_size
        self.id2word = id2word

        '''if (self.config.vocab_size % 2 == 1):
            self.decoder_state_size = config.vocab_size + 1
            print (len(self.id2word))
            id2word.append('NULL')
        else:
            self.decoder_state_size = config.vocab_size'''

        self.decoder_state_size = magic_number
        self.encoder_state_size = int(self.decoder_state_size / 2)
        self.emotion_size = 6
        self.GO_id = 1
        self.pad_id = 0
        self.IM_size = 256
        self.internalMemory = tf.get_variable("IM", shape=[self.emotion_size, self.IM_size],
                                              initializer=tf.contrib.layers.xavier_initializer())

        self.vu = tf.get_variable("vu", shape=[self.decoder_state_size, 1], initializer=tf.contrib.layers.xavier_initializer())


        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

        self.question = tf.placeholder(tf.int32, shape=[None, None], name='question')
        self.question_len = tf.placeholder(tf.int32, shape=[None], name='question_len')
        self.answer = tf.placeholder(tf.int32, shape=[None, None], name='answer')
        self.answer_len = tf.placeholder(tf.int32, shape=[None], name='answer_len')
        self.emotion_tag = tf.placeholder(tf.int32, shape=[None], name='emotion_tag')
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout", shape=())
        self.LQ = tf.placeholder(dtype=tf.int32, name='LQ', shape=())  # batch
        self.LA = tf.placeholder(dtype=tf.int32, name='LA', shape=())  # batch

        with tf.variable_scope("embeddings"):
            if self.config.retrain_embeddings:  # whether to cotrain word embedding
                embeddings = tf.Variable(self.embeddings, name="Emb", dtype=tf.float32)
            else:
                embeddings = tf.cast(self.embeddings, tf.float32)

            question_embeddings = tf.nn.embedding_lookup(embeddings, self.question)
            self.q = tf.reshape(question_embeddings, shape=[-1, self.LQ, self.config.embedding_size])
            answer_embeddings = tf.nn.embedding_lookup(embeddings, self.answer)
            self.a = tf.reshape(answer_embeddings, shape=[-1, self.LA, self.config.embedding_size])

    def encode(self, inputs, sequence_length, encoder_state_input, dropout=1.0):
        """
        In a generalized encode function, you pass in your inputs,
        sequence_length, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input (padded all to the same length)
        :param mask: mask of the sequence
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        logging.debug('-' * 5 + 'encode' + '-' * 5)
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.encoder_state_size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.encoder_state_size, state_is_tuple=True)

        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=dropout)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=dropout)

        initial_state_fw = None
        initial_state_bw = None
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input
        logging.debug('sequence_length: %s' % str(sequence_length))
        logging.debug('Inputs: %s' % str(inputs))
        # Get lstm cell output
        print(inputs.get_shape())
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fw_cell,
            cell_bw=lstm_bw_cell,
            inputs=inputs,
            sequence_length=sequence_length,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32)

        # Concatinate forward and backword hidden output vectors.
        # each vector is of size [batch_size, sequence_length, encoder_state_size]

        logging.debug('fw hidden state: %s' % str(outputs_fw))
        hidden_state = tf.concat([outputs_fw, outputs_bw], 2)
        logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        print("encode output ", final_state_fw[1].get_shape())
        concat_final_state = tf.concat([final_state_fw[1], final_state_bw[1]], 1)
        logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
        return hidden_state, concat_final_state

    def decode(self, encoder_outputs, encoder_final_state, decoder_length):
        print('decode start')

        # initialize first decode state
        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_length)  # all False at the initial step
            GO_emb = tf.ones([self.batch_size], dtype=tf.int32, name='GO')
            initial_input = tf.nn.embedding_lookup(self.embeddings, GO_emb)
            initial_cell_state = encoder_final_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            print('before return initial')
            logging.debug('initial_elements_finished: %s' % str(initial_elements_finished))
            logging.debug('initial_input: %s' % str(initial_input))
            logging.debug('initial_cell_state: %s' % str(initial_cell_state))
            logging.debug('initial_cell_output: %s' % str(initial_cell_output))
            logging.debug('initial_loop_state: %s' % str(initial_loop_state))

            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            # get next state
            print('in trans')

            def get_next_input():
                print('in get next input')
                previous_output_id = self.external_memory_function(previous_output)
                previous_output_vector = tf.nn.embedding_lookup(self.embeddings, previous_output_id)
                score = attention_mechanism(previous_state)
                weights = tf.nn.softmax(score)
                print("here")
                weights = tf.reshape(weights, [tf.shape(weights)[0], 1, tf.shape(weights)[1]])
                logging.debug('weights: %s' % str(weights))
                logging.debug('attention_mechanism.values: %s' % str(attention_mechanism.values))
                context = tf.matmul(weights, attention_mechanism.values)
                logging.debug('context: %s' % str(context))
                context = tf.reshape(context, [-1, context.get_shape().as_list()[2]])
                print("here1")
                logging.debug('previous_output_vector: %s' % str(previous_output_vector))
                logging.debug('context: %s' % str(context))
                attention = tf.layers.dense(inputs=tf.concat([previous_output_vector, context], 1), units=self.IM_size)
                read_gate = tf.sigmoid(attention, name="read_gate")
                logging.debug('read_gate: %s' % str(read_gate))
                read_gate_output = tf.nn.embedding_lookup(self.internalMemory,self.emotion_tag)
                logging.debug('gate output: %s' % str(read_gate_output))
                next_input = tf.concat(
                    [context, previous_output_vector, read_gate_output], 1)
                logging.debug('next_input: %s' % str(next_input))
                return next_input

            elements_finished = (time >= decoder_length)  # this operation produces boolean tensor of [batch_size]
            # defining if corresponding sequence has ended
            finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.pad_id)  ## undefined
            logging.debug('finished: %s' % str(finished))
            logging.debug('pad_step_embedded: %s' % str(pad_step_embedded))
            input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
            output, state = decode_cell(input, previous_state)
            loop_state = None

            write_gate = tf.sigmoid(tf.layers.dense(state, self.IM_size, name="write_gate"))
            self.internalMemory[self.emotion_tag] = self.internalMemory[self.emotion_tag] * write_gate

            return (elements_finished,
                    input,
                    state,
                    output,
                    loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:  # time == 0
                assert previous_output is None and previous_state is None
                print("initialii******")
                return loop_fn_initial()
            else:
                print("trainsition******")
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        decode_cell = tf.contrib.rnn.GRUCell(self.decoder_state_size)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.decoder_state_size, encoder_outputs)
        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decode_cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()
        return decoder_outputs

    def external_memory_function(self, decode_output):  # decode_output, shape[batch_size,vocab_size]


        print('flag1')
        gto = tf.sigmoid(tf.reduce_sum(tf.matmul(decode_output, self.vu)))
        print('flag2')
        emotion_num = self.emotion_size
        print('flag3')
        return tf.argmax(tf.concat([gto * decode_output[:, :emotion_num], (1 - gto) * decode_output[:, emotion_num:]],
                                   1), axis=1)  # [batch_size,1]

    def create_feed_dict(self, question_batch, question_len_batch, emotion_tag_batch, answer_batch=None,
                         answer_len_batch=None, is_train=True):
        feed_dict = {}
        LQ = np.max(question_len_batch)

        def add_paddings(sentence, max_length):
            pad_len = max_length - len(sentence)
            if pad_len > 0:
                padded_sentence = sentence + [0] * pad_len
            else:
                padded_sentence = sentence[:max_length]
            return padded_sentence

        def padding_batch(data, max_len):
            padded_data = []
            for sentence in data:
                d = add_paddings(sentence, max_len)
                padded_data.append(d)
            return padded_data

        feed_dict[self.question_len] = question_len_batch
        feed_dict[self.LQ] = LQ
        feed_dict[self.emotion_tag] = emotion_tag_batch
        padded_question = padding_batch(question_batch, LQ)
        print("padding question size ", np.array(padded_question).shape)
        feed_dict[self.question] = padded_question

        if is_train:
            assert answer_batch is not None
            assert answer_len_batch is not None
            LA = np.max(answer_len_batch)
            padded_answer = padding_batch(answer_batch, LA)
            feed_dict[self.answer] = padded_answer
            feed_dict[self.answer_len] = answer_len_batch
            feed_dict[self.LA] = LA
            feed_dict[self.dropout_placeholder] = 0.8
        else:
            feed_dict[self.dropout_placeholder] = 1.0

        return feed_dict

    def train(self, sess, training_set):
        question_batch, question_len_batch, answer_batch, answer_len_batch, tag_batch = training_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch, tag_batch, answer_batch,
                                           answer_len_batch, is_train=True)

        def emotion_distribution(decode_outputs):
            ids = self.external_memory_function(decode_outputs)
            return tf.cast((ids < (self.emotion_size)), dtype=tf.int64)

        def loss(results):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=self.answer)  # self.vocab_label)
            emotion_label = tf.cast((self.answer < (self.emotion_size)), dtype=tf.int64)
            loss += tf.nn.softmax_cross_entropy_with_logits(logits=emotion_distribution(results),
                                                            labels=emotion_label)
            loss += 2 * tf.nn.l2_loss(self.internalMemory)
            return loss

        encoder_outputs, encoder_final_state = self.encode(self.q, self.question_len, None, self.dropout_placeholder)
        results = self.decode(encoder_outputs, encoder_final_state, self.answer_len)
        tfloss = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss(results))
        return sess.run(tfloss, feed_dict=input_feed)

    def test(self, sess, test_set):
        question_batch, question_len_batch, tag_batch = test_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch, tag_batch, answer_batch=None,
                                           answer_len_batch=None, is_train=False)

        encoder_outputs, encoder_final_state = self.encode(self.q, self.question_len, None, self.dropout_placeholder)
        results = self.decode(encoder_outputs, encoder_final_state, self.answer_len)
        tfids = tf.argmax(results, axis=1)
        ids = sess.run(tfids, feed_dict=input_feed)
        return [self.id2word[id] for each in ids]
