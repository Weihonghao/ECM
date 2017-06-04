from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import random
#import inspect
#import numpy as np
#import copy
import logging
#import preprocess_data
import tensorflow as tf

class Encoder(object):
    def __init__(self, vocab_size, decoder_lengths, embeddings, vocab_label, emotion_label, id2word, emotion_num, embed_size, dropout = 0):
        self.batch_size = 20
        self.emotion_size = 6
        self.vocab_size = vocab_size
        self.real_vocab_size = self.vocab_size + 4
        self.state_size = 128
        self.decoder_hidden_units = self.vocab_size
        self.eos_step_embedded = None
        self.decoder_lengths = decoder_lengths
        self.embeddings = embeddings
        self.attention_size = 256
        self.IM_size = 256
        self.internalMemory = tf.get_variable("IM", shape=[self.emotion_size, self.IM_size], initializer=tf.contrib.layers.xavier_initializer())
        #self.externalMemory = tf.get_variable("IM", shape=[1, self.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
        self.pad_id = self.vocab_size
        self.vocab_label = vocab_label
        self.emotion_label = emotion_label
        self.id2word = id2word
        self.emotion_num = emotion_num
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        self.embed_size = embed_size
        #self.dropout = dropout
        #logging.info("Dropout rate for encoder: {}".format(self.dropout))

        self.input = tf.placeholder(tf.float32, shape=[None, None, self.embed_size], name= 'input')
        self.output = tf.placeholder(tf.float32, shape=[None, None, self.embed_size], name= 'output')
        self.emotionTag = tf.placeholder(tf.float32, shape=[None, self.emotion_size], name= 'emotionTag')

    def encode(self, inputs, mask = None, encoder_state_input = None, dropout = 1.0):
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

        logging.debug('-'*5 + 'encode' + '-'*5)
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.state_size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.state_size, state_is_tuple=True)


        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout)

        initial_state_fw = None
        initial_state_bw = None
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input

        logging.debug('Inputs: %s' % str(inputs))
        sequence_length = None
        if mask is not None:
            sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
            sequence_length = tf.reshape(sequence_length, [-1,])
        # Get lstm cell output
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
                                                      cell_bw=lstm_bw_cell,\
                                                      inputs=inputs,\
                                                      sequence_length=sequence_length,
                                                      initial_state_fw=initial_state_fw,\
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float32)

        # Concatinate forward and backword hidden output vectors.
        # each vector is of size [batch_size, sequence_length, cell_state_size]

        logging.debug('fw hidden state: %s' % str(outputs_fw))
        hidden_state = tf.concat([outputs_fw, outputs_bw],2)
        logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        print("the shape of last final state", final_state_bw.get_shape())
        concat_final_state = tf.concat([final_state_fw, final_state_bw], 1)
        logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
        return hidden_state, concat_final_state, (final_state_fw, final_state_bw)





    def decode(self, encoder_outputs):



        #initialize first decode state
        def loop_fn_initial(encoder_final_state):
            initial_elements_finished = (0 >= self.decoder_lengths)  # all False at the initial step
            initial_input = self.eos_step_embedded
            initial_cell_state = encoder_final_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)




        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            # get next state
            def get_next_input():

                previous_output_vector = tf.nn.embedding_lookup(self.embeddings, previous_output)
                score = attention_mechanism(previous_state)
                weights = tf.nn.softmax(score)
                context = tf.matmul(weights, attention_mechanism.values)
                attention = tf.layers.dense(self.attention_size)(tf.concat([previous_output_vector, context], 1))
                read_gate = tf.sigmoid(tf.layers.dense(attention, self.IM_size, name="read_gate"))
                next_input = tf.concat([context, previous_output_vector, read_gate * self.internalMemory],1)

                '''output_logits = tf.add(tf.matmul(previous_output, W), b)
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(embeddings, prediction)'''
                return next_input

            elements_finished = (time >= self.decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                          # defining if corresponding sequence has ended

            finished = tf.reduce_all(elements_finished) # -> boolean scalar
            pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.pad_id)## undefined
            input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
            output, state = decode_cell(input, previous_state)
            loop_state = None

            write_gate = tf.sigmoid(tf.layers.dense(state, self.IM_size, name="write_gate"))
            self.internalMemory = self.internalMemory * write_gate


            output = self.External_Memory_function(output)

            return (elements_finished,
                    input,
                    state,
                    output,
                    loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:    # time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        decode_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_units)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.decoder_hidden_units, encoder_outputs)
        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decode_cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()
        return decoder_outputs

    def External_Memory_function(self, decode_output):
        #alpha = tf.sigmoid(tf.dot(self.externalMemory, decode_output))
        #weo = tf.get_variable("weo", shape=[1, self.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
        #wgo = tf.get_variable("wgo", shape=[1, self.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
        vu = tf.get_variable("vu", shape=[1, self.vocab_size], initializer=tf.contrib.layers.xavier_initializer())
        gto = tf.sigmoid(tf.tensordot(vu*decode_output))
        #emotion_vocab =  alpha * tf.softmax(weo * decode_output)
        #non_emotion_vocab = (1-alpha) * tf.softmax(wgo * decode_output)
        return tf.arg_max(tf.concat([gto * decode_output[:self.emotion_num], (1-gto) * decode_output[self.emotion_num:]], 1)) #% self.vocab_size


    def train(self, input, output, emotionTag):

        def emotion_distribution(results):
            return tf.cast(results, dtype=tf.int64) / self.real_vocab_size
        def loss(results):
            loss =  tf.nn.softmax_cross_entropy_with_logits(logits=results, labels= self.vocab_label)
            loss += tf.nn.softmax_cross_entropy_with_logits(logits= emotion_distribution(results), labels= self.emotion_label)
            loss += 2 * tf.nn.l2_loss(self.internalMemory)
            return loss
        encoder_outputs = self.encode(inputs, mask, None, dropout = 1.0)[1]
        results = self.decode(encoder_outputs)
        tfloss = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss(results))
        return self.sess.run(tfloss, feed_dict={self.input:input, self.output:output, self.emotionTag: emotionTag})

    def test(self, input, output, emotionTag):
        encoder_outputs = self.encode(inputs, mask, None, dropout = 1.0)[1]
        results = self.decode(encoder_outputs)
        tfids = tf.arg_max(results)
        ids = self.sess.run(tfids, feed_dict={self.input:input, self.output:output, self.emotionTag: emotionTag})
        return [self.id2word[id] for each in ids]



