
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import config_tf as config

class ChatBotModel(object):
    def __init__(self, forward_only, batch_size):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size
    
    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = self.decoder_inputs[1:]
        
    def _inference(self):
        print('Create inference')
     

        single_cell = tf.nn.rnn_cell.GRUCell(config.HIDDEN_SIZE_40)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.NUM_LAYERS_3)

    def _create_loss(self):
        self.softmax_loss_function = tf.nn.softmax
        single_cell = tf.nn.rnn_cell.GRUCell(config.HIDDEN_SIZE_40)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.NUM_LAYERS_3)
        print('Creating loss...')
        start = time.time()
        if self.fw_only:
            self.outputs, self.state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                       self.encoder_inputs,
                       self.decoder_inputs,
                       self.cell,
                       num_encoder_symbols=config.ENC_VOCAB,
                       num_decoder_symbols=config.DEC_VOCAB,
                       embedding_size=config.HIDDEN_SIZE_40,
                       output_projection=None,
                       feed_previous= True)

        else:
            self.outputs, self.state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                       self.encoder_inputs,
                       self.decoder_inputs,
                       self.cell,
                       num_encoder_symbols=config.ENC_VOCAB,
                       num_decoder_symbols=config.DEC_VOCAB,
                       embedding_size=config.HIDDEN_SIZE_40,
                       output_projection=None,
                       feed_previous= False)
            self.losses = tf.losses.softmax_cross_entropy(self.decoder_inputs, self.outputs)

        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer...')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                grad_and_vars = self.optimizer.compute_gradients(self.losses)
                norm = tf.global_norm(tf.gradients(self.losses, trainables))
                self.gradient_norms.append(norm)
                self.train_ops.append(self.optimizer.apply_gradients(grad_and_vars, 
                                                            global_step=self.global_step))
                start = time.time()


    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        #self._inference()
        self._create_loss()
        self._creat_optimizer()
       # self._create_summary()
