#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mail:levy_lv@hotmail.com
# Lyu Wei @ 2017-10-10

'''
This is a LSTM decoder
'''

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import scipy.io as sio
import math
import time
from datetime import datetime

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def lstm(inputs,
         N,
         K,
         batch_size,
         dropout_keep_prob,
         scope = 'RNN'
         ):
    '''A one-layer LSTM, num_time = N, inputs = [batch_size, N, 1], outputs = [batch_size, N, hidden_size]
    '''
    num_layer = 1
    num_time = N
    hidden_size = 256
    def attn_cell():
        lstm_cell = rnn.BasicLSTMCell(num_units = hidden_size, forget_bias = 1.0, state_is_tuple = True, reuse = tf.get_variable_scope().reuse)
        lstm_cell = rnn.DropoutWrapper(cell = lstm_cell, output_keep_prob = dropout_keep_prob)
        return lstm_cell

    mlstm_cell = rnn.MultiRNNCell([attn_cell() for _ in range(num_layer)], state_is_tuple = True)
    init_state = mlstm_cell.zero_state(batch_size, dtype = tf.float32)
    outputs = list()
    state = init_state
    with tf.variable_scope(scope):
        for time_step in range(num_time):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)

    h_state = outputs[-1]
    W = tf.Variable(tf.truncated_normal([hidden_size, K], stddev = 0.1), dtype = tf.float32)
    bias = tf.Variable(tf.zeros(K), dtype = tf.float32)
    y = tf.nn.sigmoid(tf.matmul(h_state, W) + bias)
    
    return y


def get_random_batch_data(x, y, batch_size):
    '''get random batch data from x and y, which have the same length
    '''
    index = np.random.randint(0, len(x) - batch_size)
    return x[index:(index+batch_size)], y[index:(index+batch_size)]


# Parameters setting
N = 32
K = 16
data_path = 'data/'
num_epoch = 10**5
train_batch_size = 128
train_snr = np.arange(-2, 22, 2)
test_snr = np.arange(0, 6.5, 0.5)
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
epoch_setting = np.array([10**1, 10**2, 10**3, 10**4, 10**5])
res_ber = np.zeros([len(train_ratio), len(train_snr), len(test_snr), len(epoch_setting)])


# make the lstm model
keep_prob = tf.placeholder(tf.float32)
coded_words = tf.placeholder(tf.float32, [None, N])
labels = tf.placeholder(tf.float32, [None, K])
batch_size = tf.placeholder(tf.int32, []) # we need different batch size for train and test
x_input = tf.reshape(coded_words, [-1, N, 1])

logits = lstm(x_input, N, K, batch_size, keep_prob)

print('LSTM: input = %d, output = %d, batch_size = %d' % (N, K, train_batch_size))

# the Multi-label classification
total_loss = tf.losses.mean_squared_error(labels, logits) # MSE loss

optimizer = tf.train.AdamOptimizer(1e-3).minimize(total_loss)

prediction = tf.cast(logits > 0.5, tf.float32)
correct_prediction = tf.equal(prediction, labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
ber = 1.0 - accuracy

sess = tf.Session()
init = tf.global_variables_initializer()

# Get the different data of train and test
backward_batch_time_total = 0.0
forward_time_total = 0.0
for ratio_index in range(len(train_ratio)):
    for tr_snr_index in range(len(train_snr)):
        train_filename = 'ratio_' + str(train_ratio[ratio_index]) + '_train_snr_' + str(train_snr[tr_snr_index]) + 'dB.mat'
        train_data = sio.loadmat(data_path+train_filename)
        x_train = train_data['x_train']
        y_train = train_data['y_train']

        # start session
        sess.run(init)
        print('---------------------------------')
        print('New begining')
        print('---------------------------------')
        for epoch in range(num_epoch):
            x_batch, y_batch = get_random_batch_data(x_train, y_train, train_batch_size)

            # Set keep_prob = 0.9 for training
            backward_batch_time_start = time.time()
            _, train_ber = sess.run([optimizer, ber], feed_dict = {coded_words : x_batch, labels : y_batch, keep_prob : 0.8, batch_size : train_batch_size})
            duration = time.time() - backward_batch_time_start
            backward_batch_time_total += duration

            if (epoch+1) % 1000 == 0:
                print('%s: epoch = %d, train_ratio = %f, train_snr = %f dB ---> train_ber = %.4f, forward time = %.3f sec' % (datetime.now(), epoch+1, train_ratio[ratio_index], train_snr[tr_snr_index], train_ber, duration))

            if epoch+1 in epoch_setting:
                epoch_index = np.where(epoch_setting == (epoch+1))[0][0]

                print('\n')
                print('***********TEST BEGIN************')
                for te_snr_index in range(len(test_snr)):
                    test_filename = 'test_snr_' + str(test_snr[te_snr_index]) + 'dB.mat'
                    test_data = sio.loadmat(data_path+test_filename)
                    x_test = test_data['x_test']
                    y_test = test_data['y_test']
                    
                    # Set keep_prob = 1.0 for testing
                    forward_time_start = time.time()
                    res = sess.run(ber, feed_dict = {coded_words : x_test, labels: y_test, keep_prob : 1.0, batch_size : len(x_test)})
                    duration = time.time() - forward_time_start
                    forward_time_total += duration
                    print('%s: epoch = %d, train_ratio = %f, train_snr = %f dB, test_snr = %f dB ---> ber = %.4f, forward time = %.3f sec' % (datetime.now(), epoch+1, train_ratio[ratio_index], train_snr[tr_snr_index], test_snr[te_snr_index], res, duration))
                    res_ber[ratio_index, tr_snr_index, te_snr_index, epoch_index] = res
                print('***********TEST END**********')
                print('\n')

# Statistics
backward_batch_time_avg = backward_batch_time_total / (num_epoch * len(train_ratio) * len(train_snr))
forward_count = int(len(epoch_setting) * len(test_snr) * len(train_ratio) * len(train_snr))
forward_time_avg = forward_time_total / forward_count

print('\n')
print('*****************END***************')
print('For the backward time of LSTM:')
print('batch size = %d, total batch = %d, time = %.3f sec/batch' % (train_batch_size, num_epoch, backward_batch_time_avg))
print('For the forward time of LSTM:')
print('test number = %d, total test = %d, time = %.3f sec' % (len(x_test), forward_count, forward_time_avg))
sio.savemat('result/lstm_result', {'ber_trainRatio_trainSNR_testSNR_epoch' : res_ber, 'backward_batch_time_avg' : backward_batch_time_avg, 'forward_time_avg' : forward_time_avg})
