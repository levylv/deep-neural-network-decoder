#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mail:levy_lv@hotmail.com
# Lyu Wei @ 2017-07-24

import numpy as np
import scipy.io as sio
import math
import time
from datetime import datetime

''' 
Data parameters
'''
K = 16 
N = 32
num_train = 10**6
num_test = 10**5
num_total = 2**K
#train_snr = np.arange(-4, 6)
#test_snr = np.arange(7)
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])


'''
Create all possible information words
'''

def add_bool(a,b):
    if len(a) != len(b):
        raise ValueError('arrays with different length')
    k = len(a)
    s = np.zeros(k,dtype=bool)
    c = False
    for i in reversed(range(0,k)):
        s[i], c = full_adder(a[i],b[i],c)    
    if c:
        warnings.warn("Addition overflow!")
    return s

def full_adder(a,b,c):
    s = (a ^ b) ^ c
    c = (a & b) | (c & (a ^ b))
    return s,c

def inc_bool(a):
    k = len(a)
    increment = np.hstack((np.zeros(k-1,dtype=bool), np.ones(1,dtype=bool)))
    a = add_bool(a,increment)
    return a


def bitrevorder(x):
    m = np.amax(x)
    n = np.ceil(np.log2(m)).astype(int)
    for i in range(0,len(x)):
        x[i] = int('{:0{n}b}'.format(x[i],n=n)[::-1],2)  
    return x

def bool2int(x):
    integer = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            integer[i, j] = int(x[i, j])
    return integer


def bpsk(x):
    return 1 - 2*x

def add_noise(x, SNR):
    sigma = np.sqrt(1/(10**(SNR/10)))
    if len(x.shape) == 1:
        w = sigma * np.random.randn(x.shape[0])
    else:
        w = sigma * np.random.randn(x.shape[0], x.shape[1])
    return x + w

def polar_transform_iter(u):
    N = len(u)
    index =  np.log2(N).astype(int)
    fn = 1
    f = np.array([[1,0],[1,1]])
    i2 = np.array([[1,0],[0,1]])
    b = i2
    for i in range(1,index+1):
        fn = np.kron(fn, f)

    for i in range(2, index+1):
        rn = np.zeros((2**i, 2**i))
        for j in range(1, 2**(i-1)+1):
            rn[2*j-2,j-1] = 1
        for j in range(1, 2**(i-1)+1):
            rn[2*j-1, j+2**(i-1)-1] = 1
        b = rn.dot(np.kron(i2, b))
    g = b.dot(fn)
    return u.dot(g) % 2

def polar_design_awgn(N, k, design_snr_dB):  
        
    S = 10**(design_snr_dB/10)
    z0 = np.zeros(N)

    z0[0] = np.exp(-S)
    for j in range(1,int(np.log2(N))+1):
        u = 2**j
        for t in range(0,int(u/2)):
            T = z0[t]
            z0[t] = 2*T - T**2     # upper channel
            z0[int(u/2)+t] = T**2  # lower channel
        
    # sort into increasing order
    idx = np.argsort(z0)
        
    # select k best channels
    idx = np.sort(bitrevorder(idx[0:k]))
    
    A = np.zeros(N, dtype=bool)
    A[idx] = True
        
    return A

d = np.zeros((num_total,K), dtype=bool)
for i in range(1,num_total):
    d[i]= inc_bool(d[i-1])

A = polar_design_awgn(N, K, design_snr_dB=0)
u = np.zeros((num_total, N),dtype=bool)
u[:,A] = d
u = bool2int(u)
d = bool2int(d)  # origin word

x_data = np.zeros((num_total, N),dtype=int)

for i in range(0,num_total):
    x_data[i] = polar_transform_iter(u[i])  # coded word


'''
Create train and test data for different snr and ratio
'''

# Create train data from the proportional words
start_time = time.time()
print(datetime.now(), 'start save train data')
for ratio in train_ratio:
    num_train_selected = int(num_total * ratio)
    random_index1 = np.random.permutation(num_total)
    x_train_selected = x_data[random_index1[0:num_train_selected]]
    y_train_selected = d[random_index1[0:num_train_selected]]

#    for snr in train_snr:
    x_train = np.zeros([num_train, N])
    y_train = np.zeros([num_train, K])
    for i in range(num_train):
        random_index2 = np.random.randint(num_train_selected)
        x_train[i] = bpsk(x_train_selected[random_index2])
        y_train[i] = y_train_selected[random_index2]
        
    filename = 'ratio_' + str(ratio) + '.mat'
    sio.savemat('data/'+filename, {'x_train' : x_train, 'y_train' : y_train})

duration = time.time() - start_time
print(datetime.now(), 'endding, duration time is %f /sec' % duration)


# Create test data from all possible words
start_time = time.time()
print(datetime.now(), 'start save test data')

#for snr in test_snr:
x_test = np.zeros([num_test, N])
y_test = np.zeros([num_test, K])
for i in range(num_test):
    random_index3 = np.random.randint(num_total)
    x_test[i] = bpsk(x_data[random_index3])
    y_test[i] = d[random_index3]

filename = 'test.mat'
sio.savemat('data/'+filename, {'x_test' : x_test, 'y_test' : y_test})
        
duration = time.time() - start_time
print(datetime.now(), 'endding, duration time is %f /sec' % duration)

