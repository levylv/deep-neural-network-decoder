#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mail:levy_lv@hotmail.com
# Lyu Wei @ 2017-07-26

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib

'''Plot the result about ber.
'''

matplotlib.rcParams.update({'font.size': 22})

N = 32
train_snr = np.arange(-2, 22, 2) # train_snr = np.arange(-3, 9)
test_snr = np.arange(0, 6.5, 0.5)
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
epoch_setting = np.array([10**1, 10**2, 10**3, 10**4, 10**5])
result = sio.loadmat('result/cnn_result')
res_ber = result['ber_trainRatio_trainSNR_testSNR_epoch']
map_ber = np.array([0.133306, 0.0983956, 0.0690256, 0.0447175, 0.0269225, 0.0156781, 0.00815875, 0.00397, 0.00195625, 0.000963125, 0.000375, 0.00014, 0.000054375])
line_style_1 = np.array(['co-','gd-','b^-','rs-']) # four line for train_ratio
line_style_2 = np.array(['yo-','cd-','g^-','bs-', 'r<-']) # five line for epoch_setting


# Plot the NVE of train_snr under train_ratio = 1.0
NVE = np.zeros([len(train_ratio), len(train_snr)])
#epoch_index = -1 
for i in range(len(train_ratio)):
    for j in range(len(train_snr)):
        for t in range(len(test_snr) - 0):
            # Set epoch = 10**5 
            NVE[i, j] += res_ber[i, j, t, -1] / map_ber[t]
        NVE[i, j] = NVE[i, j] / (len(test_snr) - 0)


best_train_snr_index = np.zeros(len(train_ratio))
for i in range(len(train_ratio)):
    best_train_snr_index[i] = np.argmin(NVE[i, :])


# Plot the ber of test_snr about train_ratio

plt.figure()
for i in range(len(train_ratio)):
    plt.semilogy(test_snr, int(res_ber[i, best_train_snr_index[i], :, -1]), line_style_1[i], lw = 3)

plt.semilogy(test_snr, map_ber, 'k>--', lw = 3)
plt.grid(True)
plt.legend(['$p = 40 \%$', '$p = 60 \%$', '$p = 80 \%$', '$p = 100 \%$', 'MAP'], prop={'size':18}, bbox_to_anchor=[0, 0], loc = 'lower left', borderaxespad=0.2)
plt.ylim(0.000001, 1)
plt.xlabel('$E_b/N_0$ (dB)')
plt.ylabel('BER')


# Show
plt.show()

