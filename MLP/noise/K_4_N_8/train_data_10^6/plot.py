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

N = 8
train_snr = np.arange(-2, 22, 2)
test_snr = np.arange(0, 6.5, 0.5)
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
epoch_setting = np.array([10**1, 10**2, 10**3, 10**4, 10**5])
result = sio.loadmat('result/mlp_result')
res_ber = result['ber_trainRatio_trainSNR_testSNR_epoch']
map_ber = np.array([0.09104, 0.0727475, 0.05570, 0.0419, 0.0306, 0.02036, 0.013125,0.0085125, 0.0047125, 0.0026825, 0.0012475, 0.0006825, 0.000215])
line_style_1 = np.array(['co-','gd-','b^-','rs-']) # four line for train_ratio
line_style_2 = np.array(['yo-','cd-','g^-','bs-', 'r<-']) # five line for epoch_setting


# Calculate the NVE of train_snr
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
    plt.semilogy(test_snr, res_ber[i, best_train_snr_index[i], :, -1], line_style_1[i], lw = 3)

plt.semilogy(test_snr, map_ber, 'k>--', lw = 3)
plt.grid(True)
plt.legend(['$p = 40 \%$', '$p = 60 \%$', '$p = 80 \%$', '$p = 100 \%$', 'MAP'], prop={'size':18}, bbox_to_anchor=[0, 0], loc = 'lower left', borderaxespad=0.2)
plt.ylim(0.00001, 1)
plt.xlabel('$E_b/N_0$ (dB)')
plt.ylabel('BER')

# Show
plt.show()

