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
#train_snr = np.arange(-4, 6)
#test_snr = np.arange(7)
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
epoch_setting = np.array([10**1, 10**2, 10**3, 10**4, 10**5])
result = sio.loadmat('result/cnn_result')
res_ber = result['ber_trainRatio_epoch']
#sc_ber = 1/4 * np.array([413/1000, 243/1000, 263/2000, 252/4000, 263/10000, 234/44000,  226/214000])
line_style_1 = np.array(['co-','gd-','b^-','rs-']) # four line for train_ratio
line_style_2 = np.array(['yo-','cd-','g^-','bs-', 'r<-']) # five line for epoch_setting



plt.figure()
for i in range(len(train_ratio)):
    plt.semilogx(epoch_setting, res_ber[i, :], line_style_1[i], lw = 3)

#plt.semilogy(test_snr, sc_ber, 'k>--', lw = 3)
plt.grid(True)
plt.legend(['$p = 40 \%$', '$p = 60 \%$', '$p = 80 \%$', '$p = 100 \%$'], prop={'size':18}, borderaxespad=0.2)
plt.xticks(epoch_setting, ['$10^1$', '$10^2$','$10^3$', '$10^4$', '$10^5$'])
plt.ylim(0, 0.7)
plt.xlabel('$M_{ep}$')
plt.ylabel('BER')
#plt.title('N = 8 , r = 0.5')



# Show
plt.show()

