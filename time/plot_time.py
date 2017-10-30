#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mail:levy_lv@hotmail.com
# Lyu Wei @ 2017-07-29

'''Plot the computation time of NND
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 22})


N = np.array([8, 16, 24])
forward_time_MLP = np.array([0.0781, 0.1042, 0.1132]) / 10**5
backward_time_MLP = np.array([0.0014, 0.0016, 0.0016]) / 128
forward_time_CNN = np.array([0.1093, 0.2561, 0.4393]) / 10**5
backward_time_CNN = np.array([0.0016, 0.0023, 0.0032]) / 128
forward_time_RNN = np.array([5.9499, 11.7834, 23.4094]) / 10**5
backward_time_RNN = np.array([0.0251, 0.0480, 0.0918]) / 128

bar_width = 2
opacity = 0.4
plt.figure()
ax = plt.gca() 
ax.yaxis.get_major_formatter().set_powerlimits((0,1)) 

plt.bar(N[0:3], backward_time_MLP, width = bar_width, alpha = opacity, color = 'r', label = 'MLP')
plt.bar(N[0:3]+bar_width, backward_time_CNN,width = bar_width, alpha = opacity, color = 'g', label = 'CNN')
plt.bar(N[0:3]+2*bar_width, backward_time_RNN, width = bar_width, alpha = opacity, color = 'b', label = 'RNN')
plt.ylabel('Computational time (s)')
plt.xticks(N[0:3]+bar_width, ['N = 8', 'N = 16', 'N = 32'])
plt.xlim(6, 32)
plt.legend(loc = 'upper left')
#plt.title('Backward time for every batch, batch size = 128')


plt.figure()
ax = plt.gca() 
ax.yaxis.get_major_formatter().set_powerlimits((0,1)) 

plt.bar(N[0:3], forward_time_MLP, width = bar_width, alpha = opacity, color = 'r', label = 'MLP')
plt.bar(N[0:3]+bar_width, forward_time_CNN,width = bar_width, alpha = opacity, color = 'g', label = 'CNN')
plt.bar(N[0:3]+2*bar_width, forward_time_RNN, width = bar_width, alpha = opacity, color = 'b', label = 'RNN')
plt.ylabel('Computational time (s)')
plt.xticks(N[0:3]+bar_width, ['N = 8', 'N = 16', 'N = 32'])
plt.xlim(6, 32)
plt.legend(loc = 'upper left')
#plt.title('Forward time, test number = $10^5$')

plt.show()

