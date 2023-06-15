# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:51:10 2020

@author: Tom
"""

import pytrax as pt
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

if __name__ == '__main__':
    im = np.ones([1000, 1000])
    # Number of time steps and walkers
    num_t = 10000
    num_w = 800
    stride = 1
    grey_vals = np.logspace(-1, 1, 10)
    tau = []
    for grey_val in grey_vals:
        grey = im.copy()
        grey = grey.astype(float)
        grey[grey == 1.0] = grey_val
        rw = pt.RandomWalk(grey, seed=False)
        rw.run(num_t, num_w, same_start=False, stride=stride, num_proc=1)
        rw.plot_msd()
        tau.append(rw.data['axis_0_tau'])
    plt.figure()
    plt.loglog(grey_vals, tau)
