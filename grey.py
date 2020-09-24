# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:07:13 2019

@author: Tom
"""

import porespy as ps
import pytrax as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
if __name__ == '__main__':
    im = ps.generators.blobs(shape=[200, 200], porosity=0.5).astype(np.int)
    plt.figure()
    plt.imshow(im)
    dt = spim.distance_transform_edt(im)
    grey = dt.copy()/dt.max()
    # Number of time steps and walkers
    num_t = 100000
    num_w = None
    stride = 10
    
    for case in [im]:
        rw = pt.RandomWalk(case, seed=False)
        rw.run(num_t, num_w, same_start=False, stride=stride, num_proc=10)
        # Plot mean square displacement
        rw.plot_msd()
        # rw.plot_walk_2d()
        rw.colour_sq_disp()
        plt.figure()
        plt.imshow(rw.im_sq_disp)
