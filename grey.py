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
    im = ps.generators.blobs(shape=[1000, 1000], porosity=0.7).astype(np.int)
    dt = spim.distance_transform_edt(im)
    grey = dt.copy()/dt.max()
    # Number of time steps and walkers
    num_t = 10000
    num_w = 800
    stride = 1
    
    for case in [im, grey]:
        rw = pt.RandomWalk(case, seed=False)
        rw.run(num_t, num_w, same_start=False, stride=stride, num_proc=8)
        # Plot mean square displacement
        rw.plot_msd()
        rw.plot_walk_2d()
