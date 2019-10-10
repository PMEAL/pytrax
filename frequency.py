# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:34:14 2019

@author: Tom
"""

import porespy as ps
import pytrax as pt
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim
plt.close('all')
if __name__ == '__main__':
    im = ps.generators.blobs(shape=[1000, 1000], porosity=0.7).astype(np.int)
    dt = spim.distance_transform_edt(im)
    grey = dt.copy()/dt.max()
    grey = np.pad(grey, 1, mode='constant', constant_values=0)
    # Number of time steps and walkers
    num_t = 10000
    num_w = 10000
    stride = 1
    
    rw = pt.RandomWalk(grey, seed=False)
    rw.run(num_t, num_w, same_start=False, stride=stride, num_proc=12)
    # Plot mean square displacement
    rw.plot_msd()
    rw.plot_walk_2d()
    
    print('Calculating hit frequency')
    coords = rw.real_coords + 1
    freq = np.zeros_like(grey)
    if len(im.shape) == 2:
        for t in range(coords.shape[0]):
            x = coords[t, :, 0]
            y = coords[t, :, 1]
            freq[x, y] += 1
    else:
        for t in range(coords.shape[0]):
            x = coords[t, :, 0]
            y = coords[t, :, 1]
            z = coords[t, :, 2]
            freq[x, y, z] += 1
    
    some_hits = freq[freq > 0]
    frange = np.unique(some_hits)
    plt.figure()
    plt.hist(some_hits, bins=100)
    log_freq = np.log(freq)
    log_freq[freq == 0] = np.nan
    print(frange.min(), frange.max(), log_freq[freq > 0].min(), log_freq[freq > 0].max())
    plt.figure()
    plt.imshow(grey > 0, cmap='gist_gray')
    plt.imshow(log_freq)
    plt.colorbar()
    
        