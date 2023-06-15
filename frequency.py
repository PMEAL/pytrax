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
import time

plt.close('all')
def main():
    im = ps.generators.blobs(shape=[1000, 1000], blobiness=3, porosity=0.5)
    im = ps.filters.fill_blind_pores(im).astype(int)
    dt = spim.distance_transform_edt(im)
    grey = dt.copy()/dt.max()
    grey = np.pad(grey, 1, mode='constant', constant_values=0)
    # Number of time steps and walkers
    num_t = 100000
    num_w = 1000
    stride = 1
    
    rw = pt.RandomWalk(grey, seed=False)
    rw.run(num_t, num_w, same_start=False, stride=stride, num_proc=12)
    # Plot mean square displacement
    rw.plot_msd()
    rw.plot_walk_2d()
    
    print('Calculating hit frequency')
    coords = rw.real_coords
    freq = np.zeros_like(grey)
    if len(im.shape) == 2:
        x_last = coords[0, :, 0].fill(-1)
        y_last = coords[0, :, 1].fill(-1)
        for t in range(coords.shape[0]):
            x = coords[t, :, 0]
            y = coords[t, :, 1]
            same_x = x == x_last
            same_y = y == y_last
            same_xy = same_x * same_y
            freq[x[~same_xy], y[~same_xy]] += 1
            x_last = x
            y_last = y
    else:
        x_last = coords[0, :, 0].fill(-1)
        y_last = coords[0, :, 1].fill(-1)
        z_last = coords[0, :, 2].fill(-1)
        for t in range(coords.shape[0]):
            x = coords[t, :, 0]
            y = coords[t, :, 1]
            z = coords[t, :, 2]
            same_x = x == x_last
            same_y = y == y_last
            same_z = z == z_last
            same_xyz = same_x * same_y * same_z
            freq[x[~same_xyz], y[~same_xyz], z[~same_xyz]] += 1
            x_last = x
            y_last = y
            z_last = z
    return grey, freq

if __name__ == '__main__':
    st = time.time()
    grey, freq = main()
    some_hits = freq[freq > 0]
    frange = np.unique(some_hits)
    plt.figure()
    plt.hist(some_hits, bins=int(frange.max()-frange.min()))
    log_freq = np.log(freq)
    log_freq[freq == 0] = np.nan
    freq[freq == 0] = np.nan
    print(frange.min(), frange.max())
    plt.figure()
    plt.imshow(grey > 0, cmap='gist_gray')
    plt.imshow(freq)
    plt.colorbar()
    plt.figure()
    plt.imshow(grey > 0, cmap='gist_gray')
    plt.imshow(log_freq)
    plt.colorbar()
    print('Sim time', time.time() - st)