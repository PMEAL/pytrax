# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:45:35 2020

@author: tom
"""


import pytrax as pt
import porespy as ps
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.close('all')
    num_t = 10000
    num_w = 1000
    
    # %% Generate image
    im = ps.generators.overlapping_spheres(shape=[400, 400], radius=10, porosity=0.65)
    im = ps.filters.fill_blind_pores(im)
    bd = ps.tools.get_border(shape=im.shape, mode='faces')
    im = ps.filters.trim_nonpercolating_paths(im, inlets=bd, outlets=bd)
    rw = pt.RandomWalk(im, seed=False)
    for mfp in [10]:
        rw.run(num_t, num_w, same_start=False, stride=1, num_proc=10, mean_free_path=mfp)
        rw.calc_msd()
        rw.plot_walk_2d()
        rw.plot_msd()