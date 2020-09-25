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
    num_w = 5
    im = ps.generators.overlapping_spheres(shape=[400, 800], radius=10, porosity=0.9)
    bd = ps.tools.get_border(im.shape, thickness=3, mode='faces')
    im[bd] = False
    rw = pt.RandomWalk(im, seed=False)
    for mfp in [1, 2, 3]:
        rw.run(num_t, num_w, same_start=False, stride=1, num_proc=1, mean_free_path=mfp)
        rw.calc_msd()
        rw.plot_walk_2d()
        rw.plot_msd()