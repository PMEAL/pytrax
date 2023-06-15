# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:09:18 2020

@author: Tom
"""

import pytrax as pt
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
if __name__ == '__main__':
    n = 5
    im_chess = np.ones([10*n, 10*n])
    g = 0.25
    for i in range(10*n):
        for j in range(10*n):
            if (np.floor(i/n) % 2) != (np.floor(j/n) % 2):
                im_chess[i, j] = g
    plt.figure()
    plt.imshow(im_chess)
    rw_c = pt.RandomWalk(im_chess)
    rw_c.run(nt=10000, nw=10000, num_proc=1, same_start=True)
    rw_c.calc_msd()
    rw_c.plot_msd()
    rw_c.plot_walk_2d()
    rw_c.plot_walk_2d(w_id=np.argmin(rw_c.sq_disp[-1, :]), data='t')
    rw_c.plot_walk_2d(w_id=np.argmax(rw_c.sq_disp[-1, :]), data='t')

    im = np.ones([10*n, 10*n])
    for i in range(10*n):
        if (np.floor(i/n) % 2 == 0):
                im[i, :] = g
    plt.figure()
    plt.imshow(im)
    rw = pt.RandomWalk(im)
    rw.run(nt=10000, nw=10000, stride=10, num_proc=1, same_start=True)
    rw.calc_msd()
    rw.plot_msd()
    rw.plot_walk_2d()
    rw.plot_walk_2d(w_id=np.argmin(rw.sq_disp[-1, :]), data='t')
    rw.plot_walk_2d(w_id=np.argmax(rw.sq_disp[-1, :]), data='t')