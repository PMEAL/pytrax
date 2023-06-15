# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:07:23 2020

@author: tom
"""

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.ndimage as spim
    import pytrax as pt
    from scipy.interpolate import NearestNDInterpolator
    
    dots = np.zeros([1000, 1000])
    for i in range(10):
        for j in range(10):
            adjx = np.random.choice(np.arange(-25, 25, 1, int), 1)[0]
            adjy = np.random.choice(np.arange(-25, 25, 1, int), 1)[0]
            dots[(100*i)+50+adjx, (100*j)+50+adjy] = 1
    
    dt = spim.distance_transform_edt(dots)
    strel = spim.generate_binary_structure(2, 1)
    big_dots = spim.morphology.binary_dilation(dots, strel, 10)
    plt.figure()
    plt.imshow(big_dots)
    dt = spim.distance_transform_edt(1-big_dots)
    dt = dt/dt.max()
    dt = 1 - dt
    plt.figure()
    plt.imshow(dt)
    plt.figure()
    plt.hist(dt.flatten())
    
    def grey_scaler(im, lower, upper, exponent):
        im_scaled = im.copy().astype(float)
        # below lower is pore
        im_scaled[im < lower] = lower
        # above upper is nmc
        im_scaled[im > upper] = upper
        im_scaled -= lower
        # normalize scale
        im_scaled /= (upper-lower)
        # invert image
        im_scaled = (1-im_scaled)
        # apply exponent to grey values - higher exponent means slower transport in regions near nmc intensity
        im_scaled = im_scaled**exponent
        return im_scaled

    def process(im, thresh):
        tmp = im < thresh
        tmp[:10, :] = True
        tmp[-10:, :] = True
        lab, N = spim.label(tmp)
        return lab == 1

    def process_scaled(im, thresh, trange):
        tmp = grey_scaler(im, thresh, thresh+trange, 1.0)
        tmp[:10, :] = 1.0
        tmp[-10:, :] = 1.0
        tmp_bin = tmp > 0.0
        lab, N = spim.label(tmp_bin)
        tmp[lab > 1] = 0.0
        return tmp
    
    def tortuosity(image):
        rw = pt.RandomWalk(image)
        rw.run(nt=100000, nw=10000, stride=10, num_proc=10)
        rw.calc_msd()
        rw.plot_msd()
        return rw
    
    def interpolated_sq_disp(rw):
        rw.colour_sq_disp()
        im = rw.im_sq_disp
        x_len, y_len = im.shape
        points = np.argwhere(im > 1)
        colours = im[points[:, 0], points[:, 1]]
        myInterpolator = NearestNDInterpolator(points, colours)
        grid_x, grid_y = np.mgrid[0:x_len:np.complex(x_len, 0),
                                  0:y_len:np.complex(y_len, 0)]
        arr = np.log(myInterpolator(grid_x, grid_y).astype(float))
        arr[im == 0] = np.nan
        plt.figure()
        plt.imshow(arr)
        plt.colorbar()

    plt.figure()
    plt.imshow(dt)
    # thresh = np.linspace(0.525, 0.55, 3)
    thresh = [0.6]
    rws = []
    ims = []
    for i in range(len(thresh)):
        plt.figure()
        tmp = process(dt, thresh[i])
        plt.imshow(tmp)
        plt.title('Threshold: '+str(np.around(thresh[i], 3))+' Porosity: '+str(np.around(np.sum(tmp)/np.size(tmp), 3)))
        rws.append(tortuosity(tmp))
        ims.append(tmp)
    tau_0 = [r.data['axis_0_tau'] for r in rws]
    # grey_rws = []
    # grey_ims = []
    # for i in range(len(thresh)):
    #     plt.figure()
    #     tmp = process_scaled(dt, thresh[i], 0.05)
    #     plt.imshow(tmp)
    #     plt.title('Threshold: '+str(np.around(thresh[i], 3))+' Porosity: '+str(np.around(np.sum(tmp)/np.size(tmp), 3)))
    #     grey_rws.append(tortuosity(tmp))
    #     grey_ims.append(tmp)
    # grey_tau_0 = [r.data['axis_0_tau'] for r in grey_rws]
    plt.figure()
    plt.plot(tau_0)
    # plt.plot(grey_tau_0)