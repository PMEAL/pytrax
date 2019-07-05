# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:40:31 2017
Run the RandomWalk Examples
@author: Precision
"""
try:
    import porespy as ps
except ImportError:
    print('PoreSpy must be installed for the examples. ' +
          'Use pip or conda to install it')
    raise

import pytrax as pt
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request as ur
from io import BytesIO
from PIL import Image

save_figures = True
global_stride = 1
plt.close('all')
if __name__ == '__main__':
    # Change number to run different example of include many in list
    for image_run in [0, 1, 2]:
        if image_run == 0:
            # Open space
            im = np.ones([3, 3], dtype=np.int)
            fname = 'open_'
            num_t = 10000
            num_w = 10000
            stride = 10
        elif image_run == 1:
            # Load tau test image
            url = 'https://i.imgur.com/nrEJRDf.png'
            file = BytesIO(ur.urlopen(url).read())
            im = np.asarray(Image.open(file))[:, :, 3] == 0
            im = im.astype(np.int)
            im = np.pad(im, pad_width=50, mode='constant', constant_values=1)
            fname = 'tau_'
            # Number of time steps and walkers
            num_t = 20000
            num_w = 1000
            stride = 20
        elif image_run == 2:
            # Generate a Sierpinski carpet by tiling an image and blanking the
            # Middle tile recursively
            def tileandblank(image, n):
                if n > 0:
                    n -= 1
                    shape = np.asarray(np.shape(image))
                    image = np.tile(image, (3, 3))
                    image[shape[0]:2*shape[0], shape[1]:2*shape[1]] = 0
                    image = tileandblank(image, n)
                return image

            im = np.ones([1, 1], dtype=np.int)
            im = tileandblank(im, 4)
            fname = 'sierpinski_'
            # Number of time steps and walkers
            num_t = 2500
            num_w = 100000
            stride = 5
        elif image_run == 3:
            # Make an anisotropic image of 3D blobs
            im = ps.generators.blobs(shape=[300, 300, 300], porosity=0.5,
                                     blobiness=[1, 2, 5]).astype(np.int)
            fname = 'blobs_'
            # Number of time steps and walkers
            num_t = 10000
            num_w = 10000
            stride = 100
        elif image_run == 4:
            im = ps.generators.cylinders([300, 300, 300],
                                         radius=10,
                                         ncylinders=100,
                                         phi_max=90,
                                         theta_max=90).astype(np.int)
            fname = 'random_cylinders_'
            num_t = 10000
            num_w = 10000
            stride = 10
        elif image_run == 5:
            im = ps.generators.cylinders([300, 300, 300],
                                         radius=10,
                                         ncylinders=100,
                                         phi_max=0,
                                         theta_max=0).astype(np.int)
            fname = 'aligned_cylinders_'
            num_t = 10000
            num_w = 10000
            stride = 10
        print('Running Example: '+fname.strip('_'))
        # Override all strides
        if global_stride is not None:
            stride = global_stride
        # Track time of simulation
        st = time.time()
        rw = pt.RandomWalk(im, seed=False)
        rw.run(num_t, num_w, same_start=False, stride=stride, num_proc=8)
        print('run time', time.time()-st)
        rw.calc_msd()
        # Plot mean square displacement
        rw.plot_msd()
        dpi = 600
        if save_figures:
            rw._save_fig(fname+'msd.png', dpi=dpi)
        if rw.dim == 2:
            # Plot the longest walk
            rw.plot_walk_2d(w_id=np.argmax(rw.sq_disp[-1, :]), data='t')

            if save_figures:
                rw._save_fig(fname+'longest.png', dpi=dpi)
            # Plot all the walks
            rw.plot_walk_2d(check_solid=True, data='t')
            if save_figures:
                rw._save_fig(fname+'all.png', dpi=dpi)
        else:
            if save_figures:
                # export to paraview
                rw.export_walk(image=rw.im, sample=1)
