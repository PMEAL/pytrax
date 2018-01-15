# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:40:31 2017
Run the RandomWalk Examples
@author: Precision
"""
import porespy as ps
import pytrax as pt
import time
import numpy as np
import matplotlib.pyplot as plt

save_figures = False
global_stride = None
plt.close('all')
if __name__ == '__main__':
    for image_run in [3]:
        if image_run == 0:
            # Open space
            im = np.ones([3, 3], dtype=int)
            fname = 'open_'
            num_t = 10000
            num_w = 10000
            stride = 10
        elif image_run == 1:
            # Load tau test image
            im = 1 - ps.data.tau()
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

            im = np.ones([1, 1], dtype=int)
            im = tileandblank(im, 4)
            fname = 'sierpinski_'
            # Number of time steps and walkers
            num_t = 2500
            num_w = 100000
            stride = 5
        else:
            # Make an anisotropic image of 3D blobs
            im = ps.generators.blobs(shape=[300, 300, 300], porosity=0.5,
                                     blobiness=[1, 2, 5]).astype(int)
            fname = 'blobs_'
            # Number of time steps and walkers
            num_t = 10000
            num_w = 10000
            stride = 100
        print('Running Example: '+fname.strip('_'))
        # Override all strides
        if global_stride is not None:
            stride = global_stride
        # Track time of simulation
        st = time.time()
        rw = pt.RandomWalk(im, seed=False)
        rw.run(num_t, num_w, same_start=False, stride=stride, num_proc=10)
        print('run time', time.time()-st)
        rw.calc_msd()
        # Plot mean square displacement
        rw.plot_msd()
        dpi = 600
        if save_figures:
            rw._save_fig(fname+'msd.png')
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
                # pass
                rw.export_walk(image=rw.im, sample=1)
        nstrides = np.shape(rw.real_coords)[0]-1
        steps = [np.int(np.floor(nstrides*i/4)) for i in np.arange(0, 5, 1)]
        for step in steps:
            rw.axial_density_plot(time=step, bins=50)
            plt.title('Timestep: '+str(step*stride))
            if save_figures:
                rw._save_fig(fname+'density_'+str(step*stride)+'.png', dpi=dpi)
