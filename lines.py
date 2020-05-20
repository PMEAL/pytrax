# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:44:46 2020

@author: Tom
"""
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.external import tifffile
    import pytrax as pt
    import openpnm as op
    import time
    plt.close('all')
    w = 10
    h = int(w/2)
    N = 20*w
    D0 = 1.0
    f = 0.1
    im = np.ones([N, N])
    for i in range(N):
        if np.floor((i+h)/w) % 2 == 0:
            im[:, i] = f
    plt.figure()
    plt.imshow(im)
    t1 = time.time()
    rw = pt.RandomWalk(im)
    rw.run(nt=10000, nw=10000, stride=10, num_proc=10, same_start=False)
#    rw.calc_msd()
    rw.plot_msd()
    t2 = time.time()
#    phase_a = im == 1
#    walkers = rw._walkers
#    walkers_in_phase_a = np.sum(phase_a[walkers[:, 0], walkers[:, 1]])
    rw.plot_walk_2d()
#    im_int = im.copy()
#    im_int[im_int == 1.0] = 2
#    im_int[im_int == f] = 1
#    tifffile.imsave('lines.tif', im_int.astype(np.uint8))
    print('Deff_s random walk', np.around(1/rw.data['axis_1_tau'], 3))
    print('Deff_p random walk', np.around(1/rw.data['axis_0_tau'], 3))
    net = op.network.Cubic(shape=[N, N, 1], spacing=1.0)
    phase = op.phases.GenericPhase(network=net)
    net['pore.grey'] = im.flatten()
    conns = net['throat.conns']
    P1 = conns[:, 0]
    P2 = conns[:, 1]
    P1g = net['pore.grey'][P1]
    P2g = net['pore.grey'][P2]
    phase['throat.grey'] = 2*(P1g*P2g)/(P1g+P2g)
    fick_s = op.algorithms.FickianDiffusion(network=net)
    fick_s.setup(phase=phase, quantity='pore.conc', conductance='throat.grey')
    fick_s.set_value_BC(net.pores('left'), 1.0)
    fick_s.set_value_BC(net.pores('right'), 0.0)
    fick_s.run()
    print('Deff_s pnm', np.around(fick_s.calc_effective_diffusivity(domain_area=N, domain_length=N)[0], 3))
    fick_p = op.algorithms.FickianDiffusion(network=net)
    fick_p.setup(phase=phase, quantity='pore.conc', conductance='throat.grey')
    fick_p.set_value_BC(net.pores('back'), 1.0)
    fick_p.set_value_BC(net.pores('front'), 0.0)
    fick_p.run()
    print('Deff_p pnm', np.around(fick_p.calc_effective_diffusivity(domain_area=N, domain_length=N)[0], 3))
    t3 = time.time()
    print('RW time', np.around(t2-t1, 3), 'PNM time', np.around(t3-t2, 3))