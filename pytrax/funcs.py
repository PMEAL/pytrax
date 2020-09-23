# %% Import packages and define helper functions
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt


def new_vector():
    # Generate theta and phi
    q, f = np.random.rand(2)*360
    # Convert to axial components of a unit vector displacement
    z = np.cos(np.deg2rad(f))
    y = np.sin(np.deg2rad(f))*np.cos(np.deg2rad(q))
    x = np.sin(np.deg2rad(f))*np.sin(np.deg2rad(q))
    return x, y, z


def get_start_point(im):
    while True:
        start = [np.random.randint(0, im.shape[i], 1) for i in range(im.ndim)]
        start = np.array(start).flatten()
        if im[tuple(start)]:
            return start


# %% Generate image

im = ps.generators.overlapping_spheres(shape=[500, 500], radius=5, porosity=0.9)
bd = ps.tools.get_border(im.shape, thickness=3, mode='faces')
im[bd] = False

# %% Run walk

mfp = 100  # mean free path
im_path = np.zeros_like(im, dtype=int)
start = get_start_point(im)
x, y, z = new_vector()
path = [start]
loc = np.copy(start)
i = 0
while i < 100000:
    i += 1
    new_loc = loc + np.array([x, y])
    check_1 = im[tuple(new_loc.astype(int))] == False
    check_2 = np.sqrt(np.sum((new_loc-start)**2)) > mfp
    if check_1 or check_2:
        x, y, z = new_vector()
        start = np.copy(loc)
    else:
        loc = new_loc
        path.append(loc)
        im_path[tuple(loc.astype(int))] = i

plt.imshow(im_path/im, origin='xy')
