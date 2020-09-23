import numpy as np
import porespy as ps
import matplotlib.pyplot as plt


im = ps.generators.overlapping_spheres(shape=[250, 350], radius=5, porosity=0.9)
bd = ps.tools.get_border(im.shape, thickness=3, mode='faces')
im[bd] = False

# Generate starting points
while True:
    start = np.random.randint(0, 50, 2)
    if im[tuple(start)]:
        break


def new_end_point(start):
    # Generate an end point
    x_end = np.random.randint(0, im.shape[0], 1)
    y_end = np.random.randint(0, im.shape[1], 1)
    try:
        z_end = np.random.randint(0, im.shape[2], 1)
        end = np.array([x_end, y_end, z_end]).flatten()
    except IndexError:
        end = np.array([x_end, y_end]).flatten()

    H = np.sqrt(np.sum((end-start)**2))
    try:
        f = np.rad2deg(np.arcsin((end-start)[2]/H))
        h = H*np.cos(np.deg2rad(f))
        q = np.rad2deg(np.arcsin((end-start)[1]/h))
    except IndexError:
        q = np.rad2deg(np.arcsin((end-start)[0]/H))
        print(q)
    try:
        z = np.cos(np.deg2rad(f))
        y = np.sin(np.deg2rad(f))*np.cos(np.deg2rad(q))
        x = np.sin(np.deg2rad(f))*np.sin(np.deg2rad(q))
        return x, y, z
    except NameError:
        y = np.cos(np.deg2rad(q))
        x = np.sin(np.deg2rad(q))
        return x, y


im_path = np.zeros_like(im, dtype=int)
x, y = new_end_point(start)
path = [start]
loc = np.copy(start)
i = 0
while i < 10000:
    i += 1
    new_loc = loc + np.array([x, y])
    if im[tuple(new_loc.astype(int))] == False:
        x, y = new_end_point(loc)
        start = np.copy(loc)
    elif np.sqrt(np.sum((new_loc-start)**2)) > 150:
        x, y = new_end_point(loc)
        start = np.copy(loc)
    else:
        loc = new_loc
        path.append(loc)
        im_path[tuple(loc.astype(int))] = 1.0

# for xy in path:
    # im[tuple(xy.astype(int))] = False

plt.imshow(im_path + (~im)*0.5, origin='xy')
