# %% Import packages and define helper functions
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm


def new_vector(im, N=1):
    # Generate random theta and phi for each walker
    q, f = np.vstack(np.random.rand(2, N)*2*np.pi)  # in Radians
    if im.ndim == 2:
        f = np.zeros([N, ])
    # Convert to axial components of a unit vector displacement
    z = np.sin(f)
    y = np.cos(f)*np.sin(q)
    x = np.cos(f)*np.cos(q)
    return x, y, z


def get_start_point(im, N=1):
    # Find all voxels in im that are not solid
    options = np.where(im)
    # From this list of options, choice one for each walker
    inds = np.random.randint(0, len(options[0]), N)
    # Convert the chosen list location into an x,y,z index
    points = np.array([options[i][inds] for i in range(im.ndim)])
    return points


def wrap_indices(loc, shape):
    # Periodic BC using modulus of loc
    # no need to put a check since 250%400 is 250 only
    temp = np.copy(loc)
    temp[0] = np.around(loc[0]) % (shape[0])
    temp[1] = np.around(loc[1]) % (shape[1])
    if im.ndim == 3:
        temp[2] = np.around(loc[2]) % (shape[2])
    return tuple(temp.astype(int))


def calculate_msd(path):
    disp = path[:, :, :] - path[0, :, :]
    asd = disp**2
    sq_disp = np.sum(disp**2, axis=2)
    msd = np.mean(sq_disp, axis=1)
    axial_msd = np.mean(asd, axis=1)
    return msd, axial_msd


# %% Generate image
im = ps.generators.overlapping_spheres(shape=[200, 400], radius=5, porosity=0.99)

# %% Specify settings for walkers
n_walkers = 1000
n_steps = 5000
mean_free_path = 10

# %% Run walk
# Initialize arrays to store results, one image and one complete table
im_path = np.zeros_like(im, dtype=int)
path = np.zeros([n_steps, n_walkers, im.ndim])
# Generate the starting conditions
start = get_start_point(im, n_walkers)
x, y, z = new_vector(im, n_walkers)
loc = np.copy(start)
i = 0
with tqdm(range(n_steps)) as pbar:
    while i < n_steps:
        # Determine trial location of each walker
        if im.ndim == 3:
            new_loc = loc + np.array([x,y,z])
        else:
            new_loc = loc + np.array([x, y])
        # Check trial step for each walker
        temp = wrap_indices(new_loc, im.shape)
        check_1 = im[temp] == False
        check_2 = np.sqrt(np.sum((new_loc-start)**2, axis=0)) > mean_free_path
        # If either check found an invalid move, address it
        if np.any(check_1) or np.any(check_2):
            # Find the walker indices which have invalid moves
            inds = np.where((check_1 == True) + (check_2 == True))
            # Regenerate direction vectors for invalid walkers
            x[inds], y[inds], z[inds] = new_vector(im, len(inds[0]))
            # Update starting position for invalid walkers to current position
            start[:, inds] = np.around(loc[:, inds]).astype(int)
            # The while loop will re-run so this step is ignored
        else:  # If all walkers had a valid step, then execute the walk
            # Update the location of each walker with trial step
            loc = new_loc
            # Record new position of walker in path array
            path[i][:] = loc.T
            # Write walker position into image
            # Only works for few walkers and limited step number
            im_path[wrap_indices(loc, im.shape)] += 1
            # Increment the step index
            i += 1
            pbar.update()

# %%
# The following is the distance each walker travels in each step...should
# be 1.0 everywhere. Will be good for a unit test.
# d = (path[1:, :, 0] - path[:-1, :, 0])**2 + (path[1:, :, 1] - path[:-1, :, 1])**2
d = (path[1:, :, 0] - path[0, :, 0])**2 + (path[1:, :, 1] - path[0, :, 1])**2
s = np.arange(1, n_steps, 10)
fig = plt.plot(s, d[::10, :].mean(axis=1), '.')
fig = plt.plot(s, s)


# %%
# Show the image of walkers
plt.figure()
temp = np.log10(im_path+1)/im
temp = np.tile(temp, [3, 3])
plt.imshow(temp, origin='xy', cmap=plt.cm.twilight_r)
plt.axis('tight')
# imageio.volsave('Random_walk_3d.tif', (np.log10(im_path+0.1)/im).astype(int))

# calculate_msd(path)
#print(calculate_msd(path))