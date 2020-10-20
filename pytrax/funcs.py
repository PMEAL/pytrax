# %% Import packages and define helper functions
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm


def new_vector(im, N=1, L=1):
    # Generate random theta and phi for each walker
    q, f = np.vstack(np.random.rand(2, N)*2*np.pi)  # in Radians
    if im.ndim == 2:
        f = np.zeros([N, ])
    # Convert to axial components of a unit vector displacement
    z = np.sin(f)
    y = np.cos(f)*np.sin(q)
    x = np.cos(f)*np.cos(q)
    return np.array((x, y, z))*L


def get_start_point(im, N=1):
    # Find all voxels in im that are not solid
    options = np.where(im)
    # From this list of options, choice one for each walker
    inds = np.random.randint(0, len(options[0]), N)
    # Convert the chosen list location into an x,y,z index
    points = np.array([options[i][inds] for i in range(im.ndim)])
    return points.astype(float)


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
im = ps.generators.overlapping_spheres(shape=[400, 400], radius=10, porosity=0.65)
im = ps.filters.fill_blind_pores(im)
bd = ps.tools.get_border(shape=im.shape, mode='faces')
im = ps.filters.trim_nonpercolating_paths(im, inlets=bd, outlets=bd)
# im = np.ones_like(im)

# %% Specify settings for walkers
n_walkers = 1000
n_steps = 5000
mu = 1.73e-5  # Pa.s
p = 101325  # Pa
T = 298  # K
R = 8.314  # J/mol.K
MW = 0.0291  # kg/mol
L = mu/p * np.sqrt(np.pi*R*T/(2*MW)) * 1e9  # nm
res = 600   # nm/voxel
mfp = L/res  # voxel
N_Av = 6.022e23  # Avagadro's constant
k_B = 1.3806e-23  # Boltzmann's constant
v_rms = np.sqrt(3 * k_B * T / (MW / N_Av)) * 1e9  # nm/s

# %% Run walk
# Initialize arrays to store results, one image and one complete table
im_path = np.zeros_like(im, dtype=int)
path = np.zeros([n_steps, n_walkers, im.ndim])
# Generate the starting conditions
start = get_start_point(im, n_walkers)
x, y, z = new_vector(im, N=n_walkers, L=min(1, mfp))
loc = np.copy(start)
i = 0
with tqdm(range(n_steps)) as pbar:
    while i < n_steps:
        # Determine trial location of each walker
        new_loc = loc + np.array([x, y])
        # Check trial step for each walker
        temp = wrap_indices(new_loc, im.shape)
        check_1 = im[temp] == False
        check_2 = np.sqrt(np.sum((new_loc-start)**2, axis=0)) > mfp
        # If either check found an invalid move, address it
        if np.any(check_1) or np.any(check_2):
            # Find the walker indices which have invalid moves
            inds = np.where((check_1 == True) + (check_2 == True))
            # Regenerate direction vectors for invalid walkers
            x[inds], y[inds], z[inds] = new_vector(im, N=len(inds[0]), L=min(1, mfp))
            # Update starting position for invalid walkers to current position
            start[:, inds] = loc[:, inds]
            # The while loop will re-run so this step is ignored
        else:  # If all walkers had a valid step, then execute the walk
            # Update the location of each walker with trial step
            loc = new_loc
            # Record new position of walker in path array
            path[i][:] = loc.T
            # Write walker position into image
            # Works best for few walkers and limited step number
            im_path[wrap_indices(loc[:, :], im.shape)] = i
            # Increment the step index
            i += 1
            pbar.update()

# %%
# The following is the distance each walker travels in each step...should
# be 1.0 everywhere. Will be good for a unit test.
# d = (path[1:, :, 0] - path[:-1, :, 0])**2 + (path[1:, :, 1] - path[:-1, :, 1])**2
d = np.sqrt((path[1:, :, 0] - path[0, :, 0])**2 +
            (path[1:, :, 1] - path[0, :, 1])**2)
d = d * res * 1e-9  # m
f = min(1, mfp)
s = np.sqrt(np.linspace(0, n_steps*f*res/v_rms, n_steps-1))
fig = plt.plot(s[::10], d[::10, :].mean(axis=1), '.')
plt.xlabel('sqrt(t) [s]')
plt.ylabel('Average Displacement [m]')

D = (d[-1, :]**2).mean() / (6*n_steps*f*res/v_rms)
print(D)


# %%
# Show the image of walkers
if 1:
    plt.figure()
    temp = np.log10(im_path+1)/im
    temp = np.tile(temp, [3, 3])
    plt.imshow(temp, origin='xy', cmap=plt.cm.viridis, interpolation='none')
    plt.axis('tight')
# imageio.volsave('Random_walk_3d.tif', (np.log10(im_path+0.1)/im).astype(int))