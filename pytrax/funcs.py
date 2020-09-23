# %% Import packages and define helper functions
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt


def new_vector(N=1):
    # Generate random theta and phi for each walker
    angles = np.random.rand(2, N)*360
    q, f = np.vstack(angles)
    # Convert to axial components of a unit vector displacement
    z = np.cos(np.deg2rad(f))
    y = np.sin(np.deg2rad(f))*np.cos(np.deg2rad(q))
    x = np.sin(np.deg2rad(f))*np.sin(np.deg2rad(q))
    return x, y, z


def get_start_point(im, N=1):
    # Find all voxels in im that are not solid
    options = np.where(im)
    # From this list of options, choice one for each walker
    inds = np.random.randint(0, len(options[0]), N)
    # Convert the chosen list location into an x,y,z index
    points = np.array([options[i][inds] for i in range(im.ndim)])
    return points


# %% Generate image
im = ps.generators.overlapping_spheres(shape=[500, 500], radius=5, porosity=0.9)
bd = ps.tools.get_border(im.shape, thickness=3, mode='faces')
im[bd] = False

# %% Run walk
# Specify settings for walkers
n_walkers = 4
n_steps = 2000
mean_free_path = 25
# Initialize arrays to store results, one image and one complete table
im_path = np.zeros_like(im, dtype=int)
path = np.zeros([n_steps, n_walkers, im.ndim], dtype=int)
# Generate the starting conditions
start = get_start_point(im, n_walkers)
x, y, z = new_vector(n_walkers)
loc = np.copy(start)
i = 0
while i < n_steps:
    # Determine trial location of each walker
    new_loc = loc + np.array([x, y])
    # Check validiate of trial step for each walker
    check_1 = im[tuple(new_loc.astype(int))] == False
    check_2 = np.sqrt(np.sum((new_loc-start)**2, axis=0)) > mean_free_path
    # If either check found an invalid move, address it
    if np.any(check_1) or np.any(check_2):
        # Find the walker indices which have invalid moves
        inds = np.where((check_1 == True) + (check_2 == True))
        # Regenerate direction vectors for invalid walkers
        x[inds], y[inds], z[inds] = new_vector(len(inds[0]))
        # Update starting position for invalid walkers
        start[:, inds] = loc[:, inds].astype(int)
        # The while loop will re-run so this step is ignored
    else:  # If all walkers had a valid step, then execute the walk
        # Update the location of each walker with trial step
        loc = new_loc
        # Record new position of walker in path array
        path[i][:] = loc.astype(int).T
        # Write walker position into image
        # Only works for few walkers and limited step number
        im_path[tuple(loc.astype(int))] = i
        # Increment the step index
        i += 1
# Show the image of walkers
plt.imshow(im_path/im, origin='xy')
