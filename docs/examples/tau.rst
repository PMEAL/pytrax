.. _tau:


###############################################################################
 Example 2: The Tortuosity of Tau
###############################################################################

This example shows the code working on a pseudo-porous media and explains some of the things to be careful of.

.. contents:: Topics Covered in this Tutorial

**Learning Objectives**

#. Practice the code on a real image
#. Explain the significance of the number of walkers and steps.
#. Visualize the domain reflection

===============================================================================
Obtaining the image
===============================================================================

As with the previous example, the first thing to do is to import the packages that we are going to use including some packages for importing the image we will use:

.. code-block:: python

    >>> import pytrax as ps
    >>> import numpy as np
    >>> import urllib.request as ur
    >>> from io import BytesIO
    >>> import matplotlib.pyplot as plt
    >>> from PIL import Image

Next we are going to grab an image using its URL and make some modifications to prepare it for pytrax:

.. code-block:: python

    >>> url = 'https://i.imgur.com/nrEJRDf.png'
    >>> file = BytesIO(ur.urlopen(url).read())
    >>> im = np.asarray(Image.open(file))[:, :, 3] == 0
    >>> im = im.astype(int)
    >>> im = np.pad(im, pad_width=50, mode='constant', constant_values=1)

The image is a .png and has 4 layers: r, g, b and alpha. We use the alpha layer which sets the contrast and make the image binary by setting the zero-valued pixels to ``True`` then converting to a type ``int``. Finally we pad the image with additional pore space which is designated as 1.

.. code-block:: python

	>>> rw = pt.RandomWalk(image=ima, seed=False

===============================================================================
Running and ploting the walk
===============================================================================

We're now ready to run the walk setting the number of walkers to be 1,000 and the number of steps to be 20,000:

.. code-block:: python

	>>> rw.run(nt=20000, nw=1000)
	>>> rw.plot_walk_2d()

The following 2D plot is produced:

.. image:: https://i.imgur.com/mSQZVku.png
   :align: center

This time it is clear how the domain reflections happen with the original in the center surrounded by reflections of the Tau symbol along the two principal axes. Even though the number of steps taken by the walkers seems quite large, the number of pixels in the image is also large and the length of the average walk is no larger than the original domain. Let's plot the MSD to get a little more information:

.. code-block:: python

	>>> rw.plot_msd()

.. image:: https://i.imgur.com/B3L8dqc.png
   :align: center
   
The ``plot_msd`` function shows that mean square displacement and axial displacement are increasing over time but not in such a linear fashion as the previous example. There also appears to be some anisotropy as the 0th axis has a different tortuosity to the 1st axis. The MSD plot shows that we should run the simulation again for longer as the length of the walk is about the same length as the average pore size. It really needs to be at least 4 or 5 times longer so that the effect of hitting pore walls is evenly distributed among the walkers over time. The number of walkers should also be increased as we can see that the MSD is not particularly smooth and creating a larger ensemble average will give better results. Try running the same simulation again but increasing both ``nt`` and ``nw`` by a factor of 4.
