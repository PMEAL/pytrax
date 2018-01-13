
.. image:: https://readthedocs.org/projects/pytrax/badge/?version=latest
###############################################################################
Overview of pytrax
###############################################################################

*pytrax* is an implementation of a random walk to calculate the tortuosity tensor of images.

===============================================================================
Example Usage
===============================================================================

The following code block illustrates how to use pytrax to perform a random walk simulation in open space, view the results and plot the mean square displacement to get the tortuosity:

.. code-block:: python

  >>> import pytrax as pt
  >>> import numpy as np
  >>> image = np.ones([3, 3])
  >>> rw = pt.RandomWalk(image)
  >>> rw.run(1000, 1000)
  >>> rw.plot_walk_2d()
  >>> rw.calc_msd()
  >>> rw.plot_msd()
