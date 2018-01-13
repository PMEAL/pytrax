# pytrax
Random walk to calculate the tortuosity tensor of images.

To use, simply do::

  >>> import pytrax as pt
  >>> import numpy as np
  >>> image = np.ones([3, 3])
  >>> rw = pt.RandomWalk(image)
  >>> rw.run(1000, 1000)
  >>> rw.plot_walk_2d()
