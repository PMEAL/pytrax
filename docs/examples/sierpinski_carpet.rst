.. _sierpinski_carpet:


###############################################################################
 Example 3: The Sierpinksi Carpet
###############################################################################

This example will demonstrate the principle of calculating the tortuosity from a porous image with lower porosity.

.. contents:: Topics Covered in this Tutorial

**Learning Objectives**

#. Generate a fractal image with self-similarity at different length scales
#. Run the RandomWalk for a fractal image
#. Produce some visualization

===============================================================================
Instantiating the RandomWalk class
===============================================================================

Assuming that you are now familiar with how to import and instantiate the simulation objects we now define a function to produce the Sierpinski carpet:

.. code-block:: python

    >>> def tileandblank(image, n):
    >>>     if n > 0:
    >>>         n -= 1
    >>>         shape = np.asarray(np.shape(image))
    >>>         image = np.tile(image, (3, 3))
    >>>         image[shape[0]:2*shape[0], shape[1]:2*shape[1]] = 0
    >>>         image = tileandblank(image, n)
    >>>     return image
    >>> im = np.ones([1, 1], dtype=int)
    >>> im = tileandblank(im, 4)

===============================================================================
Running and ploting the walk
===============================================================================

We're now ready to instantiate and run the walk, this time lets test the power of the program and run it for 1 million walkers using 10 parallel processes (please make sure you have that many avaialble):

.. code-block:: python

    >>> rw = pt.RandomWalk(im)
    >>> rw.run(nt=2500, nw=1e6, same_start=False, stride=5, num_proc=10)
    >>> rw.plot_walk_2d()

The simulation should take no longer than a minute when running on a single process and should produce a plot like this:
	
.. image:: https://i.imgur.com/SnzeDNv.png
   :align: center

Like the open space example the pattern is radial because the image is isotropic. We can see the reflected domains with the large black squares at the center. This time the walkers have escaped the original domain and some have travelled entirely through the next set of neighboring reflected domains and reached a third relflection. This signifies that the domain has been probed effectively over time. The MSD plot is as follows:

.. code-block:: python

    >>> rw.plot_msd()

.. image:: https://imgur.com/6QPCXYq.png
   :align: center
   
The ``plot_msd`` function shows that mean square displacement and axial displacement are all the same and increase linearly with time. However, unlike the open space example the slope of the curve is less than one. This is because the walkers are impeded by the solid objects and it takes a longer time to go around them than in open space. The toruosity is calculated as the reciprocal of the MSD slope and is equal in both directions and very straight signifying that we have chosen an adequate number of walkers and steps.
