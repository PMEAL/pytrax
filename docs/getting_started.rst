.. _getting_started:

===============
Getting Started
===============

------------
Requirements
------------

**Software:** pytrax only relies on a few core python modules including Numpy and matplotlib, among others. These packages can be difficult to install from source, so it's highly recommended to download the Anaconda Python Distrubution install for your platform, which will install all of these packages for you (and many more!).  Once this is done, you can then run the installation of pytrax as described in the next section.

**Hardware:** Although there are no technical requirements, it must be noted that working with large images (>500**3) requires a substantial computer, with perhaps 16 or 32 GB of RAM.  You can work on small images using normal computers to develop work flows, then size up to a larger computer for application on larger images.

------------
Installation
------------

pytrax is available on the Python Package Index (PyPI) and can be installed with the usual ``pip`` command as follows:

.. code-block:: none

    pip install pytrax

When installing in this way, the source code is stored somewhere deep within the Python installation folder, so it's not convenient to play with or alter the code.  If you wish to customize the code, then it might be better to download the source code from github into a personal directory (e.g C:\\pytrax) then install as follows:


.. code-block:: none

    pip install -e C:\pytrax

The '-e' argument means that the package is 'editable' so any changes you make to the code will be available the next time that pytrax is imported.

-----------
Basic Usage
-----------

To use pytrax simply import it at the Python prompt:

.. code-block:: python

    >>> import pytrax as pt

At the moment the package is very lightweight and it is expected that users have their own images to analyze. For testing purposes we make use of another one of our packages called PoreSpy. As well as having lots of image analysis tools for porous media, PoreSpy also contains an image ``generators`` module to produce a sample image as follows:

.. code-block:: python

    >>> import porespy as ps
    >>> image = ps.generators.blobs(shape=[100, 100])

Running the random walk simulation to estimate the tortuosity tensor is then completed with a few extra commands:

.. code-block:: python

    >>> rw = pt.RandomWalk(image)
    >>> rw.run(nt=1000, nw=1000, same_start=False, stride=1, num_proc=None)

Here the RandomWalk class is instantiated with the image that we generated and run with some parameters: ``nt`` is the number of time steps, ``nw`` is the number of walkers, ``same_start`` sets the walkers to have the same starting position in the image and is ``False`` (by default), ``stride`` is the number of steps between successive saves for calculations and output and ``num_proc`` is the number of parallel processors to use (defaulting to half the number available).

----------------
Plotting Results
----------------

pytrax has some built in plotting functionality to plot the coordinates of the walkers and also the mean square displacement vs. time which can be viewed with the following commands:

.. code-block:: python

    >>> rw.plot_walk_2d(check_solid=True, data='t')
    >>> rw.plot_msd()

The first plotting function plots the image and the walker steps and is colored by time step, changing the ``data`` argument to be ``w`` changes the color to walker index. The ``check_solid`` argument checks that the solid voxels in the image are not walked upon which is useful when changes to the code are made as a quick sense check. The walkers are free to leave the original image providing that there is a porous pathway at the edges. When this happens they are treated to be travelling in a reflected domain and the plotting function also displays this. The second plotting function shows the mean and axial square displacement and applies linear regression to fit a straight line with intercept through zero. The gradient of the slope is inversely proportional to the tortuosity of the image in that direction. This follows the definition of tortuosity being the ratio of diffusivity in open space to diffusivity in the porous media.

-----------------
Exporting Results
-----------------

For 3D images the ``plot_walk_2d`` function can be used to view a slice of the walk and image, however, for better visualization it is recommended to use the export function and view the results in Paraview. A tutorial on how to do this is provided but the following function will export the image and walker data:

.. code-block:: python

    >>> rw.export_walk(image=None, path=None, sub='data', prefix='rw_', sample=1)

The ``image`` argument optionally lets you export the original image or the larger reflected image which are both stored on the rw object as ``rw.im`` and ``rw.im_big``, respectively. Leaving the argument as ``None`` will not export any image. ``path`` is the directory to save the data and when set to ``None`` will default to the current working directory, ``sub`` creates a subfolder under the path directory to save the data in and defaults to 'data', ``prefix`` gives all the data a prefix and defaults to 'rw_' and finally ``sample`` is a down-sampling factor which in addition to the stride function in the run command will only output walker coordinates for time steps that are multiples of this number.
