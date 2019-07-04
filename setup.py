import os
import sys
from distutils.util import convert_path

sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

main_ = {}
ver_path = convert_path('pytrax/__init__.py')
with open(ver_path) as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, main_)

setup(
    name='pytrax',
    description='A random walk for estimating the toruosity tensor of images',
    version=main_['__version__'],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics'],
    packages=['pytrax'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'tqdm'],
    author='Tom Tranter',
    author_email='t.g.tranter@gmail.com',
    url='https://pytrax.readthedocs.io/en/latest/',
    project_urls={
        'Documentation': 'https://pytrax.readthedocs.io/en/latest/',
        'Source': 'https://github.com/PMEAL/pytrax',
        'Tracker': 'https://github.com/PMEAL/pytrax/issues',
    },
)
