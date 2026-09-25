"""
PVFit - primary vertex reconstruction and performance analysis.

The package is organised in two stages:

- :mod:`pvfit.fit`      ROOT tracks -> seeding -> vertex fit -> HDF5
- :mod:`pvfit.analysis` HDF5 -> efficiency / fake / merge / split / resolution / pulls

:mod:`pvfit.optimize` drives a multi-objective scan over the algorithm parameters.
"""

__version__ = "0.1.0"
