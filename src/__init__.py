"""
PVFit - primary vertex reconstruction and performance analysis.

The package is organised in two stages:

- :mod:`src.fit`      ROOT tracks -> seeding -> vertex fit -> HDF5
- :mod:`src.analysis` HDF5 -> efficiency / fake / merge / split / resolution / pulls

:mod:`src.optimize` drives a multi-objective scan over the algorithm parameters.
"""

__version__ = "0.1.0"
