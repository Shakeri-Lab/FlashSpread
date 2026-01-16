"""
FlashSpread: GPU-Accelerated Markovian and Non-Markovian Spreading Processes

A unified framework for simulating stochastic spreading processes on complex networks,
featuring dual-engine architecture for optimal performance across different dynamical regimes.
"""

__version__ = "1.0.0"
__author__ = "Heman Shakeri"
__email__ = "hs9hd@virginia.edu"

from .core import GraphCSR, FlashNeighbor, FixedDegreeGraph, RandomGeometricGraph
from .engines import MarkovianEngine, RenewalEngine, RenewalEngineCUDAGraph
from .models import SISModel, SIRModel, SEIRModel

__all__ = [
    # Core
    "GraphCSR",
    "FlashNeighbor",
    "FixedDegreeGraph",
    "RandomGeometricGraph",
    # Engines
    "MarkovianEngine",
    "RenewalEngine",
    "RenewalEngineCUDAGraph",
    # Models
    "SISModel",
    "SIRModel",
    "SEIRModel",
]
