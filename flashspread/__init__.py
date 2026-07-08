"""
FlashSpread: GPU-Accelerated Markovian and Non-Markovian Spreading Processes

A unified framework for simulating stochastic spreading processes on complex networks,
featuring dual-engine architecture for optimal performance across different dynamical regimes.
"""

__version__ = "1.0.0"
__author__ = "Heman Shakeri"
__email__ = "hs9hd@virginia.edu"

from .core import GraphCSR, FlashNeighbor, FlashNeighborInfectivity, FixedDegreeGraph, RandomGeometricGraph
from .engines import (
    MarkovianEngine,
    RenewalEngine,
    RenewalEngineCUDAGraph,
    RenewalEngineNonMarkov,
    RenewalEngineNonMarkovCUDAGraph,
)
from .models import SISModel, SIRModel, SEIRModel
from .utils import check_env, resolve_device, seed_everything

__all__ = [
    # Environment / utilities
    "check_env",
    "resolve_device",
    "seed_everything",
    # Core
    "GraphCSR",
    "FlashNeighbor",
    "FlashNeighborInfectivity",
    "FixedDegreeGraph",
    "RandomGeometricGraph",
    # Engines
    "MarkovianEngine",
    "RenewalEngine",
    "RenewalEngineCUDAGraph",
    "RenewalEngineNonMarkov",
    "RenewalEngineNonMarkovCUDAGraph",
    # Models
    "SISModel",
    "SIRModel",
    "SEIRModel",
]
