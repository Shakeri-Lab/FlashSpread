"""
Simulation engines for FlashSpread.

This module provides the dual-engine architecture:
- MarkovianEngine: Sparse O(K) updates for memoryless processes
- RenewalEngine: Dense O(N) updates for age-dependent processes
"""

from .markovian import MarkovianEngine
from .renewal import RenewalEngine, RenewalEngineCUDAGraph

__all__ = [
    "MarkovianEngine",
    "RenewalEngine",
    "RenewalEngineCUDAGraph",
]
