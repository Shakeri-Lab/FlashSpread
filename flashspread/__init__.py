"""
FlashSpread: GPU-Accelerated Markovian and Non-Markovian Spreading Processes

A unified framework for simulating stochastic spreading processes on complex
networks, with a dual-engine architecture (sparse Markovian for SIS/SIR, fused
dense renewal for age-dependent SEIR).

Quick start::

    import flashspread as fs

    graph = fs.regular_graph(100_000, degree=8, seed=0)
    model = fs.SEIRModel(beta=0.3)
    sim   = fs.Simulator(graph, model, seed=0).seed_infection(100)
    traj  = sim.run(until=50.0, record_every=1.0)

    print(traj.peak_infected, traj.final_attack_rate)

``Simulator`` picks the right engine for the model. Power users can reach the
full engine zoo directly via ``flashspread.engines``.
"""

from __future__ import annotations

import importlib

__version__ = "1.0.0"
__author__ = "Heman Shakeri"
__email__ = "hs9hd@virginia.edu"

# --- Public surface ---------------------------------------------------------
# Resolve every public object on first access.  In particular, a bare package
# import is metadata-only and does not initialize Torch, NumPy, Triton, or the
# optional graph stack.  Resolved values are cached in this module so normal
# attribute access and ``from flashspread import Name`` retain their usual
# module semantics.
_LAZY_EXPORTS = {
    # Blessed simulation facade
    "Simulator": ("flashspread.simulator", "Simulator"),
    "EngineConfig": ("flashspread.config", "EngineConfig"),
    "Trajectory": ("flashspread.trajectory", "Trajectory"),
    # Blessed graph API
    "regular_graph": ("flashspread.graphs", "regular_graph"),
    "barabasi_albert": ("flashspread.graphs", "barabasi_albert"),
    "watts_strogatz": ("flashspread.graphs", "watts_strogatz"),
    "geometric": ("flashspread.graphs", "geometric"),
    "from_edges": ("flashspread.graphs", "from_edges"),
    "from_csr": ("flashspread.graphs", "from_csr"),
    "GraphCSR": ("flashspread.core.graph", "GraphCSR"),
    # Blessed models and utilities
    "SISModel": ("flashspread.models", "SISModel"),
    "SIRModel": ("flashspread.models", "SIRModel"),
    "SEIRModel": ("flashspread.models", "SEIRModel"),
    "check_env": ("flashspread.utils", "check_env"),
    "resolve_device": ("flashspread.utils", "resolve_device"),
    "seed_everything": ("flashspread.utils", "seed_everything"),
    # Back compatibility / power-user surface (deliberately not in __all__)
    "FixedDegreeGraph": ("flashspread.core.network", "FixedDegreeGraph"),
    "RandomGeometricGraph": ("flashspread.core.network", "RandomGeometricGraph"),
    "load_graph": ("flashspread.core.network", "load_graph"),
    "FlashNeighbor": ("flashspread.core.flash_neighbor", "FlashNeighbor"),
    "FlashNeighborInfectivity": (
        "flashspread.core.flash_neighbor",
        "FlashNeighborInfectivity",
    ),
    "MarkovianEngine": ("flashspread.engines.markovian", "MarkovianEngine"),
    "MarkovianEngineCUDAGraph": (
        "flashspread.engines.markovian",
        "MarkovianEngineCUDAGraph",
    ),
    "ReferenceEnsembleEngine": (
        "flashspread.engines.ensemble",
        "ReferenceEnsembleEngine",
    ),
    "EnsembleEngine": (
        "flashspread.engines.ensemble",
        "EnsembleEngine",
    ),
    "RenewalEngine": ("flashspread.engines.renewal", "RenewalEngine"),
    "RenewalEngineCUDAGraph": (
        "flashspread.engines.renewal",
        "RenewalEngineCUDAGraph",
    ),
    "RenewalEngineNonMarkov": (
        "flashspread.engines.renewal",
        "RenewalEngineNonMarkov",
    ),
    "RenewalEngineNonMarkovCUDAGraph": (
        "flashspread.engines.renewal",
        "RenewalEngineNonMarkovCUDAGraph",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'flashspread' has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    # Simulation
    "Simulator",
    "EngineConfig",
    "Trajectory",
    # Graphs
    "regular_graph",
    "barabasi_albert",
    "watts_strogatz",
    "geometric",
    "from_edges",
    "from_csr",
    "GraphCSR",
    "load_graph",
    # Models
    "SISModel",
    "SIRModel",
    "SEIRModel",
    # Environment / utilities
    "check_env",
    "resolve_device",
    "seed_everything",
    "__version__",
]
