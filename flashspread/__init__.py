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

__version__ = "1.0.0"
__author__ = "Heman Shakeri"
__email__ = "hs9hd@virginia.edu"

# --- Blessed public surface -------------------------------------------------
from .simulator import Simulator
from .trajectory import Trajectory
from .graphs import barabasi_albert, geometric, regular_graph, watts_strogatz
from .models import SEIRModel, SIRModel, SISModel
from .utils import check_env, resolve_device, seed_everything

# --- Kept importable for back-compat / power users (not advertised) ---------
# The engine zoo and low-level structures remain reachable, e.g.
#   from flashspread import RenewalEngine
#   from flashspread.engines import RenewalEngineFusedCUDAGraph
from .core import (  # noqa: F401
    FixedDegreeGraph,
    FlashNeighbor,
    FlashNeighborInfectivity,
    GraphCSR,
    RandomGeometricGraph,
)
from .engines import (  # noqa: F401
    MarkovianEngine,
    RenewalEngine,
    RenewalEngineCUDAGraph,
    RenewalEngineNonMarkov,
    RenewalEngineNonMarkovCUDAGraph,
)

__all__ = [
    # Simulation
    "Simulator",
    "Trajectory",
    # Graphs
    "regular_graph",
    "barabasi_albert",
    "watts_strogatz",
    "geometric",
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
