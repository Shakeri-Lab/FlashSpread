"""
Simulation engines for FlashSpread.

This module provides the dual-engine architecture:
- MarkovianEngine: Sparse O(K) updates for memoryless processes (SIS, SIR)
- RenewalEngine: Dense O(N) updates for age-dependent processes (SEIR)

Recommended Usage:
    # For non-Markovian SEIR simulations (use CUDA Graph for best performance)
    from flashspread.engines import RenewalEngineCUDAGraph
    engine = RenewalEngineCUDAGraph(graph, model, steps_per_launch=50)

    # For Markovian SIS/SIR simulations
    from flashspread.engines import MarkovianEngine
    engine = MarkovianEngine(graph, model)

Performance Tips:
    - RenewalEngineCUDAGraph provides ~2.8x speedup over RenewalEngine
    - Use steps_per_launch=50-100 for optimal throughput
    - See docs/PERFORMANCE_ANALYSIS.md for detailed guidance
"""

from .markovian import MarkovianEngine
from .renewal import RenewalEngine, RenewalEngineCUDAGraph
from .renewal_tunable import (
    RenewalEngineTunable,
    RenewalEngineTunableCUDAGraph,
    estimate_flops_per_step,
    estimate_memory_bytes_per_step,
)


# Recommended default configurations
RENEWAL_DEFAULTS = {
    "epsilon": 0.03,        # Accuracy parameter
    "tau_max": 1.0,         # Maximum time step
    "steps_per_launch": 50, # CUDA Graph batch size (for CUDAGraph variant)
}

MARKOVIAN_DEFAULTS = {
    "max_prob": 0.1,   # Maximum transition probability per step
    "theta": 0.01,     # Target fraction of nodes transitioning
    "tau_max": 1.0,    # Maximum time step
}


def create_renewal_engine(
    graph,
    model,
    device: str = "cuda",
    use_cuda_graph: bool = True,
    **kwargs
):
    """
    Factory function to create the recommended Renewal engine.

    Args:
        graph: Network object with edge_index and csr attributes
        model: Non-Markovian compartmental model (e.g., SEIRModel)
        device: PyTorch device (default: "cuda")
        use_cuda_graph: Use CUDA Graph batching for 2.8x speedup (default: True)
        **kwargs: Override default parameters

    Returns:
        RenewalEngineCUDAGraph if use_cuda_graph=True, else RenewalEngine
    """
    params = {**RENEWAL_DEFAULTS, **kwargs}

    if use_cuda_graph:
        return RenewalEngineCUDAGraph(
            graph, model, device=device,
            epsilon=params["epsilon"],
            tau_max=params["tau_max"],
            steps_per_launch=params["steps_per_launch"],
        )
    else:
        return RenewalEngine(
            graph, model, device=device,
            epsilon=params["epsilon"],
            tau_max=params["tau_max"],
        )


def create_markovian_engine(
    graph,
    model,
    device: str = "cuda",
    **kwargs
):
    """
    Factory function to create the recommended Markovian engine.

    Args:
        graph: Network object with edge_index and csr attributes
        model: Markovian compartmental model (e.g., SISModel, SIRModel)
        device: PyTorch device (default: "cuda")
        **kwargs: Override default parameters

    Returns:
        MarkovianEngine with recommended defaults
    """
    params = {**MARKOVIAN_DEFAULTS, **kwargs}

    return MarkovianEngine(
        graph, model, device=device,
        max_prob=params["max_prob"],
        theta=params["theta"],
        tau_max=params["tau_max"],
    )


__all__ = [
    # Core engines
    "MarkovianEngine",
    "RenewalEngine",
    "RenewalEngineCUDAGraph",
    # Tunable variants (for benchmarking)
    "RenewalEngineTunable",
    "RenewalEngineTunableCUDAGraph",
    # Utility functions
    "estimate_flops_per_step",
    "estimate_memory_bytes_per_step",
    # Factory functions (recommended)
    "create_renewal_engine",
    "create_markovian_engine",
    # Default configurations
    "RENEWAL_DEFAULTS",
    "MARKOVIAN_DEFAULTS",
]
