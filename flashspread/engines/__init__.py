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
from .renewal import (
    RenewalEngine,
    RenewalEngineCUDAGraph,
    RenewalEngineNonMarkov,
    RenewalEngineNonMarkovCUDAGraph,
)
from .renewal_fused import RenewalEngineFused, RenewalEngineFusedCUDAGraph
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
    nonmarkov_edges: bool = True,
    use_fused: bool = True,
    bf16_weights: bool = False,
    transmission_mode: str = "constant",
    **kwargs
):
    """
    Factory function to create the recommended Renewal engine.

    Default configuration: RenewalEngineFusedCUDAGraph (fastest).

    Args:
        graph: Network object with edge_index and csr attributes
        model: Non-Markovian compartmental model (e.g., SEIRModel)
        device: PyTorch device (default: "cuda")
        use_cuda_graph: Use CUDA Graph batching (default: True)
        nonmarkov_edges: Use infectivity-based edge kernel (default: True)
        use_fused: Use fused Triton kernel (default: True). Requires
                  nonmarkov_edges=True. Falls back to unfused if False.
        bf16_weights: Downcast edge weights to bfloat16 (default: False)
        transmission_mode: "constant" (default, Markovian-equivalent) or
                          "age_dependent" (source-node compromise with h_IR)
        **kwargs: Override default parameters

    Returns:
        Appropriate engine variant based on options.
    """
    params = {**RENEWAL_DEFAULTS, **kwargs}
    engine_kwargs = dict(
        device=device,
        epsilon=params["epsilon"],
        tau_max=params["tau_max"],
        bf16_weights=bf16_weights,
    )

    # Set transmission mode on model before engine creation
    if hasattr(model, 'transmission_mode'):
        model.transmission_mode = transmission_mode

    # Fused kernel requires the infectivity path (nonmarkov_edges)
    if use_fused and nonmarkov_edges:
        if use_cuda_graph:
            return RenewalEngineFusedCUDAGraph(
                graph, model,
                steps_per_launch=params["steps_per_launch"],
                **engine_kwargs,
            )
        else:
            return RenewalEngineFused(graph, model, **engine_kwargs)
    elif nonmarkov_edges:
        if use_cuda_graph:
            return RenewalEngineNonMarkovCUDAGraph(
                graph, model,
                steps_per_launch=params["steps_per_launch"],
                **engine_kwargs,
            )
        else:
            return RenewalEngineNonMarkov(graph, model, **engine_kwargs)
    else:
        if use_cuda_graph:
            return RenewalEngineCUDAGraph(
                graph, model,
                steps_per_launch=params["steps_per_launch"],
                **engine_kwargs,
            )
        else:
            return RenewalEngine(graph, model, **engine_kwargs)


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
    # Non-Markovian edge engines
    "RenewalEngineNonMarkov",
    "RenewalEngineNonMarkovCUDAGraph",
    # Fused Triton kernel engines
    "RenewalEngineFused",
    "RenewalEngineFusedCUDAGraph",
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
