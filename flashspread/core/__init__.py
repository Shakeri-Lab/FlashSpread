"""
Core components for FlashSpread: graph structures, kernels, and network utilities.
"""

from __future__ import annotations

import importlib

_LAZY_EXPORTS = {
    "GraphCSR": ("flashspread.core.graph", "GraphCSR"),
    "FlashNeighbor": ("flashspread.core.flash_neighbor", "FlashNeighbor"),
    "FlashNeighborInfectivity": (
        "flashspread.core.flash_neighbor",
        "FlashNeighborInfectivity",
    ),
    "ensemble_influence_csr": (
        "flashspread.core.flash_ensemble",
        "ensemble_influence_csr",
    ),
    "ensemble_infectivity_csr": (
        "flashspread.core.flash_ensemble",
        "ensemble_infectivity_csr",
    ),
    "ensemble_seir_renewal_rates_csr": (
        "flashspread.core.flash_ensemble",
        "ensemble_seir_renewal_rates_csr",
    ),
    "pack_ensemble_infectious_mask": (
        "flashspread.core.flash_ensemble",
        "pack_ensemble_infectious_mask",
    ),
    "finalize_ensemble_renewal_tau": (
        "flashspread.core.flash_ensemble_step",
        "finalize_ensemble_renewal_tau",
    ),
    "transition_ensemble_seir": (
        "flashspread.core.flash_ensemble_step",
        "transition_ensemble_seir",
    ),
    "FixedDegreeGraph": ("flashspread.core.network", "FixedDegreeGraph"),
    "RandomGeometricGraph": ("flashspread.core.network", "RandomGeometricGraph"),
    "BarabasiAlbertGraph": ("flashspread.core.network", "BarabasiAlbertGraph"),
    "WattsStrogatzGraph": ("flashspread.core.network", "WattsStrogatzGraph"),
    "load_edges": ("flashspread.core.network", "load_edges"),
    "load_graph": ("flashspread.core.network", "load_graph"),
    "save_edges_txt": ("flashspread.core.network", "save_edges_txt"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'flashspread.core' has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "GraphCSR",
    "FlashNeighbor",
    "FlashNeighborInfectivity",
    "ensemble_influence_csr",
    "ensemble_infectivity_csr",
    "ensemble_seir_renewal_rates_csr",
    "pack_ensemble_infectious_mask",
    "finalize_ensemble_renewal_tau",
    "transition_ensemble_seir",
    "FixedDegreeGraph",
    "RandomGeometricGraph",
    "BarabasiAlbertGraph",
    "WattsStrogatzGraph",
    "load_edges",
    "load_graph",
    "save_edges_txt",
]
