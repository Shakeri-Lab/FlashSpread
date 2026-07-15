"""
Compartmental models and hazard functions for FlashSpread.

This module provides standard epidemic models (SIS, SIR, SEIR) and
hazard functions for age-dependent (non-Markovian) dynamics.
"""

from .compartmental import SISModel, SIRModel, SEIRModel
from .hazards import (
    lognormal_hazard,
    lognormal_hazard_stable,
    weibull_hazard,
    gamma_hazard,
    build_hazard_from_params,
)

__all__ = [
    "SISModel",
    "SIRModel",
    "SEIRModel",
    "lognormal_hazard",
    "lognormal_hazard_stable",
    "weibull_hazard",
    "gamma_hazard",
    "build_hazard_from_params",
]
