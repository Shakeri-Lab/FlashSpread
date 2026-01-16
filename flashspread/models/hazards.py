"""
Hazard functions for non-Markovian epidemic simulation.

This module provides numerically stable hazard function implementations
for common dwell-time distributions used in epidemiology.

The hazard function h(t) = f(t)/S(t) represents the instantaneous
probability of transitioning given survival to time t, where f is the
density and S is the survival function.

Key function: lognormal_hazard_stable uses the scaled complementary
error function (erfcx) to avoid catastrophic cancellation for large ages.
"""

import torch
import math
from typing import Callable


def lognormal_hazard(
    age: torch.Tensor,
    mean: float,
    median: float,
) -> torch.Tensor:
    """
    Compute log-normal hazard rate (basic version).

    For a log-normal distribution with given mean and median:
        mu = ln(median)
        sigma = sqrt(2 * ln(mean / median))

    The hazard is h(t) = f(t) / S(t) where:
        f(t) = (1 / (t * sigma * sqrt(2*pi))) * exp(-(ln(t) - mu)^2 / (2*sigma^2))
        S(t) = 1 - Phi((ln(t) - mu) / sigma)

    Warning: This basic version can suffer from numerical issues for large ages.
    Use lognormal_hazard_stable for production code.

    Args:
        age: Tensor of holding times (must be > 0).
        mean: Mean of the log-normal distribution.
        median: Median of the log-normal distribution.

    Returns:
        Tensor of hazard rates.
    """
    # Convert mean/median to mu/sigma
    mu = math.log(median)
    sigma = math.sqrt(2.0 * math.log(mean / median))

    # Clamp age to avoid log(0)
    t = torch.clamp(age, min=1e-10)

    # Standardized variable
    z = (torch.log(t) - mu) / sigma

    # PDF: f(t) = phi(z) / (t * sigma)
    phi_z = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    pdf = phi_z / (t * sigma)

    # Survival: S(t) = Phi^c(z) = 1 - Phi(z)
    # Using erfc for better numerical stability
    survival = 0.5 * torch.erfc(z / math.sqrt(2))

    # Hazard: h(t) = f(t) / S(t)
    hazard = pdf / (survival + 1e-10)

    return hazard


def lognormal_hazard_stable(
    age: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Numerically stable log-normal hazard using erfcx.

    This formulation avoids catastrophic cancellation for large ages
    by using the scaled complementary error function:
        erfcx(z) = exp(z^2) * erfc(z)

    The stable formula is:
        h(t) = sqrt(2/pi) / (t * sigma * erfcx(z))
    where z = (ln(t) - mu) / (sigma * sqrt(2))

    This remains well-conditioned for all t > 0.

    Args:
        age: Tensor of holding times.
        mu: Log-normal location parameter (= ln(median)).
        sigma: Log-normal scale parameter.

    Returns:
        Tensor of hazard rates.
    """
    # Clamp age to avoid log(0)
    t = torch.clamp(age, min=1e-10)

    # Standardized variable for erfcx
    z = (torch.log(t) - mu) / (sigma * math.sqrt(2))

    # Numerically stable hazard using erfcx
    # h(t) = sqrt(2/pi) / (t * sigma * erfcx(z))
    erfcx_z = torch.special.erfcx(z)

    # Avoid division by zero for very small erfcx values
    erfcx_z = torch.clamp(erfcx_z, min=1e-30)

    hazard = math.sqrt(2.0 / math.pi) / (t * sigma * erfcx_z)

    return hazard


def weibull_hazard(
    age: torch.Tensor,
    shape: float,
    scale: float,
) -> torch.Tensor:
    """
    Compute Weibull hazard rate.

    For Weibull distribution with shape k and scale lambda:
        h(t) = (k / lambda) * (t / lambda)^(k-1)

    The Weibull distribution is flexible:
    - k < 1: decreasing hazard (infant mortality)
    - k = 1: constant hazard (exponential = Markovian)
    - k > 1: increasing hazard (aging)

    Args:
        age: Tensor of holding times.
        shape: Weibull shape parameter (k).
        scale: Weibull scale parameter (lambda).

    Returns:
        Tensor of hazard rates.
    """
    t = torch.clamp(age, min=1e-10)
    hazard = (shape / scale) * torch.pow(t / scale, shape - 1)
    return hazard


def gamma_hazard(
    age: torch.Tensor,
    shape: float,
    rate: float,
) -> torch.Tensor:
    """
    Compute Gamma distribution hazard rate.

    The Gamma hazard does not have a closed form but can be computed as:
        h(t) = f(t) / S(t)

    For shape parameter k and rate parameter beta:
        f(t) = beta^k * t^(k-1) * exp(-beta*t) / Gamma(k)

    Note: This uses numerical approximation and may be slow for large tensors.

    Args:
        age: Tensor of holding times.
        shape: Gamma shape parameter (k).
        rate: Gamma rate parameter (beta).

    Returns:
        Tensor of hazard rates.
    """
    t = torch.clamp(age, min=1e-10)

    # Use the identity: f(t)/S(t) where S uses regularized incomplete gamma
    # This is computationally expensive, so prefer Weibull or lognormal
    log_pdf = (
        shape * math.log(rate)
        + (shape - 1) * torch.log(t)
        - rate * t
        - math.lgamma(shape)
    )
    pdf = torch.exp(log_pdf)

    # Survival function using regularized upper incomplete gamma
    # S(t) = Q(shape, rate*t) = gammaincc(shape, rate*t)
    # PyTorch has igammac for this
    survival = torch.igammac(torch.tensor(shape), rate * t)
    survival = torch.clamp(survival, min=1e-10)

    hazard = pdf / survival
    return hazard


def build_hazard_from_params(
    hazard_type: str,
    device: torch.device = None,
    **params,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Factory function to create hazard functions from parameters.

    This is useful for building models programmatically from configuration.

    Args:
        hazard_type: Type of hazard ("lognormal", "weibull", "network").
        device: Device for precomputed parameters.
        **params: Distribution parameters.

    Returns:
        Callable hazard function: (age, pressure) -> rate.

    Example:
        >>> hazard_fn = build_hazard_from_params(
        ...     "lognormal", device="cuda", mean=5.0, median=4.0
        ... )
        >>> rates = hazard_fn(age_tensor, pressure_tensor)
    """
    if hazard_type == "lognormal":
        mean = float(params["mean"])
        median = float(params["median"])

        if device is not None:
            device = torch.device(device)
            mu = torch.tensor(math.log(median), device=device, dtype=torch.float32)
            sig = torch.tensor(
                math.sqrt(2.0 * math.log(mean / median)),
                device=device,
                dtype=torch.float32,
            )

            def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
                return lognormal_hazard_stable(age, mu, sig)

        else:

            def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
                return lognormal_hazard(age, mean, median)

        return _hazard

    elif hazard_type == "weibull":
        shape = float(params["shape"])
        scale = float(params["scale"])

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return weibull_hazard(age, shape, scale)

        return _hazard

    elif hazard_type == "network":
        # Pure network-driven rate (Markovian)
        beta = float(params["beta"])

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return pressure * beta

        return _hazard

    elif hazard_type == "constant":
        # Constant rate (exponential = Markovian)
        rate = float(params["rate"])

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return torch.full_like(age, rate)

        return _hazard

    else:
        raise ValueError(f"Unknown hazard_type: {hazard_type}")
