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

from ..utils import validate_fp32_control


def lognormal_hazard(
    age: torch.Tensor,
    mean: float,
    median: float,
) -> torch.Tensor:
    """
    Compute log-normal hazard from mean/median parameters.

    For a log-normal distribution with given mean and median:
        mu = ln(median)
        sigma = sqrt(2 * ln(mean / median))

    The hazard is h(t) = f(t) / S(t) where:
        f(t) = (1 / (t * sigma * sqrt(2*pi))) * exp(-(ln(t) - mu)^2 / (2*sigma^2))
        S(t) = 1 - Phi((ln(t) - mu) / sigma)

    This compatibility signature delegates to the stable ``erfcx``
    implementation; it no longer maintains a second tail-unstable formula.

    Args:
        age: Tensor of holding times (must be > 0).
        mean: Mean of the log-normal distribution.
        median: Median of the log-normal distribution.

    Returns:
        Tensor of hazard rates.
    """
    mean = validate_fp32_control("mean", mean, positive=True)
    median = validate_fp32_control("median", median, positive=True)
    if median <= 0.0 or mean <= median:
        raise ValueError("Log-normal parameters require mean > median > 0")

    # Convert mean/median and delegate to the one production implementation.
    mu = math.log(median)
    sigma = math.sqrt(2.0 * math.log(mean / median))
    return lognormal_hazard_stable(age, mu, sigma)


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

    This remains well-conditioned for all t > 0. It is the unchecked tensor
    primitive used inside hot engine paths: ``mu`` must be finite and
    ``sigma`` must be finite and strictly positive. Public callers with
    mean/median parameters should prefer :func:`lognormal_hazard`, which
    validates and converts those parameters.

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


def erfcx_rational_approx(z: torch.Tensor) -> torch.Tensor:
    """
    Rational approximation of erfcx(z) = exp(z^2) * erfc(z).

    Reference implementation in PyTorch for validating the Triton version.
    Uses the direct identity for moderate inputs and a short asymptotic series
    for large positive inputs. It is a lightweight standalone test oracle, not
    the polynomial/range decomposition used by the current Triton kernel and
    not a high-accuracy special-function replacement.

    Args:
        z: Input tensor.

    Returns:
        Tensor of erfcx values.
    """
    az = torch.abs(z)

    # Region 1: use the direct identity for moderate z:
    # erfcx(z) = exp(z^2) * erfc(z)
    # For |z| <= 4, erfc is well-conditioned, so this is safe in fp32
    small = torch.exp(az * az) * torch.erfc(az)

    # Region 2: |z| > 4 — asymptotic expansion
    # erfcx(z) ~ 1/(z*sqrt(pi)) * (1 - 1/(2z^2) + 3/(4z^4) - ...)
    inv_z = 1.0 / az
    inv_z2 = inv_z * inv_z
    rsqrt_pi = 0.5641895835477563  # 1/sqrt(pi)
    large = rsqrt_pi * inv_z * (
        1.0 - 0.5 * inv_z2 + 0.75 * inv_z2 * inv_z2
    )

    result_pos = torch.where(az <= 4.0, small, large)

    # For z < 0: erfcx(z) = 2*exp(z^2) - erfcx(-z)
    result = torch.where(
        z >= 0, result_pos, 2.0 * torch.exp(z * z) - result_pos
    )

    return torch.clamp(result, min=1e-30)


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
    shape = validate_fp32_control("shape", shape, positive=True)
    scale = validate_fp32_control("scale", scale, positive=True)
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
    shape = validate_fp32_control("shape", shape, positive=True)
    rate = validate_fp32_control("rate", rate, positive=True)
    result_dtype = age.dtype if age.is_floating_point() else torch.get_default_dtype()
    t = torch.clamp(age.to(torch.float64), min=1e-10)

    # Use the identity: f(t)/S(t) where S uses regularized incomplete gamma
    # This is computationally expensive, so prefer Weibull or lognormal
    log_pdf = (
        shape * math.log(rate)
        + (shape - 1) * torch.log(t)
        - rate * t
        - math.lgamma(shape)
    )
    # Survival function using regularized upper incomplete gamma
    # S(t) = Q(shape, rate*t) = gammaincc(shape, rate*t)
    # PyTorch has igammac for this
    shape_t = torch.tensor(shape, device=age.device, dtype=torch.float64)
    survival = torch.igammac(shape_t, rate * t)

    # Evaluate the ratio in log space. For extreme tails where even fp64 Q
    # underflows, evaluate the upper incomplete gamma with Lentz's continued
    # fraction. In that representation h(t) simplifies to rate / (x * cf),
    # avoiding both the density and survival underflows.
    safe_survival = torch.where(survival > 0.0, survival, 1.0)
    hazard = torch.exp(log_pdf - torch.log(safe_survival))
    x = rate * t
    x_cf = torch.maximum(x, torch.full_like(x, shape + 1.0))
    tiny = 1e-300
    b = x_cf + 1.0 - shape
    c = torch.full_like(x_cf, 1.0 / tiny)
    d = 1.0 / b
    fraction = d
    tiny_t = torch.full_like(x_cf, tiny)
    for order in range(1, 33):
        coefficient = -float(order) * (float(order) - shape)
        b = b + 2.0
        d = coefficient * d + b
        d = torch.where(
            d.abs() < tiny,
            torch.copysign(tiny_t, d),
            d,
        )
        c = b + coefficient / c
        c = torch.where(
            c.abs() < tiny,
            torch.copysign(tiny_t, c),
            c,
        )
        d = 1.0 / d
        fraction = fraction * d * c
    tail_hazard = rate / (x_cf * fraction)
    hazard = torch.where(survival > 0.0, hazard, tail_hazard)
    return hazard.to(result_dtype)


def build_hazard_from_params(
    hazard_type: str,
    device: torch.device = None,
    **params,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Factory function to create hazard functions from parameters.

    This is useful for building models programmatically from configuration.

    Args:
        hazard_type: ``lognormal``, ``weibull``, ``gamma``, ``constant``, or
            ``network``.
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
    if not isinstance(hazard_type, str):
        raise TypeError("hazard_type must be a string")
    if hazard_type == "lognormal":
        mean = validate_fp32_control("mean", params["mean"], positive=True)
        median = validate_fp32_control(
            "median", params["median"], positive=True
        )
        if median <= 0.0 or mean <= median:
            raise ValueError("Log-normal parameters require mean > median > 0")
        if device is not None:
            torch.device(device)  # validate the optional compatibility hint
        mu = math.log(median)
        sig = math.sqrt(2.0 * math.log(mean / median))
        validate_fp32_control("log(median)", mu)
        validate_fp32_control("sigma", sig, positive=True)

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return lognormal_hazard_stable(age, mu, sig)

        return _hazard

    elif hazard_type == "weibull":
        shape = validate_fp32_control("shape", params["shape"], positive=True)
        scale = validate_fp32_control("scale", params["scale"], positive=True)

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return weibull_hazard(age, shape, scale)

        return _hazard

    elif hazard_type == "gamma":
        shape = validate_fp32_control("shape", params["shape"], positive=True)
        rate = validate_fp32_control("rate", params["rate"], positive=True)

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return gamma_hazard(age, shape, rate)

        return _hazard

    elif hazard_type == "network":
        # Pure network-driven rate (Markovian)
        beta = validate_fp32_control(
            "beta", params["beta"], nonnegative=True
        )

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return pressure * beta

        return _hazard

    elif hazard_type == "constant":
        # Constant rate (exponential = Markovian)
        rate = validate_fp32_control(
            "rate", params["rate"], nonnegative=True
        )

        def _hazard(age: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
            return torch.full_like(age, rate)

        return _hazard

    else:
        raise ValueError(f"Unknown hazard_type: {hazard_type}")
