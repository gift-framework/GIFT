"""TCS profile functions for Joyce/Kovalev-style G2 constructions.

This module implements smooth 1D profile functions that encode the
geometry of Twisted Connected Sum (TCS) G2 manifolds:

- left_plateau: ~1 on CY3_L side, ~0 on CY3_R side
- right_plateau: ~0 on CY3_L side, ~1 on CY3_R side
- neck_bump: Localized around the neck/gluing region

All functions are designed to be:
1. C-infinity smooth (infinitely differentiable)
2. Vectorized PyTorch operations
3. Parameterized by neck position and characteristic width
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "smooth_step",
    "left_plateau",
    "right_plateau",
    "neck_bump",
    "neck_bump_normalized",
    "derivative_smooth_step",
    "TCSProfiles",
]


def smooth_step(
    x: torch.Tensor,
    x0: float = 0.0,
    width: float = 0.1,
    steepness: float = 1.0,
) -> torch.Tensor:
    """C-infinity smooth sigmoid around x0 with characteristic width.

    Uses the logistic function: sigma(t) = 1 / (1 + exp(-t))
    Centered at x0 with transition width controlled by `width`.

    Parameters
    ----------
    x : torch.Tensor
        Input coordinates, shape [N] or [N, 1, ...] for broadcasting.
    x0 : float
        Center of the transition region.
    width : float
        Characteristic width of transition (larger = smoother).
    steepness : float
        Additional steepness multiplier (default 1.0).

    Returns
    -------
    torch.Tensor
        Values in [0, 1], ~0 for x << x0, ~1 for x >> x0.
    """
    # Avoid division by zero
    w = max(width, 1e-8)
    t = steepness * (x - x0) / w
    return torch.sigmoid(t)


def left_plateau(
    lambda_coord: torch.Tensor,
    lambda_L: float = -1.0,
    lambda_neck: float = 0.0,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Smooth function ~1 on the left (CY3_L), ~0 on the right.

    Profile for modes that are "born" in the left CY3 region and
    decay toward the right CY3 region.

    Parameters
    ----------
    lambda_coord : torch.Tensor
        The neck coordinate (typically x[..., 0]).
    lambda_L : float
        Position of the left CY3 region (where function ~ 1).
    lambda_neck : float
        Position of the neck/transition region.
    sigma : float
        Width of the transition region.

    Returns
    -------
    torch.Tensor
        Profile values in [0, 1].
    """
    # Transition from 1 to 0 as lambda goes from lambda_L to lambda_neck and beyond
    # Use 1 - smooth_step to get decreasing function
    return 1.0 - smooth_step(lambda_coord, x0=lambda_neck, width=sigma)


def right_plateau(
    lambda_coord: torch.Tensor,
    lambda_R: float = 1.0,
    lambda_neck: float = 0.0,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Smooth function ~1 on the right (CY3_R), ~0 on the left.

    Profile for modes that are "born" in the right CY3 region and
    decay toward the left CY3 region.

    Parameters
    ----------
    lambda_coord : torch.Tensor
        The neck coordinate (typically x[..., 0]).
    lambda_R : float
        Position of the right CY3 region (where function ~ 1).
    lambda_neck : float
        Position of the neck/transition region.
    sigma : float
        Width of the transition region.

    Returns
    -------
    torch.Tensor
        Profile values in [0, 1].
    """
    # Transition from 0 to 1 as lambda goes from left to right
    return smooth_step(lambda_coord, x0=lambda_neck, width=sigma)


def neck_bump(
    lambda_coord: torch.Tensor,
    lambda_neck: float = 0.0,
    sigma: float = 0.1,
    peak_height: float = 1.0,
) -> torch.Tensor:
    """Localized bump around the neck, vanishing at both ends.

    This profile represents modes that are concentrated in the
    gluing region of the TCS construction.

    Parameters
    ----------
    lambda_coord : torch.Tensor
        The neck coordinate (typically x[..., 0]).
    lambda_neck : float
        Center position of the neck region.
    sigma : float
        Width of the bump.
    peak_height : float
        Maximum value of the bump.

    Returns
    -------
    torch.Tensor
        Bump profile, peaked at lambda_neck.
    """
    # Gaussian-like bump using smooth approximation
    # Use exp(-((x - x0)/sigma)^2) structure
    t = (lambda_coord - lambda_neck) / max(sigma, 1e-8)
    return peak_height * torch.exp(-t * t)


def neck_bump_normalized(
    lambda_coord: torch.Tensor,
    lambda_neck: float = 0.0,
    sigma: float = 0.1,
    domain: Tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Normalized neck bump with integral = 1 over the domain.

    Parameters
    ----------
    lambda_coord : torch.Tensor
        The neck coordinate.
    lambda_neck : float
        Center position of the neck region.
    sigma : float
        Width of the bump.
    domain : Tuple[float, float]
        Integration domain (lambda_L, lambda_R).

    Returns
    -------
    torch.Tensor
        Normalized bump profile.
    """
    # For a Gaussian, integral = sqrt(pi) * sigma
    # Normalize so integral over domain is approximately 1
    norm_factor = 1.0 / (math.sqrt(math.pi) * max(sigma, 1e-8))
    t = (lambda_coord - lambda_neck) / max(sigma, 1e-8)
    return norm_factor * torch.exp(-t * t)


def derivative_smooth_step(
    x: torch.Tensor,
    x0: float = 0.0,
    width: float = 0.1,
    steepness: float = 1.0,
) -> torch.Tensor:
    """Derivative of smooth_step with respect to x.

    Useful for constructing d(profile)/d(lambda) which appears
    in closure/coclosure constraints.

    Parameters
    ----------
    x, x0, width, steepness : same as smooth_step

    Returns
    -------
    torch.Tensor
        Derivative values (positive bump centered at x0).
    """
    w = max(width, 1e-8)
    t = steepness * (x - x0) / w
    sig = torch.sigmoid(t)
    # d/dx sigmoid(t) = sigmoid(t) * (1 - sigmoid(t)) * dt/dx
    return (steepness / w) * sig * (1.0 - sig)


@dataclass
class TCSProfiles:
    """Container for a family of TCS profile functions.

    This class bundles together the left/right/neck profiles with
    consistent parameterization for use in global mode construction.

    Attributes
    ----------
    lambda_L : float
        Position of left CY3 region.
    lambda_R : float
        Position of right CY3 region.
    lambda_neck : float
        Position of neck/gluing region.
    sigma_L : float
        Width of left profile transition.
    sigma_R : float
        Width of right profile transition.
    sigma_neck : float
        Width of neck bump.
    """

    lambda_L: float = 0.0
    lambda_R: float = 1.0
    lambda_neck: float = 0.5
    sigma_L: float = 0.15
    sigma_R: float = 0.15
    sigma_neck: float = 0.1

    def left(self, lambda_coord: torch.Tensor) -> torch.Tensor:
        """Evaluate left plateau profile."""
        return left_plateau(
            lambda_coord,
            lambda_L=self.lambda_L,
            lambda_neck=self.lambda_neck,
            sigma=self.sigma_L,
        )

    def right(self, lambda_coord: torch.Tensor) -> torch.Tensor:
        """Evaluate right plateau profile."""
        return right_plateau(
            lambda_coord,
            lambda_R=self.lambda_R,
            lambda_neck=self.lambda_neck,
            sigma=self.sigma_R,
        )

    def neck(self, lambda_coord: torch.Tensor) -> torch.Tensor:
        """Evaluate neck bump profile."""
        return neck_bump(
            lambda_coord,
            lambda_neck=self.lambda_neck,
            sigma=self.sigma_neck,
        )

    def neck_normalized(self, lambda_coord: torch.Tensor) -> torch.Tensor:
        """Evaluate normalized neck bump profile."""
        return neck_bump_normalized(
            lambda_coord,
            lambda_neck=self.lambda_neck,
            sigma=self.sigma_neck,
            domain=(self.lambda_L, self.lambda_R),
        )

    def all_profiles(self, lambda_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all three profiles evaluated at lambda_coord.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (left, right, neck) profiles.
        """
        return self.left(lambda_coord), self.right(lambda_coord), self.neck(lambda_coord)

    def profile_derivatives(
        self, lambda_coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return derivatives of all three profiles.

        Useful for computing d(omega)/d(lambda) terms in closure constraints.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (d_left/d_lambda, d_right/d_lambda, d_neck/d_lambda).
        """
        # d(left)/d(lambda) = -d(smooth_step)/d(lambda)
        d_left = -derivative_smooth_step(
            lambda_coord, x0=self.lambda_neck, width=self.sigma_L
        )

        # d(right)/d(lambda) = d(smooth_step)/d(lambda)
        d_right = derivative_smooth_step(
            lambda_coord, x0=self.lambda_neck, width=self.sigma_R
        )

        # d(neck)/d(lambda) for Gaussian: -2*t/sigma^2 * neck
        t = (lambda_coord - self.lambda_neck) / max(self.sigma_neck, 1e-8)
        neck_val = self.neck(lambda_coord)
        d_neck = -2.0 * t / max(self.sigma_neck, 1e-8) * neck_val

        return d_left, d_right, d_neck

    @classmethod
    def from_domain(
        cls,
        domain: Tuple[float, float] = (0.0, 1.0),
        neck_fraction: float = 0.5,
        transition_width: float = 0.15,
    ) -> "TCSProfiles":
        """Create TCSProfiles from domain specification.

        Parameters
        ----------
        domain : Tuple[float, float]
            (min, max) of lambda coordinate.
        neck_fraction : float
            Fraction of domain where neck is located (0.5 = center).
        transition_width : float
            Width of transitions as fraction of domain size.

        Returns
        -------
        TCSProfiles
            Configured profile container.
        """
        lambda_L, lambda_R = domain
        domain_size = lambda_R - lambda_L
        lambda_neck = lambda_L + neck_fraction * domain_size
        sigma = transition_width * domain_size

        return cls(
            lambda_L=lambda_L,
            lambda_R=lambda_R,
            lambda_neck=lambda_neck,
            sigma_L=sigma,
            sigma_R=sigma,
            sigma_neck=sigma * 0.7,  # Neck slightly narrower
        )
