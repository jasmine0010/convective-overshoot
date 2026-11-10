"""Convection-zone free-fall reference scales and dimensionless background."""

import numpy as np


def free_fall_scales(L_cz, N2_cz=-1.0, g=1.0, alpha=1.0):
    if not np.all(np.isfinite([L_cz, N2_cz, g, alpha])) or L_cz <= 0 or N2_cz >= 0 or g <= 0 or alpha <= 0:
        raise ValueError('Positive reference scales and negative N2_cz required')
    U_ff = L_cz*np.sqrt(abs(N2_cz))
    return {'L0': float(L_cz), 'DeltaT0': float(abs(N2_cz)*L_cz/(g*alpha)),
            'U_ff': float(U_ff), 't_ff': float(L_cz/U_ff), 'D0': float(U_ff*L_cz)}


def diffusion_coefficients(Ra, Pr, kappa_C):
    if not np.all(np.isfinite([Ra, Pr, kappa_C])) or Ra <= 0 or Pr <= 0 or kappa_C <= 0:
        raise ValueError('Ra, Pr and kappa_C must be positive')
    return float(np.sqrt(Pr/Ra)), float(1/np.sqrt(Ra*Pr)), float(kappa_C)


def background_profiles(z, zb, S, tw):
    """Return beta=dT0/dz and its antiderivative in free-fall units."""
    if not np.all(np.isfinite([S, tw])) or S <= 0 or tw <= 0:
        raise ValueError('S and tw must be positive')
    z = np.asarray(z)
    beta = 0.5*(-1-S)*(1+np.tanh((z-zb)/tw))+S
    T0 = 0.5*(-1+S)*z + 0.5*(-1-S)*tw*np.log(np.cosh((z-zb)/tw))
    return beta, T0


def dimensional_diffusivity(D, scales):
    """Convert D to reference units; physical inputs give physical diffusivity."""
    return np.asarray(D)*scales['U_ff']*scales['L0']
