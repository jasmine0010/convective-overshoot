from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASKS = {
    'velocity': 'velocity',
    'thermal': 'entropy_perturbation',
    'thermal_profile': 's_havg',
    'convective_flux': 'F_conv_havg',
    'C1': 'C1_concentration',
    'C2': 'C2_concentration'
}


def get_run_folder(run_id, mode):
    base = PROJECT_ROOT / 'data' / f'{mode}s'
    found = sorted(base.glob(f'{run_id}_*'))
    if not found:
        raise FileNotFoundError(f'No run folder: {base}/{run_id}_*')
    if len(found) > 1:
        raise RuntimeError(f'Ambiguous run folder: {run_id!r}')
    return str(found[0])


def compute_Nu_anelastic(p, ds0dz_cz_ref, z_grid, cz_mask, rho0, T0, chi, ds0dz):
    F_cond = -rho0*T0*chi*ds0dz
    Nu_havg = (p[TASKS['convective_flux']] + F_cond) / (-chi * ds0dz_cz_ref)
    Nu = Nu_havg[:,cz_mask].mean(axis=1)
    return Nu, Nu_havg, F_cond


def compute_turbulent_thermal_diffusivity(p, z_grid, rho0, T0, chi, ds0dz):
    return -p[TASKS['convective_flux']] / (rho0*T0*ds0dz)
