import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boussinesq_units import background_profiles, diffusion_coefficients, dimensional_diffusivity, free_fall_scales
from metadata_io import load_run_metadata, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]


def test_default_matches_original():
    assert all(value == 1 for value in free_fall_scales(1).values())
    nu, kappa, kappa_C = diffusion_coefficients(5e6, 0.5, 0.001)
    assert nu == pytest.approx(np.sqrt(0.5/5e6))
    assert kappa == pytest.approx(np.sqrt(1/(5e6*0.5)))
    assert kappa_C == 0.001
    z = np.linspace(0, 3, 100)
    beta, T0 = background_profiles(z, 2, 5, 0.05)
    np.testing.assert_allclose(beta, 0.5*(-1-5)*(1+np.tanh((z-2)/0.05))+5)
    np.testing.assert_allclose(T0, 0.5*(-1+5)*z+0.5*(-1-5)*0.05*np.log(np.cosh((z-2)/0.05)))


def test_nonunit_reference_and_diffusivity_conversion():
    L, N2, g, alpha = 2.5, -4.0, 3.0, 0.2
    scales = free_fall_scales(L, N2, g, alpha)
    assert scales['DeltaT0'] == pytest.approx(L*abs(N2)/(g*alpha))
    assert scales['U_ff'] == pytest.approx(5)
    assert scales['t_ff'] == pytest.approx(0.5)
    assert g*alpha*scales['DeltaT0']*L/scales['U_ff']**2 == pytest.approx(1)
    nu, kappa, kappa_C = diffusion_coefficients(2e6, 0.4, 0.003)
    nu_dim, kappa_dim = dimensional_diffusivity(np.array([nu, kappa]), scales)
    assert nu_dim/kappa_dim == pytest.approx(0.4)
    assert abs(N2)*L**4/(nu_dim*kappa_dim) == pytest.approx(2e6)
    assert dimensional_diffusivity(kappa_C, scales) == pytest.approx(kappa_C*12.5)
    assert 1/kappa_C == pytest.approx(scales['U_ff']*L/dimensional_diffusivity(kappa_C, scales))


def test_beta_limits_and_temperature_derivative():
    S, tw, zb = 5.0, 0.08, 2.0
    beta, _ = background_profiles(np.array([zb-20*tw, zb+20*tw]), zb, S, tw)
    np.testing.assert_allclose(beta, [S, -1])
    z = np.linspace(zb-4*tw, zb+4*tw, 40)
    h = 1e-6
    beta, _ = background_profiles(z, zb, S, tw)
    _, plus = background_profiles(z+h, zb, S, tw)
    _, minus = background_profiles(z-h, zb, S, tw)
    np.testing.assert_allclose((plus-minus)/(2*h), beta, rtol=1e-7, atol=1e-8)
    scales = free_fall_scales(2.5, -4, 3, 0.2)
    np.testing.assert_allclose(beta*scales['DeltaT0']/scales['L0'], beta*4/(3*0.2))


@pytest.mark.parametrize('values', [(0, -1, 1, 1), (1, 1, 1, 1), (1, -1, 0, 1)])
def test_invalid_reference_scales(values):
    with pytest.raises(ValueError):
        free_fall_scales(*values)


def test_metadata_computational_geometry(tmp_path):
    tree = ast.parse((ROOT/'convection.py').read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'write_info')
    import os
    namespace = {'free_fall_scales': free_fall_scales, 'write_json_atomic': write_json_atomic, 'os': os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'convection.py', 'exec'), namespace)
    args = SimpleNamespace(L_cz=2, Lx=8, Lz=6, N2_cz=-9, g=2, alpha=0.5, run_id='synthetic',
                           Nx=16, Nz=8, dealias=1.5, Ra=1e6, Pr=1, S=5, tw=0.05,
                           max_timestep=0.1, n_cores=1, stop_sim_time=10)
    info = namespace['write_info'](tmp_path, args)
    assert info['params']['L_cz'] == 1
    assert info['params']['Lx'] == 4
    assert info['params']['Lz'] == 3
    assert info['params']['L_rz'] == 2
    assert info['units']['scales']['t_ff'] == pytest.approx(1/3)
    assert info['units']['reference']['g'] == 2
    info['background'] = {'g': 1.0, 'alpha': 1.0, 'N2_cz': -1.0}
    write_json_atomic(tmp_path/'info.json', info)
    _, normalized = load_run_metadata(tmp_path/'info.json')
    bg = normalized['background']
    assert bg['N2_cz']/(bg['g']*bg['alpha']) == -1
    assert normalized['params']['L_cz'] == 1
