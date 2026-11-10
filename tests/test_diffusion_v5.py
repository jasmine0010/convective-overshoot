import inspect
from pathlib import Path

import numpy as np
import pytest

import diffusion_v5 as diffusion


@pytest.mark.parametrize('Nz', [24, 41, 65])
def test_mixing_length_covers_entire_cz_and_exponential_starts_at_boundary(Nz):
    z_dns = np.array([0., 0.2, 0.8, 1.05, 1.2, 1.5, 2.])
    w = np.array([0., 0.04, 0.1, 0.2, 0.35, 0.3, 0.])
    ell, decay = 0.6, 0.16
    z, D = diffusion.build_Dmix_1d(1., 1., 2., ell, decay, z_dns, w, Nz=Nz)
    cz = z >= 1.
    D_boundary = ell*np.interp(1., z_dns, w)/3
    assert D[cz] == pytest.approx(ell*np.interp(z[cz], z_dns, w)/3)
    assert D[~cz] == pytest.approx(D_boundary*np.exp((z[~cz]-1.)/decay))
    if Nz == 41:
        assert D[Nz//2] == pytest.approx(D_boundary)
    _, stronger = diffusion.build_Dmix_1d(1., 1., 2., ell, decay, z_dns, 4*w, Nz=Nz)
    assert stronger == pytest.approx(4*D)
    assert np.all(np.isfinite(D)) and np.all(D >= 0)


@pytest.fixture
def fitted_model():
    z, _ = diffusion.finite_volume_grid(2., 56)
    z_dns = 1-np.cos(np.linspace(0, np.pi, 80))
    w = 0.2+0.08*np.sin(np.pi*z_dns/2)**2
    w_cz = diffusion.diag.compute_u_rms_from_moments(w[None, :]**2, z_dns, (1., 2.), np.array([True]))
    C0 = np.exp(-((z-1.45)/0.25)**2)
    times = np.array([0.012, 0.035, 0.061, 0.09, 0.13, 0.3, 0.6])
    _, D = diffusion.build_Dmix_1d(1., 1., 2., 0.6, 0.18, z_dns, w, Nz=len(z))
    _, training = diffusion.solve_tracer_diffusion_full(z, D, 1e-3, times, C0, 0., 0.005)
    pointwise = np.linspace(-0.01, 0.01, len(z))
    original = pointwise.copy()
    models, fit = diffusion.fit_frozen_models(z, C0, 0., times, training, 1., 1., 2., w_cz, 1e-3, pointwise, 0.005, z_dns, w)
    assert np.array_equal(pointwise, original)
    return z, z_dns, w, C0, D, pointwise, models, fit


def test_fit_recovers_only_mixing_length_and_decay(fitted_model):
    z, z_dns, w, _, D, _, models, fit = fitted_model
    assert fit['exponential_fit_success']
    assert fit['ell_mix_fit'] == pytest.approx(0.6, rel=1e-3)
    assert fit['lambda_over_Lcz'] == pytest.approx(0.18, rel=1e-3)
    assert models['fitted_exponential'] == pytest.approx(D, abs=1e-6)
    assert fit['D_boundary'] == pytest.approx(fit['ell_mix_fit']*np.interp(1., z_dns, w)/3)
    assert 'q0' not in fit
    assert 'validation' not in inspect.signature(diffusion.fit_frozen_models).parameters


def test_flux_gradient_shares_cz_without_modifying_rz(fitted_model):
    z, _, _, _, _, pointwise, models, _ = fitted_model
    cz = z >= 1.
    assert np.array_equal(models['flux-gradient'][cz], models['fitted_exponential'][cz])
    assert np.array_equal(models['flux-gradient'][~cz], np.maximum(pointwise[~cz], 0))
    assert np.array_equal(models['cz_mixing_only'][cz], models['fitted_exponential'][cz])
    assert np.all(models['cz_mixing_only'][~cz] == 0)


def test_frozen_models_conserve_mass_with_irregular_validation_times(fitted_model):
    z, _, _, C0, _, _, models, _ = fitted_model
    originals = {name: D.copy() for name, D in models.items()}
    times = np.array([0.171, 0.24, 0.39])
    dns = np.tile(C0, (len(times), 1))
    results = diffusion.evaluate_models(z, models, 1e-3, times, C0, 0., dns, 1., 2., 0.005)
    for name, result in results.items():
        assert np.max(np.abs(result['metrics']['total_mass_drift'])) < 1e-11
        assert np.array_equal(models[name], originals[name])


def test_convergence_uses_same_boundary_anchored_profile():
    metadata = {'params': {'Lz': 2., 'L_cz': 1., 'tw': 0.05}, 'background': {'zb': 1.}, 'timing': {'release_time_C1': 1.}}
    z, edges = diffusion.finite_volume_grid(2., 80)
    C0, _, _ = diffusion.release_initial_condition('C1', edges, metadata)
    w = 0.2+0.05*np.sin(np.pi*z/2)
    times = np.array([1.1, 1.25])
    _, D = diffusion.build_Dmix_1d(1., 1., 2., 0.6, 0.14, z, w, Nz=80)
    _, dns = diffusion.solve_tracer_diffusion_full(z, D, 1e-3, times, C0, 1., 0.01)
    results = diffusion.convergence_check(z, dns, times, 'C1', metadata, 1., 1., 2., 0.6, 0.14, 1e-3, (40, 80), (0.04, 0.02), w)
    assert len(results) == 4
    assert all(result['max_abs_total_mass_drift'] < 1e-11 for result in results.values())


def test_v5_outputs_do_not_overwrite_v4_or_run_metadata():
    source = Path(diffusion.__file__).read_text()
    assert 'diffusion_v4' not in source
    assert 'q0' not in source
    assert '_diffusivity_validation_v5.json' in source
    for name in ('diffusivity_comparison', 'rmse', 'rmse_matched_age', 'exponential_residuals', 'model_diffusivities'):
        assert f"'v5_{name}'" in source
    assert "write_json_atomic(f'{run_folder}/info.json'" not in source
