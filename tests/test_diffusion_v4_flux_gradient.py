from types import SimpleNamespace

import numpy as np
import pytest

import diffusion_v4 as diffusion


def test_flux_gradient_preserves_resolved_intervals_and_missing_values():
    z_dns = np.arange(7, dtype=float)
    Dt = np.array([np.nan, 0.2, 0.4, np.nan, 0., -0.1, np.nan])
    z = np.array([0.5, 1.5, 2.5, 3.5, 4., 4.5, 5., 5.5])
    D = diffusion.build_pointwise_Dmix_1d(Dt, z_dns, z, 3.)
    assert D[[1, 4, 5, 6]] == pytest.approx([0.3, 0., 0., 0.])
    assert np.all(np.isnan(D[[0, 2, 3, 7]]))


def test_unresolved_cz_uses_fitted_profile_without_replacing_measured_cz(monkeypatch):
    monkeypatch.setattr(diffusion, 'least_squares', lambda *args, **kwargs: SimpleNamespace(x=np.log([0.6, 0.3, 0.18]), jac=np.eye(3), success=True))
    monkeypatch.setattr(diffusion, 'minimize_scalar', lambda *args, **kwargs: SimpleNamespace(x=np.log(0.01), success=True))
    z, _ = diffusion.finite_volume_grid(2., 8)
    pointwise = np.array([np.nan, 0.002, np.nan, -0.003, 0.004, 0., np.nan, np.nan])
    original = pointwise.copy()
    models, _ = diffusion.fit_frozen_models(z, np.ones_like(z), 0., np.array([0.1]), np.ones((1, len(z))), 1., 1., 2., 0.2, 1e-3, pointwise, 0.01, z, np.full_like(z, 0.2))
    D = models['flux-gradient']
    assert D[[0, 2]] == pytest.approx(0.)
    assert D[[1, 3, 4, 5]] == pytest.approx([0.002, 0., 0.004, 0.])
    assert np.array_equal(D[6:], models['fitted_exponential'][6:])
    assert np.array_equal(pointwise, original, equal_nan=True)
    assert np.all(np.isfinite(D)) and np.all(D >= 0)
