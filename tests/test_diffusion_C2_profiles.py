import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('version', [4, 5])
def test_C2_profiles_show_same_times_and_exponential_only(version):
    diffusion = importlib.import_module(f'diffusion_v{version}')
    z = np.linspace(0, 3, 12)
    release = 2200.
    ages = np.array([42., 52., 198., 202., 398., 399.9])
    dns = np.arange(len(ages))[:, None]+z[None, :]
    result = {'times': release+ages, 'dns': dns, 'evaluated': {
        'fitted_exponential': {'profiles': dns+0.01}}}
    fig, axes = diffusion.plot_C2_profiles(z, 2., 1., result, release)
    fig.canvas.draw()
    assert axes.shape == (2, 3)
    for ax, residual_ax, index in zip(axes[0], axes[1], [1, 2, 5]):
        lines = {line.get_label(): line for line in ax.lines}
        assert lines['2D simulation'].get_ydata() == pytest.approx(dns[index])
        assert lines['Exponential fit'].get_ydata() == pytest.approx(dns[index]+0.01)
        assert 'Flux-gradient profile' not in lines
        assert lines['2D simulation'].get_xdata() == pytest.approx(2.-z)
        assert ax.get_title() == f'{ages[index]:.0f} after release'
        assert ax.lines[-1].get_color() == diffusion.diag.COLOR_BOUNDARY
        assert ax.get_shared_y_axes().joined(axes[0, 0], ax)
        residual_lines = {line.get_label(): line for line in residual_ax.lines}
        assert residual_lines['Exponential fit'].get_ydata() == pytest.approx(0.01)
        assert 'Flux-gradient profile' not in residual_lines
        assert residual_ax.get_shared_x_axes().joined(ax, residual_ax)
        assert residual_ax.get_shared_y_axes().joined(axes[1, 0], residual_ax)
        assert not residual_ax.get_shared_y_axes().joined(ax, residual_ax)
        assert residual_ax.get_position().height < ax.get_position().height/2
        assert residual_ax.lines[-1].get_color() == diffusion.diag.COLOR_BOUNDARY
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ['2D simulation', 'Exponential fit']
    assert f"'v{version}_C2_profiles_over_time'" in Path(diffusion.__file__).read_text()


@pytest.mark.parametrize('version', [4, 5])
def test_flux_gradient_is_never_plotted_or_evolved(version, monkeypatch):
    diffusion = importlib.import_module(f'diffusion_v{version}')
    z, _ = diffusion.finite_volume_grid(2., 12)
    C0 = np.linspace(0, 1, len(z))
    times = np.array([0.1, 0.3])
    dns = np.tile(C0, (len(times), 1))
    models = {name: np.full_like(z, D) for name, D in [('fitted_exponential', 0.02), ('constant_training_fit', 0.01), ('molecular_only', 0.), ('cz_mixing_only', 0.015), ('flux-gradient', 0.03)]}
    calls = []

    def solver(z, D, kappa, times, initial, start, max_dt):
        calls.append(D)
        return times, np.tile(initial, (len(times), 1))

    monkeypatch.setattr(diffusion, 'solve_tracer_diffusion_full', solver)
    evaluated = diffusion.evaluate_models(z, models, 1e-3, times, C0, 0., dns, 1., 2., 0.01)
    assert len(calls) == 4
    assert 'flux-gradient' not in evaluated
    assert all(D is not models['flux-gradient'] for D in calls)
    validation = {'C2': {'times': times, 'dns': dns, 'evaluated': evaluated}}
    fig, _ = diffusion.plot_comparison(z, 1., 1., validation)
    molecular_line = next(line for line in fig.axes[0].lines if line.get_label() == 'Molecular only')
    assert molecular_line.get_color() == '#CC79A7'
    assert molecular_line.get_linestyle() == '--'
    assert all('Flux-gradient' not in text.get_text() for text in fig.legends[0].get_texts())
    fig, axes = diffusion.plot_model_diffusivities(z, 1., 1., models)
    assert len(axes) == 1
    depth = diffusion.diag.normalized_depth(z, 1., 1.)
    assert axes[0].get_xlim() == pytest.approx((np.min(depth), np.max(depth)))
    for ax in axes:
        assert all('Flux-gradient' not in line.get_label() for line in ax.lines)
        assert all(line.get_label() != 'Molecular only' for line in ax.lines)
    assert axes[0].get_ylim()[1] == pytest.approx(1.1*np.max(models['fitted_exponential']))
