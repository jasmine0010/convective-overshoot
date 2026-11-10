import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

import diffusion_v4 as diffusion


def metadata():
    return {
        'params': {'Lz': 2.0, 'L_cz': 1.0, 'tw': 0.05},
        'background': {'zb': 1.0},
        'timing': {'release_time_C1': 1.0, 'release_time_C2': 2.0, 'release_time_C3': 3.0},
    }


@pytest.mark.parametrize(
    ('tracer', 'source', 'release_time'),
    [('C1', 'prescribed_tanh', 1.0), ('C2', 'prescribed_tanh', 2.0),
     ('C3', 'prescribed_gaussian', 3.0)],
)
def test_exact_release_profiles_are_cell_averaged(tracer, source, release_time):
    _, edges = diffusion.finite_volume_grid(2.0, 40)
    profile, start, actual_source = diffusion.release_initial_condition(tracer, edges, metadata())

    if tracer in ('C1', 'C2'):
        tw, zb = 0.05, 1.0
        primitive = lambda z: 0.5*(z+tw*np.logaddexp((z-zb)/tw, -(z-zb)/tw))
    else:
        mu, sigma = 1.15, 0.15
        primitive = lambda z: sigma*np.sqrt(np.pi/2)*erf((z-mu)/(np.sqrt(2)*sigma))
    expected = np.diff(primitive(edges))/np.diff(edges)

    assert start == release_time
    assert actual_source == source
    assert profile == pytest.approx(expected, rel=2e-11, abs=2e-11)


def test_scaled_C3_matches_C1_injected_mass():
    meta = metadata()
    Lz, zb, tw, L_cz = 2.0, 1.0, 0.05, 1.0
    mu, sigma = zb+0.15*L_cz, 0.15*L_cz
    a, b = (Lz-zb)/tw, zb/tw
    C1_mass = 0.5*(Lz+tw*(np.logaddexp(a, -a)-np.logaddexp(b, -b)))
    C3_mass = sigma*np.sqrt(np.pi/2)*(erf((Lz-mu)/(np.sqrt(2)*sigma))-erf(-mu/(np.sqrt(2)*sigma)))
    meta['background']['C3_amplitude'] = C1_mass/C3_mass

    _, edges = diffusion.finite_volume_grid(Lz, 80)
    C1, _, _ = diffusion.release_initial_condition('C1', edges, meta)
    C3, _, _ = diffusion.release_initial_condition('C3', edges, meta)
    assert np.sum(C3*np.diff(edges)) == pytest.approx(np.sum(C1*np.diff(edges)), rel=1e-12)
    assert np.max(C3) > np.max(C1)


def test_model_evaluation_uses_one_identical_rollout_per_baseline(monkeypatch):
    z, _ = diffusion.finite_volume_grid(2.0, 12)
    C0 = np.exp(-((z-1.2)/0.2)**2)
    times = np.array([1.2, 1.7, 2.4])
    models = {
        'fitted_exponential': np.linspace(0.04, 0.01, len(z)),
        'constant_training_fit': np.full_like(z, 0.02),
        'nonnegative_pointwise': np.linspace(0, 0.03, len(z)),
        'molecular_only': np.zeros_like(z),
    }
    calls = []

    def fake_solver(z_arg, D, kappa, times_arg, C0_arg, start, max_dt):
        calls.append((C0_arg.copy(), times_arg.copy(), start, max_dt))
        return times_arg, np.tile(C0_arg, (len(times_arg), 1))

    monkeypatch.setattr(diffusion, 'solve_tracer_diffusion_full', fake_solver)
    diffusion.evaluate_models(
        z, models, 1e-3, times, C0, 1.0, np.tile(C0, (len(times), 1)), 1.0, 2.0, 0.05,
    )

    assert len(calls) == len(models)
    assert all(np.array_equal(call[0], C0) and np.array_equal(call[1], times) for call in calls)
    assert all(call[2:] == (1.0, 0.05) for call in calls)


def test_fit_is_training_only_and_constant_baseline_is_fitted():
    assert 'validation' not in inspect.signature(diffusion.fit_frozen_models).parameters

    z, _ = diffusion.finite_volume_grid(2.0, 36)
    C0 = (z >= 1.0).astype(float)
    times = np.array([0.1, 0.25, 0.5])
    w_rms_z = np.full_like(z, 0.2)
    _, D_true = diffusion.build_Dmix_1d(1.0, 1.0, 2.0, 0.6, 0.06, 0.16, z, w_rms_z, Nz=len(z))
    _, training = diffusion.solve_tracer_diffusion_full(z, D_true, 1e-3, times, C0, 0.0, max_dt=0.025)

    models_1, fit_1 = diffusion.fit_frozen_models(
        z, C0, 0.0, times, training, 1.0, 1.0, 2.0, 0.2, 1e-3,
        np.maximum(D_true-0.005, 0), 0.025, z, w_rms_z,
    )
    held_out = training.copy()
    held_out[-1] += 10
    models_2, fit_2 = diffusion.fit_frozen_models(
        z, C0, 0.0, times, training, 1.0, 1.0, 2.0, 0.2, 1e-3,
        np.maximum(D_true-0.005, 0), 0.025, z, w_rms_z,
    )

    assert fit_1 == fit_2
    assert np.array_equal(models_1['fitted_exponential'], models_2['fitted_exponential'])
    assert np.all(models_1['constant_training_fit'] == pytest.approx(fit_1['constant_D']))
    assert not np.shares_memory(held_out, training)
    assert fit_1['D_cz'] == pytest.approx(fit_1['ell_mix_fit']*0.2/3)


@pytest.mark.parametrize('Nz', [24, 65])
def test_depth_dependent_diffusivity_uses_raw_velocity_and_off_grid_reference(Nz):
    z_dns = np.array([0., 0.2, 0.8, 1.05, 1.2, 1.5, 2.])
    w_rms_z = np.array([0., 0.04, 0.1, 0.2, 0.35, 0.3, 0.])
    ell_mix, q0, lambda_ov = 0.6, 0.173, 0.16
    z, D = diffusion.build_Dmix_1d(1., 1., 2., ell_mix, q0, lambda_ov, z_dns, w_rms_z, Nz=Nz)
    z_ref = 1.+q0
    convective = z >= z_ref
    D_ref = ell_mix*np.interp(z_ref, z_dns, w_rms_z)/3

    assert D[convective] == pytest.approx(ell_mix*np.interp(z[convective], z_dns, w_rms_z)/3)
    assert D[~convective] == pytest.approx(D_ref*np.exp((z[~convective]-z_ref)/lambda_ov))
    assert np.all(np.isfinite(D)) and np.all(D >= 0)
    _, stronger_flow = diffusion.build_Dmix_1d(1., 1., 2., ell_mix, q0, lambda_ov, z_dns, 4*w_rms_z, Nz=Nz)
    assert stronger_flow == pytest.approx(4*D)


def test_mixing_length_fit_recovers_variable_velocity_model():
    z, _ = diffusion.finite_volume_grid(2., 56)
    z_dns = 1-np.cos(np.linspace(0, np.pi, 80))
    w_rms_z = 0.2+0.08*np.sin(np.pi*z_dns/2)**2
    time_mask = np.array([True])
    w_rms_cz = diffusion.diag.compute_u_rms_from_moments(w_rms_z[None, :]**2, z_dns, (1., 2.), time_mask)
    C0 = np.exp(-((z-1.45)/0.25)**2)
    times = np.array([0.012, 0.035, 0.061, 0.09, 0.13])
    _, D = diffusion.build_Dmix_1d(1., 1., 2., 0.6, 0.12, 0.18, z_dns, w_rms_z, Nz=len(z))
    _, training = diffusion.solve_tracer_diffusion_full(z, D, 1e-3, times, C0, 0., 0.005)

    models, fit = diffusion.fit_frozen_models(z, C0, 0., times, training, 1., 1., 2., w_rms_cz, 1e-3, D, 0.005, z_dns, w_rms_z)
    assert fit['exponential_fit_success']
    assert fit['ell_mix_fit'] == pytest.approx(0.6, rel=1e-3)
    assert fit['q0'] == pytest.approx(0.12, abs=1e-3)
    assert fit['lambda_over_Lcz'] == pytest.approx(0.18, rel=1e-3)
    assert models['fitted_exponential'] == pytest.approx(D, abs=1e-6)
    expected_boundary = fit['ell_mix_fit']*np.interp(1.+fit['q0'], z_dns, w_rms_z)*np.exp(-fit['q0']/fit['lambda_over_Lcz'])/3
    assert fit['D_boundary'] == pytest.approx(expected_boundary)


def test_metric_normalization_and_mass_conservation():
    z, _ = diffusion.finite_volume_grid(2.0, 40)
    C0 = np.exp(-((z-1.25)/0.2)**2)
    times = np.array([0.1, 0.3, 0.8])
    D = np.full_like(z, 0.03)
    _, model = diffusion.solve_tracer_diffusion_full(z, D, 1e-3, times, C0, 0.0, max_dt=0.02)
    dns = model.copy()
    stable = z < 1.0
    dns[:, stable] -= 0.01

    metrics = diffusion.prediction_metrics(z, model, dns, C0, 1.0, 2.0)
    scaled_metrics = diffusion.prediction_metrics(z, 3*model, 3*dns, 3*C0, 1.0, 2.0)
    injected_mass = np.sum(C0)*(z[1]-z[0])
    expected = np.sum((model[0]-dns[0])*(z[1]-z[0])*(z < 1.0))/injected_mass

    assert metrics['stable_layer_mass_error_over_injected_mass'][0] == pytest.approx(expected)
    assert metrics['normalized_full_profile_rmse'] == pytest.approx(metrics['full_profile_rmse']/np.max(C0))
    assert scaled_metrics['normalized_full_profile_rmse'] == pytest.approx(metrics['normalized_full_profile_rmse'])
    assert np.max(np.abs(metrics['total_mass_drift'])) < 1e-11
    summary = diffusion.metric_summary(metrics)
    assert set(summary) == {'mean_full_profile_rmse', 'mean_stable_layer_rmse',
                            'final_stable_layer_mass_error_over_injected_mass', 'max_abs_total_mass_drift'}
    assert summary['mean_stable_layer_rmse'] == pytest.approx(np.mean(metrics['stable_layer_rmse']))
    assert all(np.isscalar(value) for value in summary.values())


def test_conservative_remap_preserves_piecewise_linear_mass():
    z_source = np.linspace(0, 2, 17)
    profiles = np.vstack((1+z_source, 2-z_source/2))
    _, edges = diffusion.finite_volume_grid(2.0, 23)

    mapped = diffusion.remap(z_source, profiles, edges)

    assert np.sum(mapped[0]*np.diff(edges)) == pytest.approx(np.trapezoid(profiles[0], z_source))
    assert np.sum(mapped[1]*np.diff(edges)) == pytest.approx(np.trapezoid(profiles[1], z_source))


def test_two_resolution_two_timestep_check():
    meta = metadata()
    z_ref, edges_ref = diffusion.finite_volume_grid(2.0, 80)
    C0, _, _ = diffusion.release_initial_condition('C1', edges_ref, meta)
    times = np.array([1.1, 1.25])
    w_rms_z = 0.2+0.05*np.sin(np.pi*z_ref/2)
    _, D_ref = diffusion.build_Dmix_1d(1.0, 1.0, 2.0, 0.6, 0.06, 0.14, z_ref, w_rms_z, Nz=80)
    _, dns = diffusion.solve_tracer_diffusion_full(z_ref, D_ref, 1e-3, times, C0, 1.0, max_dt=0.01)

    results = diffusion.convergence_check(
        z_ref, dns, times, 'C1', meta, 1.0, 1.0, 2.0, 0.6, 0.06, 0.14, 1e-3,
        resolutions=(40, 80), timesteps=(0.04, 0.02), w_rms_z=w_rms_z,
    )

    assert set(results) == {
        'Nz=40,dt=0.04', 'Nz=40,dt=0.02',
        'Nz=80,dt=0.04', 'Nz=80,dt=0.02',
    }
    assert all(result['max_abs_total_mass_drift'] < 1e-11 for result in results.values())


def test_diffusivity_figure_shows_exact_solver_inputs():
    z, _ = diffusion.finite_volume_grid(2.0, 80)
    D_cz = 0.04
    models = {
        'fitted_exponential': np.linspace(0.001, D_cz, len(z)),
        'constant_training_fit': np.full_like(z, 0.02),
        'flux-gradient': np.linspace(0, 0.4, len(z)),
        'molecular_only': np.zeros_like(z),
        'cz_mixing_only': np.where(z >= 1., 0.04, 0.),
    }
    original = {name: D.copy() for name, D in models.items()}

    fig, axes = diffusion.plot_model_diffusivities(z, 1.0, 1.0, models)

    for ax in axes:
        lines = {line.get_label(): line for line in ax.lines}
        assert ax.get_yscale() == 'linear'
        assert np.asarray(lines['Exponential fit'].get_ydata()) == pytest.approx(models['fitted_exponential'])
        assert np.asarray(lines['Constant fit'].get_ydata()) == pytest.approx(models['constant_training_fit'])
        assert all('Flux-gradient' not in label for label in lines)
        assert 'Molecular only' not in lines
        assert np.asarray(lines['CZ only'].get_ydata()) == pytest.approx(models['cz_mixing_only'])
        assert lines['Convective boundary'].get_color() == diffusion.diag.COLOR_BOUNDARY
    assert axes[0].get_ylim()[1] == pytest.approx(1.1*D_cz)
    assert len(axes) == 1
    depth = diffusion.diag.normalized_depth(z, 1.0, 1.0)
    assert axes[0].get_xlim() == pytest.approx((np.min(depth), np.max(depth)))
    assert all(np.array_equal(models[name], original[name]) for name in models)
    assert axes[0].get_ylabel() == r'$D_{\rm turb}$'


def test_residual_colorbar_stays_outside_tracer_panels():
    z, _ = diffusion.finite_volume_grid(2.0, 30)
    times = np.array([1400.0, 1450.0, 1500.0])
    dns = np.tile(np.linspace(0, 1, len(z)), (len(times), 1))
    validation = {
        tracer: {'times': times, 'dns': dns,
                 'evaluated': {'fitted_exponential': {'profiles': dns + 0.01*(index+1)}}}
        for index, tracer in enumerate(('C1', 'C2', 'C3'))
    }

    fig, axes = diffusion.plot_residuals(z, 1.0, 1.0, validation)
    fig.canvas.draw()

    assert fig.axes[-1].get_position().x0 > axes[-1].get_position().x1
    assert fig.axes[-1].get_ylabel() == '1D - 2D concentration'


def test_rmse_plot_uses_raw_stable_and_full_profile_errors():
    series = {
        tracer: {'times': np.array(times), 'metrics': {'stable_layer_rmse': np.array([0.1, 0.2])*scale,
                                                    'full_profile_rmse': np.array([0.3, 0.4])*scale}}
        for tracer, times, scale in [('C1', [801, 802], 1), ('C2', [1402, 1404], 2), ('C3', [1503, 1506], 3)]
    }
    fig, axes = diffusion.plot_rmse(series, (1220, 1360))
    for ax, metric in zip(axes, ('stable_layer_rmse', 'full_profile_rmse')):
        lines = {line.get_label(): line for line in ax.lines}
        for tracer, result in series.items():
            assert np.asarray(lines[tracer].get_xdata()) == pytest.approx(result['times'])
            assert np.asarray(lines[tracer].get_ydata()) == pytest.approx(result['metrics'][metric])
        fit_band = ax.patches[0]
        assert (fit_band.get_x(), fit_band.get_x()+fit_band.get_width()) == pytest.approx([1220, 1360])
    assert axes[1].get_xlabel() == 'Simulation time'
    assert [ax.get_ylabel() for ax in axes] == ['Stable-layer RMSE', 'Full-profile RMSE']
    assert 'C1 fit interval' in [text.get_text() for text in fig.legends[0].get_texts()]


def test_matched_age_rmse_uses_common_elapsed_time_range():
    releases = {'C1': 800.0, 'C2': 1400.0, 'C3': 1400.0}
    ages = {'C1': np.array([1., 5., 9.]), 'C2': np.array([2., 4., 8.]), 'C3': np.array([1., 3., 6.])}
    series = {tracer: {'times': ages[tracer]+releases[tracer],
                      'metrics': {'stable_layer_rmse': np.array([0.1, 0.2, 0.3]),
                                  'full_profile_rmse': np.array([0.4, 0.5, 0.6])}}
              for tracer in releases}
    fig, axes = diffusion.plot_rmse_matched_age(series, releases, (1220, 1360))

    for ax, metric in zip(axes, ('stable_layer_rmse', 'full_profile_rmse')):
        assert ax.get_xlim() == pytest.approx((2, 6))
        assert len(ax.patches) == 0
        for line in ax.lines:
            tracer = line.get_label()
            assert np.asarray(line.get_xdata()) == pytest.approx(ages[tracer])
            assert np.asarray(line.get_ydata()) == pytest.approx(series[tracer]['metrics'][metric])
            assert series[tracer]['times'] == pytest.approx(ages[tracer]+releases[tracer])
    assert axes[1].get_xlabel() == 'Time since release'
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ['C1', 'C2', 'C3']


def test_scaled_residuals_use_separate_symmetric_colorbars():
    z, _ = diffusion.finite_volume_grid(2.0, 30)
    times = np.array([1400.0, 1450.0, 1500.0])
    dns = np.zeros((len(times), len(z)))
    validation = {
        tracer: {'times': times, 'dns': dns,
                 'evaluated': {'fitted_exponential': {'profiles': dns + amplitude}}}
        for tracer, amplitude in zip(('C1', 'C2', 'C3'), (0.001, 0.01, 0.2))
    }

    fig, axes = diffusion.plot_residuals_scaled(z, 1.0, 1.0, validation)

    assert len(fig.axes) == 6
    assert [ax.get_ylabel() for ax in axes] == ['Time', '', '']
    assert all(ax.get_ylabel() == '' for ax in fig.axes[3:])
    for ax, amplitude in zip(axes, (0.001, 0.01, 0.2)):
        assert ax.collections[0].get_clim() == pytest.approx((-amplitude, amplitude))
        assert any('RMSE' in label.get_text() for label in ax.texts)
    assert any('99th percentile' in label.get_text() for label in fig.texts)


def test_v4_outputs_are_separate_from_run_metadata():
    source = Path(diffusion.__file__).read_text()

    assert '_diffusivity_validation_v4.json' in source
    for name in ('v4_diffusivity_comparison', 'v4_exponential_residuals',
                 'v4_exponential_residuals_scaled', 'v4_model_diffusivities', 'v4_rmse', 'v4_rmse_matched_age'):
        assert f"'{name}'" in source
    assert "fig.savefig(f'{figure_dir}/{args.run_id}_{name}.pdf')" in source
    assert "fig.savefig(f'{figure_dir}/{args.run_id}_{name}.png', dpi=180)" in source
    assert "write_json_atomic(f'{run_folder}/info.json'" not in source
    assert "'errors':" in source
    assert "'C1_training_window':" in source
    assert "'convergence_C1':" in source
    assert "'time_resolved':" not in source
