from pathlib import Path

import numpy as np
import pytest

import diffusion_v3 as diffusion


def test_diffusivity_plot_uses_positive_linear_range_without_changing_data():
    z = np.linspace(0, 2, 6)
    Dt = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1e6])
    Dt_before = Dt.copy()
    Dmix = np.linspace(6.0, 0.5, len(z))

    fig, ax = diffusion.plot_mesa_Dt_model(z, Dt, z, Dmix, 1.0, 1.0, 5.0, 0.1)

    assert ax.get_yscale() == 'linear'
    assert ax.get_ylim() == pytest.approx((-0.06, 1.32))
    assert np.array_equal(ax.lines[0].get_xdata(), 1.0-z)
    assert np.array_equal(ax.lines[0].get_ydata(), Dt_before/5.0)
    assert np.array_equal(fig.axes[1].lines[-1].get_ydata(), Dt_before/5.0)
    assert ax.get_ylabel() == r'$D/D_{\rm conv}$'
    assert fig.axes[1].get_ylabel() == r'$D_{\rm point}/D_{\rm conv}$'
    assert np.array_equal(Dt, Dt_before)
    fig.canvas.draw()
    assert fig.legends[0].get_window_extent().y0-max(axis.get_window_extent().y1 for axis in fig.axes[:2]) > 8
    threshold = next(line for line in ax.lines if np.array_equal(line.get_ydata(), [1, 1]))
    assert threshold.get_color() == diffusion.diag.COLOR_THRESHOLD
    assert threshold.get_linestyle() == ':'
    assert threshold.get_linewidth() == diffusion.diag.LINEWIDTH_REFERENCE
    assert [line.get_label() for line in ax.lines[:2]] == [
        'Pointwise flux–gradient diagnostic',
        'Fitted 1D diffusivity',
    ]


def test_objective_plot_uses_viridis_and_labels_normalized_error():
    q0 = np.geomspace(0.01, 0.1, 3)
    lambda_ov = np.geomspace(0.02, 0.2, 3)
    objective = np.arange(1, 10, dtype=float).reshape(3, 3)

    fig, ax = diffusion.plot_mesa_objective_surface(q0, lambda_ov, objective, 0.03, 0.06)

    assert ax.collections[0].cmap.name == 'viridis'
    assert ax.get_legend() is None
    assert ax.lines[0].get_markersize() == diffusion.diag.MARKER_SIZE
    assert ax.lines[0].get_marker() == 'x'
    assert ax.lines[0].get_markeredgecolor() == ax.lines[0].get_color()
    assert not np.any(ax.collections[0].get_antialiased())
    assert fig.axes[-1].get_ylabel() == r'$\log_{10}\mathcal{E}$'


def test_prediction_plot_uses_simulation_time_and_shared_legend():
    time = np.array([0.0, 2.0, 4.0])
    z = np.array([0.0, 1.0, 2.0])
    C_dns = np.array([[0.0, 0.5, 1.0], [0.1, 0.5, 0.9], [0.2, 0.5, 0.8]])
    C_model = C_dns + 0.01

    fig, axes = diffusion.plot_1d_tracer_prediction(
        time, z, C_model, C_dns, 1.0, 1.0, 2.0, 'C1',
    )

    assert np.array_equal(axes[0].lines[0].get_xdata(), 1.0-z)
    assert axes[0].lines[0].get_color() == diffusion.diag.COLOR_SIMULATION
    assert axes[0].lines[1].get_color() == diffusion.diag.COLOR_MODEL
    assert axes[0].lines[0].get_linestyle() == '-'
    assert axes[0].lines[1].get_linestyle() == '--'
    assert all(line.get_linewidth() == diffusion.diag.LINEWIDTH_TIMESERIES for line in axes[0].lines[:6])
    assert np.array_equal(axes[1].lines[0].get_xdata(), time)
    assert [ax.get_ylabel() for ax in axes] == [r'$C$', r'$\mathrm{RMSE}(C)$', r'$M_{C,\rm RZ}$']
    assert len(fig.legends) == 1
    fig.canvas.draw()
    assert fig.legends[0].get_window_extent().y0-max(ax.get_window_extent().y1 for ax in axes) > 8
    assert not fig.legends[0].get_frame_on()
    assert [text.get_text() for text in axes[0].texts if text.get_text().startswith('(')] == ['(a)']


def test_diffusion_figures_are_saved_as_pdf():
    source = Path(diffusion.__file__).read_text()

    for stem in (
        '_p4_C1_1d_Dt',
        '_p4_C1_mesa_fit',
        '_p4_C1_mesa_objective',
        '_p4_C1_1d_prediction',
        '_p4_C1Dt_C2_1d_prediction',
        '_p4_C1Dt_C3_1d_prediction',
    ):
        assert stem in source

    assert "figure_name}.pdf'" in source
    assert "figure_name}.png'" in source


def test_optional_tracer_absent_task_is_unavailable_without_key_error():
    profiles = {'time': np.array([0.0, 5.0, 10.0]), 'C1_havg': np.ones((3, 2))}

    assert diffusion.get_optional_tracer_tasks({'C1_havg'}) == ['C1_havg']
    assert diffusion.get_optional_tracer_release(profiles, 'C2', None) is None


def test_optional_tracer_present_but_never_released_is_unavailable():
    profiles = {
        'time': np.array([0.0, 5.0, 10.0]),
        'C3_havg': np.zeros((3, 2)),
        'C3_total': np.zeros(3),
    }

    assert diffusion.get_optional_tracer_release(profiles, 'C3', None) is None
    assert diffusion.get_optional_tracer_release(profiles, 'C3', 4.0) is None


def test_optional_tracer_explicit_release_is_preserved_after_nonzero_output():
    profiles = {
        'time': np.array([0.0, 5.0, 10.0]),
        'C2_havg': np.array([[0.0, 0.0], [0.0, 0.0], [0.2, 0.3]]),
    }

    assert diffusion.get_optional_tracer_release(profiles, 'C2', 7.5) == 7.5


def test_optional_legacy_tracer_uses_first_nonzero_saved_output_time():
    profiles = {
        'time': np.array([0.0, 5.0, 10.0]),
        'C2_havg': np.array([[0.0, 0.0], [0.0, 0.0], [0.2, 0.3]]),
        'C2_total': np.array([0.0, 1.0, 1.0]),
    }

    tasks = diffusion.get_optional_tracer_tasks({'C1_havg', 'C2_havg', 'C2_total'})
    assert tasks == ['C1_havg', 'C2_havg', 'C2_total']
    assert diffusion.get_optional_tracer_release(profiles, 'C2', None) == 5.0


def test_mesa_profile_is_positive_and_matches_piecewise_definition():
    zb = 1.0
    L_cz = 1.0
    Lz = 2.0
    D_conv = 0.08
    q0 = 0.1
    lambda_ov = 0.2

    z, Dmix = diffusion.build_mesa_Dmix_1d(
        zb, L_cz, Lz, D_conv, q0, lambda_ov, Nz=160
    )
    q = (zb-z)/L_cz
    expected = np.where(
        q <= -q0,
        D_conv,
        D_conv*np.exp(-(q+q0)/lambda_ov),
    )

    assert np.all(np.isfinite(Dmix))
    assert np.all(Dmix > 0)
    assert np.allclose(Dmix, expected)
    assert D_conv*np.exp(-(-q0+q0)/lambda_ov) == D_conv


def test_diffusion_solver_uses_irregular_times_and_conserves_mass():
    z, Dmix = diffusion.build_mesa_Dmix_1d(
        zb=1.0,
        L_cz=1.0,
        Lz=2.0,
        D_conv=0.04,
        q0=0.08,
        lambda_ov=0.15,
        Nz=80,
    )
    C0 = np.exp(-((z-1.25)/0.12)**2)
    time_eval = np.array([0.03, 0.11, 0.27, 0.6])

    returned_time, C_model = diffusion.solve_tracer_diffusion_full(
        z,
        Dmix,
        kappa_C=1e-3,
        time_eval=time_eval,
        C0=C0,
        t_start=0.0,
    )

    assert np.array_equal(returned_time, time_eval)
    assert np.allclose(
        np.sum(C_model, axis=1),
        np.sum(C0),
        rtol=1e-11,
        atol=1e-11,
    )


def test_fit_recovers_mesa_profile_with_flat_convection_zone():
    Nz = 100
    Lz = 2.0
    zb = 1.0
    L_cz = 1.0
    D_conv = 0.05
    kappa_C = 1e-3
    q0_true = 0.08
    lambda_true = 0.18

    z, Dmix = diffusion.build_mesa_Dmix_1d(
        zb,
        L_cz,
        Lz,
        D_conv,
        q0_true,
        lambda_true,
        Nz=Nz,
    )
    C0 = (z >= zb).astype(float)
    time = np.array([0.0, 0.08, 0.22, 0.5, 1.0])
    _, evolved = diffusion.solve_tracer_diffusion_full(
        z,
        Dmix,
        kappa_C,
        time[1:],
        C0,
        time[0],
    )
    profiles = np.vstack((C0, evolved))

    fit = diffusion.fit_mesa_diffusivity(
        z,
        profiles,
        time,
        zb,
        L_cz,
        Lz,
        D_conv,
        kappa_C,
        Nz=Nz,
        initial_q0=0.03,
        initial_lambda=0.1,
    )

    assert fit['q0'] == pytest.approx(q0_true, rel=0.05)
    assert fit['lambda_ov'] == pytest.approx(lambda_true, rel=0.05)
    assert fit['objective_rms'] < 1e-8
    assert not fit['at_bounds']
    assert not fit['poorly_localized']


def test_uninformative_profiles_are_flagged_as_poorly_localized():
    z = (np.arange(40)+0.5)/20
    profiles = np.ones((3, len(z)))
    time = np.array([0.0, 0.1, 0.3])

    with pytest.warns(RuntimeWarning, match='poorly localized'):
        fit = diffusion.fit_mesa_diffusivity(
            z,
            profiles,
            time,
            zb=1.0,
            L_cz=1.0,
            Lz=2.0,
            D_conv=0.05,
            kappa_C=1e-3,
            Nz=40,
        )

    assert fit['poorly_localized']


def test_mesa_parameter_conversion():
    converted = diffusion.mesa_overshoot_parameters(
        q0=0.08,
        lambda_ov=0.12,
        L_cz=2.0,
        H_P=4.0,
    )

    assert converted == pytest.approx({'f': 0.12, 'f0': 0.04})


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'q0': -0.1}, 'q0'),
        ({'lambda_ov': 0.0}, 'lambda_ov'),
        ({'D_conv': 0.0}, 'D_conv'),
    ],
)
def test_mesa_profile_rejects_invalid_parameters(kwargs, message):
    parameters = {
        'zb': 1.0,
        'L_cz': 1.0,
        'Lz': 2.0,
        'D_conv': 0.05,
        'q0': 0.05,
        'lambda_ov': 0.1,
        'Nz': 40,
    }
    parameters.update(kwargs)

    with pytest.raises(ValueError, match=message):
        diffusion.build_mesa_Dmix_1d(**parameters)



def test_out_of_range_decay_is_reported_as_bound_limited():
    Nz = 60
    z, Dmix = diffusion.build_mesa_Dmix_1d(
        1.0, 1.0, 2.0, 0.05, 0.08, 3.0, Nz=Nz
    )
    C0 = (z >= 1.0).astype(float)
    time = np.array([0.0, 0.1, 0.3, 0.7, 1.5])
    _, evolved = diffusion.solve_tracer_diffusion_full(
        z, Dmix, 1e-3, time[1:], C0, time[0]
    )

    with pytest.warns(RuntimeWarning, match='reached a bound'):
        fit = diffusion.fit_mesa_diffusivity(
            z,
            np.vstack((C0, evolved)),
            time,
            1.0,
            1.0,
            2.0,
            0.05,
            1e-3,
            Nz=Nz,
        )

    assert fit['at_bounds']


def test_objective_surface_is_finite_and_contains_fit_neighbourhood():
    Nz = 40
    z, Dmix = diffusion.build_mesa_Dmix_1d(
        1.0, 1.0, 2.0, 0.04, 0.07, 0.16, Nz=Nz
    )
    C0 = (z >= 1.0).astype(float)
    time = np.array([0.0, 0.1, 0.3])
    _, evolved = diffusion.solve_tracer_diffusion_full(
        z, Dmix, 1e-3, time[1:], C0, time[0]
    )
    fit = diffusion.fit_mesa_diffusivity(
        z,
        np.vstack((C0, evolved)),
        time,
        1.0,
        1.0,
        2.0,
        0.04,
        1e-3,
        Nz=Nz,
    )

    q0, lambda_ov, objective = diffusion.evaluate_mesa_objective_surface(
        fit,
        1.0,
        1.0,
        2.0,
        0.04,
        1e-3,
        points=5,
    )

    assert objective.shape == (5, 5)
    assert np.all(np.isfinite(objective))
    assert q0[0] <= fit['q0'] <= q0[-1]
    assert lambda_ov[0] <= fit['lambda_ov'] <= lambda_ov[-1]


def test_pointwise_baseline_is_clipped_on_independent_grid():
    z_dns = np.linspace(0.05, 1.95, 39)
    z_1d = (np.arange(64)+0.5)*2/64
    Dt = 0.02*(z_dns-0.7)
    Dt[:4] = np.nan

    Dmix = diffusion.build_pointwise_Dmix_1d(Dt, z_dns, z_1d, 1.0)

    assert len(Dmix) != len(Dt)
    assert np.all(Dmix >= 0)
    assert np.any(Dmix > 0)


def test_model_comparison_preserves_one_continuous_initial_value_problem():
    z, fitted = diffusion.build_mesa_Dmix_1d(1.0, 1.0, 2.0, 0.04, 0.06, 0.14, Nz=48)
    C0 = np.exp(-((z-1.25)/0.18)**2)
    times = np.array([0.1, 0.25, 0.55])
    _, C_dns = diffusion.solve_tracer_diffusion_full(z, fitted, 1e-3, times, C0, 0.0, max_dt=0.025)
    models = {
        'fitted_exponential': fitted,
        'molecular_only': np.zeros_like(z),
        'constant_Dconv': np.full_like(z, 0.04),
        'nonnegative_pointwise': np.maximum(fitted-0.005, 0),
    }

    comparisons = diffusion.compare_diffusivity_models(
        z, models, 1e-3, times, C0, 0.0, C_dns, 1.0, 2.0, max_dt=0.025,
    )

    assert set(comparisons) == set(models)
    assert comparisons['fitted_exponential']['errors'][0][-1] < 1e-12
    assert comparisons['molecular_only']['errors'][0][-1] > 0
    for result in comparisons.values():
        assert result['C_model'].shape == C_dns.shape
        assert result['errors'][5] < 1e-11


def test_stable_layer_mass_error_is_normalized_by_injected_mass():
    z = (np.arange(4)+0.5)/2
    C_dns = np.ones((1, 4))
    C_model = C_dns.copy()
    C_model[0, :2] += 0.25

    errors = diffusion.compute_prediction_errors(z, C_model, C_dns, 1.0, 2.0, injected_mass=4.0)

    assert errors[4][0] == pytest.approx(0.0625)


def test_small_convergence_check_uses_two_grids_and_two_timesteps():
    z_dns = np.linspace(0, 2, 51)
    C0 = np.exp(-((z_dns-1.25)/0.2)**2)
    times = np.array([0.1, 0.25])
    z_ref, D_ref = diffusion.build_mesa_Dmix_1d(1.0, 1.0, 2.0, 0.04, 0.06, 0.14, Nz=80)
    _, C_ref = diffusion.solve_tracer_diffusion_full(
        z_ref, D_ref, 1e-3, times, np.interp(z_ref, z_dns, C0), 0.0, max_dt=0.01,
    )
    C_dns = np.array([np.interp(z_dns, z_ref, C) for C in C_ref])

    results = diffusion.numerical_convergence_check(
        z_dns, C0, times, C_dns, 0.0, 1.0, 1.0, 2.0, 0.04, 0.06, 0.14, 1e-3,
        resolutions=(40, 80), timesteps=(0.04, 0.02),
    )

    assert len(results) == 4
    assert set(results) == {'Nz=40,dt=0.04', 'Nz=40,dt=0.02', 'Nz=80,dt=0.04', 'Nz=80,dt=0.02'}
    assert all('final_rmse_full' in result for result in results.values())


def test_main_records_separate_comparison_output_and_frozen_protocol():
    source = Path(diffusion.__file__).read_text()

    assert '_diffusivity_validation.json' in source
    assert "'frozen_after_fit': True" in source
    assert "'one_continuous_trajectory_per_tracer': True" in source
    assert "'molecular_only'" in source
    assert "'constant_Dconv'" in source
    assert "'nonnegative_pointwise'" in source
