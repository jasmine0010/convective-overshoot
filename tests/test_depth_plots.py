import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator

import diagnostics as diag


ROOT = Path(__file__).resolve().parents[1]
DEPTH_LABEL = 'Depth'


def depth_grid():
    return np.linspace(0.0, 1.5, 7)


def text_values(ax):
    return [text.get_text() for text in ax.texts]


def reference_labels(ax):
    return {text.get_text() for text in ax.get_legend().get_texts()}


def has_horizontal_line(ax, value):
    return any(np.size(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), value) for line in ax.lines)


def expected_flux_depth(profile, z, zb, peak_kind):
    rz = z < zb
    z_rz = z[rz]
    stable = profile[rz]
    idx_peak = np.argmax(stable) if peak_kind == 'max' else np.argmin(stable)
    threshold = 0.05*stable[idx_peak]
    roots = PchipInterpolator(z_rz, stable).solve(y=threshold, extrapolate=False)
    roots_deep = roots[roots < z_rz[idx_peak]]
    return zb-np.max(roots_deep)


def test_variance_plots_use_normalized_depth_profiles_without_mutating_input():
    z = depth_grid()
    variance = np.array([0.02, 0.05, 0.18, 0.5, 1.0, 0.65, 0.2])
    original = variance.copy()

    fig, ax = diag.plot_w_variance(variance, z, 1.0, 1.0, 0.55, 0.03)

    np.testing.assert_allclose(ax.lines[0].get_xdata(), 1.0-z)
    assert np.nanmax(ax.lines[0].get_ydata()) == 1.0
    assert ax.get_xlabel() == 'Depth'
    assert ax.get_ylabel() == r'$\widetilde{\sigma}_w^2$'
    assert len(ax.patches) == 1
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        'Convective boundary', 'Variance depth',
    ]
    assert ax.get_legend().get_frame_on() is False
    assert ax.lines[1].get_color() == '#D62728'
    np.testing.assert_array_equal(variance, original)
    plt.close(fig)


def test_variance_symmetry_has_normalized_panels_threshold_and_shared_legend():
    z = depth_grid()
    variance = np.array([0.02, 0.06, 0.2, 0.55, 1.0, 0.6, 0.15])
    original = variance.copy()

    fig, axes = diag.plot_w_variance_symmetry(1.5, 1.0, variance, z, 1.0)

    np.testing.assert_allclose(axes[0].lines[2].get_xdata(), 1.0-z)
    assert np.nanmax(axes[0].lines[2].get_ydata()) == 1.0
    assert axes[-1].get_xlabel() == DEPTH_LABEL
    assert all(len(ax.patches) == 1 for ax in axes)
    residual = axes[1].lines[2].get_ydata()
    assert np.any(residual < 0) and np.any(residual > 0)
    assert axes[1].get_ylabel() == r'$\Delta\sigma_w^2$'
    assert [[text.get_text() for text in ax.texts if text.get_text().startswith('(')] for ax in axes] == [
        ['(a)'], ['(b)']
    ]
    legend = axes[0].get_legend()
    assert legend is not None and legend.get_frame_on() is False
    assert [text.get_text() for text in legend.get_texts()] == [
        'Original', 'Convective boundary', 'Mirrored', 'Residual depth'
    ]
    assert axes[1].get_legend() is None
    variance_display = np.concatenate((axes[0].lines[2].get_ydata(), axes[0].lines[3].get_ydata()))
    assert axes[0].get_ylim()[1]-np.max(variance_display) < 0.1*np.ptp(variance_display)
    assert axes[1].get_ylim()[1]-np.max(residual) >= 0.54*np.ptp(residual)
    np.testing.assert_array_equal(variance, original)
    plt.close(fig)


def test_ke_depth_plot_normalizes_display_but_preserves_fitted_depths_and_input():
    z = depth_grid()
    delta = 0.25
    base = np.exp(-0.5*(z-1.0)**2/delta**2)
    kinetic_energy = np.stack((0.8*base, base, 1.2*base))
    original = kinetic_energy.copy()

    result = diag.compute_depth_from_KE_frames(1.0, z, 1.0, kinetic_energy)
    fig, d_width_t, d_width_mean, _, d_5_t, d_5_mean, _ = result
    ax = fig.axes[0]

    np.testing.assert_allclose(d_width_t, delta, rtol=1e-6)
    np.testing.assert_allclose(d_width_mean, delta, rtol=1e-6)
    expected_depth = 1.0-np.max(PchipInterpolator(z[z < 1.0], base[z < 1.0]).solve(y=0.05*np.max(base), extrapolate=False))
    np.testing.assert_allclose(d_5_t, expected_depth, rtol=1e-6)
    np.testing.assert_allclose(d_5_mean, expected_depth, rtol=1e-6)
    assert not np.isclose(d_5_mean, np.sqrt(2*np.log(20))*delta)
    assert np.nanmax(ax.lines[0].get_ydata()) <= 1.0
    assert ax.get_xlabel() == DEPTH_LABEL
    assert ax.get_ylabel() == r'$\widetilde{KE}_{\rm RZ}$'
    assert len(ax.patches) == 1
    assert {'Convective boundary', '5% depth'} <= reference_labels(ax)
    np.testing.assert_array_equal(kinetic_energy, original)
    plt.close(fig)


def test_flux_depth_plots_preserve_sign_raw_depth_results_and_inputs():
    z = depth_grid()
    kinetic_profile = np.array([0.0, 0.02, 0.2, 1.0, 0.6, 0.2, -0.1])
    convective_profile = -kinetic_profile
    kinetic_flux = np.stack((kinetic_profile, kinetic_profile))
    convective_flux = np.stack((convective_profile, convective_profile))
    kinetic_original = kinetic_flux.copy()
    convective_original = convective_flux.copy()

    fig_ke, _, depth_ke, _ = diag.compute_depth_from_KE_flux_frames(kinetic_flux, z, 1.0, 1.0)
    fig_conv, _, depth_conv, _ = diag.compute_depth_from_F_conv_frames(convective_flux, z, 1.0, 1.0)
    ax_ke, ax_conv = fig_ke.axes[0], fig_conv.axes[0]

    assert np.isclose(depth_ke, expected_flux_depth(kinetic_profile, z, 1.0, 'max'))
    assert np.isclose(depth_conv, expected_flux_depth(convective_profile, z, 1.0, 'min'))
    assert np.nanmax(ax_ke.lines[0].get_ydata()) == 1.0
    assert np.nanmin(ax_conv.lines[0].get_ydata()) == -1.0
    assert ax_ke.get_ylabel() == r'$\widetilde{F}_{\rm KE}$'
    assert ax_conv.get_ylabel() == r'$\widetilde{F}_{\rm conv}$'
    assert not has_horizontal_line(ax_ke, 0.05)
    assert not has_horizontal_line(ax_conv, -0.05)
    for ax in (ax_ke, ax_conv):
        zero = next(line for line in ax.lines if np.size(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), 0))
        assert zero.get_color() == 'black'
        assert zero.get_zorder() < ax.lines[0].get_zorder()
    assert {'Convective boundary', '5% depth'} <= reference_labels(ax_ke)
    assert {'Convective boundary', '5% depth'} <= reference_labels(ax_conv)
    assert len(ax_ke.patches) == len(ax_conv.patches) == 1
    np.testing.assert_array_equal(kinetic_flux, kinetic_original)
    np.testing.assert_array_equal(convective_flux, convective_original)
    plt.close(fig_ke)
    plt.close(fig_conv)


def test_mixing_depth_uses_frame_mean_and_variability():
    z = np.linspace(0, 2, 17)
    tracer = np.stack((z, z))
    diffusivity = np.stack((z/0.75, z/0.5))
    flux = -diffusivity

    fig, depth_mean, depth_std, profile = diag.compute_mixing_depth(flux, tracer, 1.0, z, 1.0, 1.0)
    ax = fig.axes[0]

    assert np.isclose(depth_mean, 0.375)
    assert np.isclose(depth_std, np.std([0.25, 0.5], ddof=1))
    np.testing.assert_allclose(profile, np.mean(diffusivity, axis=0))
    assert ax.get_ylabel() == r'$D_{\rm turb}/\kappa_C$'
    assert reference_labels(ax) == {'Convective boundary', 'Molecular diffusivity', 'Mixing depth'}
    plt.close(fig)

def test_combined_depth_diagnostics_preserve_individual_panel_styles():
    z = depth_grid()
    variance = np.array([0.02, 0.05, 0.18, 0.5, 1.0, 0.65, 0.2])
    delta = 0.25
    kinetic_profile = np.exp(-0.5*(z-1.0)**2/delta**2)
    kinetic_energy = np.stack((0.8*kinetic_profile, kinetic_profile, 1.2*kinetic_profile))
    flux_profile = np.array([0.0, 0.02, 0.2, 1.0, 0.6, 0.2, -0.1])
    kinetic_flux = np.stack((flux_profile, flux_profile))
    convective_flux = -kinetic_flux
    diffusivity = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 3.0, 3.0])

    table_depths = {'var': 0.45, 'var_symmetry': 0.6, 'KE': 0.3, 'KE_flux': 0.35, 'conv_flux': 0.4, 'mixing': 0.2}
    fig, axes = diag.plot_depth_diagnostics(variance, kinetic_energy, kinetic_flux, convective_flux, diffusivity, 1.0, z, 1.0, 1.0, table_depths)

    # Depth lines are drawn at the supplied (Table 1) values, not recomputed from the mean profiles.
    assert np.allclose(axes[0].lines[2].get_xdata(), table_depths['var'])
    assert np.allclose(axes[3].lines[1].get_xdata(), table_depths['KE'])
    assert np.allclose(axes[6].lines[-1].get_xdata(), table_depths['mixing'])

    assert len(axes) == 7
    assert [ax.get_ylabel() for ax in axes] == [
        r'$\sigma_w^2$', r'$\sigma_w^2$', r'$\Delta\sigma_w^2$',
        r'$\mathrm{KE}$', r'$F_{\mathrm{KE}}$',
        r'$F_{\mathrm{conv}}$', r'$D_{\rm turb}/\kappa_C$',
    ]
    assert all(ax.get_xlabel() == DEPTH_LABEL for ax in (axes[0], *axes[2:]))
    assert [[text.get_text() for text in ax.texts if text.get_text().startswith('(')] for ax in axes] == [
        ['(a)'], ['(b)'], [], ['(c)'], ['(d)'], ['(e)'], ['(f)'],
    ]
    assert axes[0].get_legend() is None
    assert [text.get_text() for text in axes[1].get_legend().get_texts()] == ['Original', 'Mirrored']
    assert axes[2].get_legend() is None
    assert axes[3].get_legend() is None
    np.testing.assert_allclose(axes[3].lines[0].get_xdata(), 1.0-z)
    assert len(axes[3].lines) == 3
    assert axes[4].get_legend() is axes[5].get_legend() is None
    assert reference_labels(axes[6]) == {'Molecular diffusivity'}
    assert axes[6].lines[1].get_color() == diag.COLOR_BOUNDARY
    assert np.allclose(axes[6].lines[1].get_xdata(), 0)
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ['Convective boundary', 'Diagnostic depth']
    assert all(not legend.get_frame_on() for legend in (axes[1].get_legend(), axes[6].get_legend(), fig.legends[0]))
    assert all(len(axes[index].patches) == 1 for index in (0, 1, 2, 3, 4, 5))
    assert len(axes[6].patches) == 0
    assert fig.get_size_inches()[0] == diag.FIG_DOUBLE
    plt.close(fig)


def test_plot_run_depth_callers_and_outputs_are_pdf():
    source = (ROOT / 'plot_run.py').read_text()
    stems = ('w_var_sym', 'w_var', 'depth_half_gaussian', 'depth_KE_flux', 'depth_F_conv', 'mixing_C1', 'depth_diagnostics')

    for stem in stems:
        assert f"_{stem}.pdf'" in source
        assert f"_{stem}.png'" in source
    assert 'plot_w_variance(w_var, z_grid, zb, L_cz, var_depth_mean, var_depth_std)' in source
    assert 'compute_depth_from_var_5pct_frames(w_var_t, z_grid, zb)' in source
    assert 'compute_depth_from_KE_flux_frames(' in source
    assert "p['KE_flux_havg'], z_grid, zb, L_cz)" in source
    assert 'compute_depth_from_F_conv_frames(' in source
    assert "p['F_conv_havg'], z_grid, zb, L_cz)" in source
    assert 'compute_mixing_depth(F_C1, C1_havg, kappa_C, z_grid, zb, L_cz)' in source


def test_variance_symmetry_depth_reports_mean_and_variability():
    z = np.linspace(0, 2, 17)
    frames = np.stack((np.exp(-((z-1.35)/0.3)**2), np.exp(-((z-1.4)/0.3)**2)))
    mean, std = diag.compute_variance_symmetry_depth(frames, z, 1.0)
    assert np.isfinite(mean)
    assert np.isfinite(std)


def test_variance_5pct_depth_matches_threshold_crossing_per_frame():
    z = np.linspace(0.0, 3.0, 301)
    zb = 2.0
    frames = []
    expected = []
    for decay in (0.08, 0.12):
        profile = np.where(z >= zb, 1.0, np.exp(-0.5*((z-zb)/decay)**2))
        frames.append(profile)
        expected.append(decay*np.sqrt(2*np.log(20)))
    depths = diag.compute_depth_from_var_5pct_frames(np.array(frames), z, zb)
    np.testing.assert_allclose(depths, expected, atol=1e-3)
