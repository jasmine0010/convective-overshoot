import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import diagnostics as diag


ROOT = Path(__file__).resolve().parents[1]
DEPTH_LABEL = 'Depth'
TIME_LABEL = 'Time'


def synthetic_profiles():
    time = np.array([0.0, 2.0, 4.0])
    z = np.array([0.0, 0.5, 1.0, 1.5])
    values = np.arange(12, dtype=float).reshape(3, 4) + 1
    p = {'time': time, 'KE_havg': values, 'KE_flux_havg': values-6.0}
    return p, np.sqrt(values), z


def test_profile_stage_uses_normalized_coordinates_and_physical_labels():
    p, w_rms, z = synthetic_profiles()
    fig_bg, ax_bg = diag.plot_bg_stratification(z, np.array([2.0, 1.0, -1.0, -2.0]), 1.0, 0.5, 0.05)
    average_mask = np.array([False, True, True])
    fig_nu, ax_nu = diag.plot_Nu(p['time'], np.array([1.0, 1.2, 1.1]), average_mask)
    fig_w, ax_w = diag.plot_w_rms_interface(
        p, np.array([0.2, 0.3, 0.25]), z, 1.0, average_mask)

    np.testing.assert_allclose(ax_bg.lines[0].get_xdata(), [2.0, 1.0, 0.0, -1.0])
    np.testing.assert_allclose(ax_nu.lines[0].get_xdata(), [0.0, 2.0, 4.0])
    np.testing.assert_allclose(ax_w.lines[0].get_xdata(), [0.0, 2.0, 4.0])
    assert ax_bg.get_xlabel() == DEPTH_LABEL
    assert ax_bg.get_ylabel() == r'$dT_0/dz$'
    assert ax_bg.get_xlim() == (-1.0, 2.0)
    assert ax_nu.get_xlabel() == ax_w.get_xlabel() == TIME_LABEL
    assert ax_nu.get_ylabel() == r'$\mathrm{Nu}_{\rm CZ}$'
    assert ax_w.get_ylabel() == r'$w_{\rm rms}(z_b)$'
    for ax in (ax_nu, ax_w):
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ['Averaging interval']
        assert ax.get_legend().legend_handles[0].get_facecolor() == ax.patches[0].get_facecolor()
    assert all(fig._suptitle is None for fig in (fig_bg, fig_nu, fig_w))
    assert len(ax_bg.patches) == 2
    transition = ax_bg.patches[1]
    np.testing.assert_allclose([transition.get_x(), transition.get_x()+transition.get_width()], [-0.2, 0.2])
    assert transition.get_label() == 'Transition'
    assert ax_bg.get_ylim()[1] > np.max(ax_bg.lines[0].get_ydata())
    plt.close('all')


def test_profiles_use_normalized_positive_downward_depth_and_panel_labels():
    p, w_rms, z = synthetic_profiles()
    fig, axes = diag.plot_profiles(p, w_rms, z, 1.0, 0.5)

    for ax in axes:
        assert ax.get_ylabel() == DEPTH_LABEL
        assert ax.get_ylim()[0] > ax.get_ylim()[1]
    assert axes[-1].get_xlabel() == TIME_LABEL
    panel_labels = [[text.get_text() for text in ax.texts if text.get_text().startswith('(')] for ax in axes]
    assert panel_labels == [['(a)'], ['(b)'], ['(c)']]
    assert [colorbar.ax.get_ylabel() for colorbar in [image.colorbar for image in axes[0].collections[:1] + axes[1].collections[:1] + axes[2].collections[:1]]] == [
        r'$w_{\rm rms}$', r'$KE$', r'$F_{\rm KE}$'
    ]
    assert fig._suptitle is None
    plt.close(fig)


def test_heat_fluxes_use_depth_shading_panel_labels_and_external_legend():
    _, _, z = synthetic_profiles()
    F_conv = np.array([[0.1, -0.2, 0.4, 0.6], [0.2, -0.1, 0.5, 0.7]])
    theta = np.array([[0.0, 0.1, 0.2, 0.3], [0.0, 0.2, 0.4, 0.6]])
    fig, axes = diag.plot_heat_fluxes(z, 1.0, 0.5, F_conv, theta, 0.1, np.ones(4))

    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [2.0, 1.0, 0.0, -1.0])
    assert axes[-1].get_xlabel() == DEPTH_LABEL
    assert [ax.get_ylabel() for ax in axes] == [r'$F_{\rm pert}$', r'$F_{\rm full}$']
    assert [ax.texts[-1].get_text() for ax in axes] == ['(a)', '(b)']
    assert all(len(ax.patches) == 1 for ax in axes)
    assert len(fig.legends) == 1
    assert fig.legends[0].get_frame_on() is False
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        'Convective', 'Diffusive', 'Total', 'Convective boundary'
    ]
    assert fig._suptitle is None
    plt.close(fig)


def test_plot_run_profile_formats_and_tau_fallback_are_explicit():
    source = (ROOT / 'plot_run.py').read_text()

    for stem in ('stratification', 'Nu', 'w_rms_inter', 'heat_fluxes'):
        assert f"_{stem}.pdf'" in source
        assert f"_{stem}.png'" in source
    assert "_profiles.png', dpi=300" in source
    assert source.count("step=20)") == 1
    assert "tau_conv = info.get('derived', {}).get('tau_conv')" in source
    assert 'if args.fields or tau_conv is None:' in source
    assert 'diag.compute_u_rms_from_moments' in source
    assert 'diag.compute_lambda_E' in source
    assert 'diag.estimate_tau_conv' in source
    assert 'diag.plot_profiles(p, w_rms, z_grid, zb, L_cz)' in source
