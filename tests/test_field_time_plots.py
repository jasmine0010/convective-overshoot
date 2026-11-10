import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import diagnostics as diag


ROOT = Path(__file__).resolve().parents[1]
TIME_LABEL = 'Time'
DEPTH_LABEL = 'Depth'


def field_data():
    time = np.array([0.0, 2.0, 4.0, 6.0])
    z = np.array([0.0, 0.5, 1.0, 1.5])
    ux = np.arange(16, dtype=float).reshape(4, 4)/10
    w = -ux/2
    mask = np.array([False, False, True, True])
    return time, z, ux, w, mask


def panel_texts(axes):
    return [[text.get_text() for text in ax.texts if text.get_text().startswith('(')] for ax in axes]


def test_velocity_panels_use_simulation_time_and_normalized_depth_without_mutating_inputs():
    time, z, ux, w, mask = field_data()
    originals = [array.copy() for array in (time, z, ux, w, mask)]

    fig, axes = diag.plot_u_components_from_moments(time, ux, w, z, 1.0, 0.5, mask)

    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(axes[2].lines[0].get_xdata(), [2.0, 1.0, 0.0, -1.0])
    assert axes[0].get_xlabel() == axes[1].get_xlabel() == TIME_LABEL
    assert axes[0].get_ylabel() == r'$\langle u_x\rangle_{\rm CZ}$'
    assert axes[1].get_ylabel() == r'$\langle w\rangle_{\rm CZ}$'
    assert all(ax.lines[0].get_linewidth() == diag.LINEWIDTH_TIMESERIES for ax in axes)
    assert axes[2].get_xlabel() == axes[3].get_xlabel() == DEPTH_LABEL
    assert len(axes[0].patches) == len(axes[1].patches) == 1
    assert len(axes[2].patches) == len(axes[3].patches) == 1
    assert panel_texts(axes) == [['(a)'], ['(b)'], ['(c)'], ['(d)']]
    assert fig.get_size_inches()[0] == diag.FIG_DOUBLE
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ['Averaging interval']
    assert axes[0].patches[0].get_alpha() == diag.AVERAGING_INTERVAL_ALPHA
    for actual, expected in zip((time, z, ux, w, mask), originals):
        np.testing.assert_array_equal(actual, expected)
    plt.close(fig)


def test_reynolds_plots_use_simulation_time_and_existing_interval():
    time, z, ux, _, mask = field_data()
    max_re = np.array([1.0, 2.0, 3.0, 4.0])
    kinetic_energy = np.array([0.2, 0.4, 0.8, 1.0])

    fig_time, axes_time = diag.plot_Re_KE(time, max_re, kinetic_energy, mask)
    fig_mixed, axes_mixed = diag.plot_Re_cz_from_moments(time, ux+2, z, 1.0, 0.5, mask)

    np.testing.assert_allclose(axes_time[0].lines[0].get_xdata(), time)
    assert axes_time[-1].get_xlabel() == TIME_LABEL
    assert axes_time[0].get_ylabel() == r'$\mathrm{Re}_{\max}$'
    assert axes_time[1].get_ylabel() == r'$\langle KE\rangle$'
    assert all(ax.lines[0].get_linewidth() == diag.LINEWIDTH_TIMESERIES for ax in axes_time)
    assert all(len(ax.patches) == 1 for ax in axes_time)
    np.testing.assert_allclose(axes_mixed[1].lines[0].get_xdata(), [2.0, 1.0, 0.0, -1.0])
    assert axes_mixed[0].get_xlabel() == TIME_LABEL
    assert axes_mixed[0].get_ylabel() == r'$\mathrm{Re}_{\rm CZ}$'
    assert axes_mixed[1].lines[0].get_linewidth() == axes_mixed[0].lines[0].get_linewidth()
    assert axes_mixed[0].get_ylim()[1]-np.max(axes_mixed[0].lines[0].get_ydata()) >= 0.24*np.ptp(axes_mixed[0].lines[0].get_ydata())
    assert axes_mixed[1].get_xlabel() == DEPTH_LABEL
    assert len(axes_mixed[0].patches) == 1
    assert len(axes_mixed[1].patches) == 1
    assert panel_texts(axes_time) == [['(a)'], ['(b)']]
    assert all(ax.get_ylim()[1] > np.max(line.get_ydata()) for ax, line in
               zip(axes_time, (axes_time[0].lines[0], axes_time[1].lines[0])))
    assert panel_texts(axes_mixed) == [['(a)'], ['(b)']]
    assert [text.get_text() for text in fig_time.legends[0].get_texts()] == ['Averaging interval']
    assert [text.get_text() for text in fig_mixed.legends[0].get_texts()] == ['Averaging interval']
    plt.close(fig_time)
    plt.close(fig_mixed)


def test_single_time_series_use_physical_labels_shading_and_external_legend():
    time, _, _, _, mask = field_data()
    ratio = np.array([0.1, 0.2, 0.3, 0.4])
    convective = np.array([1.0, 1.2, 1.1, 1.3])
    stable = np.array([0.1, 0.15, 0.2, 0.25])

    fig_ratio, ax_ratio = diag.plot_KE_ratio(time, ratio, mask)
    fig_energy, ax_energy = diag.plot_KE(time, convective, stable, mask)
    fig_diss, ax_diss = diag.plot_KE_dissipation_rate(time, convective/10, mask)

    for ax in (ax_ratio, ax_energy, ax_diss):
        np.testing.assert_allclose(ax.lines[0].get_xdata(), time)
        assert all(line.get_linewidth() == diag.LINEWIDTH_TIMESERIES for line in ax.lines)
        assert ax.get_xlabel() == TIME_LABEL
        assert len(ax.patches) == 1
    assert ax_ratio.get_ylabel() == r'$R_{\rm KE}$'
    assert ax_diss.get_ylabel() == r'$\epsilon_{\rm KE}$'
    for ax in (ax_ratio, ax_diss):
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ['Averaging interval']
    assert len(fig_energy.legends) == 1
    assert fig_energy.legends[0].get_frame_on() is False
    assert [text.get_text() for text in fig_energy.legends[0].get_texts()] == [
        'Convective layer', 'Stable layer', 'Averaging interval'
    ]
    assert fig_energy.legends[0]._ncols == 2
    fig_energy.canvas.draw()
    legend = fig_energy.legends[0].get_window_extent()
    plot = ax_energy.get_window_extent()
    assert abs((legend.x0+legend.x1)/2-(plot.x0+plot.x1)/2) < 1
    assert 0 < legend.y0-plot.y1 < 20
    plt.close('all')



def test_stationarity_plot_combines_flow_and_vertical_motion():
    time = np.arange(6)*50.0
    z = np.array([0.0, 0.3, 0.8, 1.2, 1.5])
    mask = time >= 200
    Re_cz = np.array([0.0, 20.0, 10.0, 12.0, 11.0, 13.0])
    a, b = np.arange(1, 7), np.arange(6)**2/10
    w_sq = a[:, None]+b[:, None]*z
    originals = [array.copy() for array in (time, Re_cz, mask, z, w_sq)]

    fig, axes = diag.plot_stationarity(time, Re_cz, time, mask, mask, w_sq, z, 1.0, 0.5)

    assert len(axes) == 2
    for ax in axes:
        assert len(ax.patches) == 1
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), Re_cz)
    expected = (np.sqrt(a+1.25*b), np.sqrt(a+0.5*b), np.sqrt(a+0.75*b))
    for line, values in zip(axes[1].lines, expected):
        np.testing.assert_allclose(line.get_xdata(), time)
        np.testing.assert_allclose(line.get_ydata(), values)
    assert [ax.get_ylabel() for ax in axes] == [r'$\mathrm{Re}_{\rm CZ}$', r'$w_{\rm rms}$']
    assert axes[-1].get_xlabel() == TIME_LABEL
    assert panel_texts(axes) == [['(a)'], ['(b)']]
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        'Convective layer', 'Stable layer', 'Overall', 'Averaging interval'
    ]
    assert fig.legends[0].get_frame_on() is False
    assert axes[1].get_ylim()[0] < np.min(expected[1])
    assert axes[1].get_ylim()[1] > np.max(expected[0])
    assert all(line.get_linewidth() == diag.LINEWIDTH_TIMESERIES for ax in axes for line in ax.lines)
    regime_texts = [text for text in axes[0].texts if not text.get_text().startswith('(')]
    assert [text.get_text() for text in regime_texts] == ['Initial transient', 'Statistically\nstationary regime']
    assert all(text.get_position()[1] > 1 for text in regime_texts)  # above panel (a)
    assert axes[0].get_xlim() == (0, time[-1])
    assert axes[0].get_ylim()[0] == axes[1].get_ylim()[0] == 0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels_top = max(text.get_window_extent(renderer).y1 for text in regime_texts)
    legend_gap = fig.legends[0].get_window_extent().y0-labels_top
    assert 0 < legend_gap < 20
    for actual, expected in zip((time, Re_cz, mask, z, w_sq), originals):
        np.testing.assert_array_equal(actual, expected)
    plt.close(fig)

def test_diagnostic_summary_uses_simulation_time_and_shared_legend():
    time, z, ux, _, mask = field_data()
    f = {'time': time, 'KE_domavg': np.arange(4, dtype=float)}
    p = {'time': time, 'F_conv_havg': ux}
    info = {'background': {'z': z.tolist(), 'zb': 1.0}}
    convective = np.array([1.0, 1.2, 1.1, 1.3])
    stable = np.array([0.1, 0.15, 0.2, 0.25])

    fig = diag.plot_diagnostics(
        info, f, p, f, convective, stable, np.array([0.2, 0.3, 0.25, 0.28]), mask, mask)
    axes = fig.axes

    assert len(axes) == 4
    for ax in axes:
        np.testing.assert_allclose(ax.lines[0].get_xdata(), time)
        assert all(line.get_linewidth() == diag.LINEWIDTH_TIMESERIES for line in ax.lines)
        assert ax.get_xlabel() == TIME_LABEL
        assert len(ax.patches) == 1
    assert panel_texts(axes) == [['(a)'], ['(b)'], ['(c)'], ['(d)']]
    assert axes[2].get_ylabel() == r'$F_{\rm conv}(z_b)$'
    assert axes[3].get_ylabel() == r'$w_{\rm rms}(z_b)$'
    assert all(ax.get_ylim()[1] > np.max(ax.lines[0].get_ydata()) for ax in axes)
    assert len(fig.legends) == 1
    assert fig.legends[0].get_frame_on() is False
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        'Convective layer', 'Stable layer', 'Averaging interval'
    ]
    assert fig._suptitle is None
    plt.close(fig)


def test_plot_run_field_time_outputs_are_pdf():
    source = (ROOT / 'plot_run.py').read_text()
    stems = (
        'stationarity', 'u', 'Re_KE', 'Re_CZ', 'ratio', 'KE_layers',
        'KE_dissipation_rate', 'diagnostics',
    )

    for stem in stems:
        assert f"_{stem}.pdf'" in source
        assert f"_{stem}.png'" in source
    assert 'tau_conv, flow_mask_f' not in source
    assert 'tau_conv, flow_mask_f, flow_mask_p' not in source
