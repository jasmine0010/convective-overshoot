import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import diagnostics as diag


ROOT = Path(__file__).resolve().parents[1]
DEPTH_LABEL = 'Depth'


def tracer_fields():
    time = np.linspace(0.0, 8.0, 9)
    x = np.linspace(0.0, 1.0, 4)
    z = np.linspace(0.0, 1.5, 7)
    tt = time[:, None, None]
    xx = x[None, :, None]
    zz = z[None, None, :]
    concentration = (1+0.05*tt)*(zz-0.6)**2 + 0.08*xx*zz + 0*tt*xx
    velocity = (xx-0.5)*(1+0.1*tt)*(1+0.2*zz)
    return time, x, z, concentration, velocity


def has_horizontal_line(ax, value):
    return any(np.size(line.get_ydata()) == 2 and np.allclose(line.get_ydata(), value) for line in ax.lines)


def panel_texts(axes):
    return [[text.get_text() for text in ax.texts if text.get_text().startswith('(')] for ax in np.ravel(axes)]


def test_tracer_diagnostic_panels_normalize_depth_diffusivity_and_preserve_inputs():
    z = np.linspace(0.0, 1.5, 7)
    arrays = [np.linspace(0.1, 0.7, 7), np.linspace(-0.2, 0.3, 7), np.linspace(1.0, -1.0, 7), np.linspace(-0.4, 2.0, 7)]
    originals = [array.copy() for array in arrays]

    fig, axes = diag.tracer_plots(z, *arrays, 1.0, 1.0, 0.2)
    flat = np.ravel(axes)

    for ax in flat:
        np.testing.assert_allclose(ax.lines[0].get_xdata(), 1.0-z)
        assert len(ax.patches) == 1
        assert ax.get_xlabel() == DEPTH_LABEL
        assert ax.get_title() == ''
    np.testing.assert_allclose(flat[3].lines[0].get_ydata(), arrays[3]/0.2)
    assert has_horizontal_line(flat[3], 1.0)
    assert flat[0].get_ylabel() == r'$C$'
    assert flat[1].get_ylabel() == r'$F_C$'
    assert flat[2].get_ylabel() == r'$\partial C/\partial z$'
    assert flat[3].get_ylabel() == r'$D_{\rm turb}/\kappa_C$'
    assert panel_texts(flat) == [['(a)'], ['(b)'], ['(c)'], ['(d)']]
    assert fig._suptitle is None
    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)
    plt.close(fig)


def test_diffusivity_comparison_uses_physical_scale_threshold_and_external_legend():
    z = np.linspace(0.0, 1.5, 7)
    pointwise = np.linspace(-0.2, 1.0, 7)
    mean = np.linspace(0.1, 1.3, 7)
    std = np.full(7, 0.1)
    originals = [array.copy() for array in (pointwise, mean, std)]

    fig, ax = diag.plot_turbulent_tracer_diffusivity_comparison(pointwise, mean, std, z, 1.0, 1.0, 0.2)

    np.testing.assert_allclose(ax.lines[0].get_xdata(), 1.0-z)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), mean/0.2)
    np.testing.assert_allclose(ax.lines[1].get_ydata(), pointwise/0.2)
    assert has_horizontal_line(ax, 1.0)
    assert {'Convective boundary', 'Molecular diffusivity'} <= {text.get_text() for text in ax.get_legend().get_texts()}
    assert len(ax.patches) == 1
    assert len(fig.legends) == 1 and fig.legends[0].get_frame_on() is False
    assert ax.get_title() == '' and fig._suptitle is None
    for array, original in zip((pointwise, mean, std), originals):
        np.testing.assert_array_equal(array, original)
    plt.close(fig)


def test_tracer_time_profiles_use_simulation_time_progressive_colors_and_shared_legend():
    time, _, z, concentration, velocity = tracer_fields()
    concentration_original = concentration.copy()
    velocity_original = velocity.copy()
    times = np.array([0.4, 2.0, 4.0, 8.0])

    fig, axes = diag.plot_tracer_profiles_times(time, concentration, velocity, z, 1.0, 1.0, times, 0.5, 'C1')

    for ax in axes:
        for line in ax.lines[:4]:
            np.testing.assert_allclose(line.get_xdata(), 1.0-z)
        assert len(ax.patches) == 1
        assert ax.get_title() == ''
    colors = np.array([mcolors.to_rgb(line.get_color()) for line in axes[0].lines[:4]])
    assert np.all(np.diff(colors.sum(axis=1)) < 0)
    assert len({line.get_linestyle() for line in axes[0].lines[:4]}) == 4
    assert all(label.get_text().startswith('$t=') for label in fig.legends[0].get_texts())
    assert axes[-1].get_xlabel() == DEPTH_LABEL
    assert has_horizontal_line(axes[-1], 1.0)
    assert panel_texts(axes) == [['(a)'], ['(b)'], ['(c)']]
    assert len(fig.legends) == 1 and fig.legends[0].get_frame_on() is False
    assert fig._suptitle is None
    np.testing.assert_array_equal(concentration, concentration_original)
    np.testing.assert_array_equal(velocity, velocity_original)
    plt.close(fig)


def test_single_profile_uses_tracer_colors_simulation_time_and_no_title():
    time, _, z, concentration, _ = tracer_fields()
    p = {'time': time, 'C2_havg': concentration.mean(axis=1)}
    original = p['C2_havg'].copy()

    fig, axes = diag.plot_C(p, z, 1.0, 1.0, 3.1, 'C2')

    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), 1.0-z)
    assert mcolors.to_hex(axes[0].lines[0].get_color()).upper() == diag.COLOR_C2
    assert any(text.get_text().startswith('$t=') for text in axes[0].texts)
    assert axes[-1].get_xlabel() == DEPTH_LABEL
    assert all(len(ax.patches) == 1 for ax in axes)
    assert all(ax.get_title() == '' for ax in axes)
    assert fig._suptitle is None
    np.testing.assert_array_equal(p['C2_havg'], original)
    plt.close(fig)


def test_concentration_difference_uses_normalized_downward_raster_and_physical_labels():
    _, x, z, concentration, _ = tracer_fields()
    original = concentration.copy()

    fig, (field_ax, profile_ax) = diag.plot_C_diff(concentration, x, z, 1.0, 1.0, 'C3')

    np.testing.assert_allclose(profile_ax.lines[0].get_xdata(), 1.0-z)
    assert field_ax.yaxis_inverted()
    assert field_ax.get_ylabel() == DEPTH_LABEL
    assert profile_ax.get_xlabel() == DEPTH_LABEL
    assert len(profile_ax.patches) == 1
    main_axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>'][:4]
    assert panel_texts(main_axes) == [['(a)'], ['(b)'], ['(c)'], ['(d)']]
    colorbar_labels = {ax.get_ylabel() for ax in fig.axes if ax.get_label() == '<colorbar>'}
    assert {r'$\Delta C$', r'$C$'} <= colorbar_labels
    assert fig._suptitle is None
    np.testing.assert_array_equal(concentration, original)
    plt.close(fig)


def test_diffusivity_windows_use_normalized_depth_simulation_time_and_kappa_without_mutation():
    time, _, z, concentration, velocity = tracer_fields()
    concentration_original = concentration.copy()
    velocity_original = velocity.copy()
    windows = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    fig, ax = diag.plot_Dt_time_windows(time, concentration, velocity, z, 1.0, 1.0, 0.0, 0.5, windows)

    for line in ax.lines[:4]:
        np.testing.assert_allclose(line.get_xdata(), 1.0-z)
    assert all(text.get_text().startswith('$') and r'\leq t<' in text.get_text() for text in fig.legends[0].get_texts())
    assert len({line.get_linestyle() for line in ax.lines[:4]}) == 4
    assert has_horizontal_line(ax, 1.0)
    assert len(ax.patches) == 1
    assert len(fig.legends) == 1 and fig.legends[0].get_frame_on() is False
    assert ax.get_xlabel() == DEPTH_LABEL
    assert ax.get_title() == '' and fig._suptitle is None
    np.testing.assert_array_equal(concentration, concentration_original)
    np.testing.assert_array_equal(velocity, velocity_original)
    plt.close(fig)


def test_plot_tracer_callers_use_simulation_time_and_preserve_output_mapping():
    source = (ROOT / 'plot_tracer.py').read_text()
    diagnostics_source = (ROOT / 'diagnostics.py').read_text()
    ast.parse(source)
    assert 'constrained_layout=True' not in diagnostics_source

    pdf_stems = ('C1_diag', 'C1_dt_comparison', '{tracer_name}_profiles_time', '{tracer_name}', 'C1_Dt_time_windows')
    for stem in pdf_stems:
        assert f'_{stem}.pdf' in source
    assert "_{tracer_name}_diff.png', dpi=300" in source
    assert '_C1_diag.png' in source
    assert '_C1_dt_comparison.png' in source
    assert '_C1_Dt_time_windows.png' in source
    assert 'load_run_metadata' in source
    assert 'tau_conv' not in source
    assert 'np.array([0.05, 0.25, 0.5, 1.0])' in source
    assert 'zb, L_cz, times, kappa_C, tracer_name' in source
    assert "zb, L_cz, release_time+1, tracer_name" in source
    assert 'z_grid, zb, L_cz, tracer_name' in source
    assert "if tracer_name == 'C3':" in source
    assert source.index("_{tracer_name}_profiles_time.pdf") < source.index("if tracer_name == 'C3':")
    assert source.index("if tracer_name == 'C3':") < source.index('diag.plot_C(p,')
    assert "('C1', 'C2', 'C3')" in source
