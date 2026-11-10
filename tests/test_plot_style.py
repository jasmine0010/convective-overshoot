import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

import diagnostics as diag


def test_manuscript_rcparams_and_constants():
    assert diag.FIG_SINGLE == 3.4
    assert diag.FIG_DOUBLE == 7.0
    assert diag.PANEL_LABEL_SIZE == 10
    assert diag.LINEWIDTH_PRIMARY == 1.6
    assert diag.LINEWIDTH_SECONDARY == 1.2
    assert diag.LINEWIDTH_TIMESERIES == 1.0
    assert diag.LINEWIDTH_REFERENCE == 0.8
    assert diag.MARKER_SIZE == 3.5
    assert diag.COLOR_SIMULATION == '#0072B2'
    assert diag.COLOR_MODEL == '#E69F00'
    assert diag.COLOR_AVERAGING_INTERVAL == '#C7D7EE'
    assert diag.AVERAGING_INTERVAL_ALPHA == 0.45
    assert diag.COLOR_CONVECTIVE_LAYER == '#DDEBDD'
    assert diag.COLOR_CONVECTIVE_LAYER != diag.COLOR_AVERAGING_INTERVAL
    assert (diag.COLOR_C1, diag.COLOR_C2, diag.COLOR_C3) == ('#0072B2', '#E69F00', '#009E73')
    assert matplotlib.rcParams['font.family'] == ['STIXGeneral']
    assert matplotlib.rcParams['mathtext.fontset'] == 'stix'
    assert matplotlib.rcParams['font.size'] == 8.5
    assert matplotlib.rcParams['xtick.labelsize'] == 8.5
    assert matplotlib.rcParams['ytick.labelsize'] == 8.5
    assert matplotlib.rcParams['axes.labelsize'] == 9.5
    assert matplotlib.rcParams['legend.fontsize'] == 8.5
    assert matplotlib.rcParams['legend.frameon'] is False
    assert matplotlib.rcParams['lines.linewidth'] == 1.2
    assert matplotlib.rcParams['lines.markersize'] == 3.5
    assert matplotlib.rcParams['savefig.bbox'] == 'tight'


def test_normalized_depth_is_downward_and_does_not_mutate_input():
    z = np.array([0.0, 0.5, 1.0, 1.5])
    z_before = z.copy()

    depth = diag.normalized_depth(z, 1.0, 0.5)
    np.testing.assert_allclose(depth, [2.0, 1.0, 0.0, -1.0])
    assert depth[0] > depth[-1]
    np.testing.assert_array_equal(z, z_before)


def test_panel_labels_use_consistent_position_and_style():
    fig, axes = plt.subplots(1, 2)
    texts = diag.add_panel_labels(axes)

    assert [text.get_text() for text in texts] == ['(a)', '(b)']
    assert all(text.get_position() == (0.02, 0.98) for text in texts)
    assert all(text.get_fontsize() == 10 for text in texts)
    assert all(text.get_fontweight() == 'normal' for text in texts)
    assert all(text.get_ha() == 'left' and text.get_va() == 'top' for text in texts)
    plt.close(fig)


def test_layer_shading_adds_expected_depth_spans():
    depth = np.array([-1.0, 0.0, 2.0])
    fig, (ax_x, ax_y) = plt.subplots(1, 2)

    x_artists = diag.shade_layers(ax_x, depth)
    y_artists = diag.shade_layers(ax_y, depth, orientation='y')

    assert len(ax_x.patches) == len(ax_y.patches) == 1
    assert all(artist in ax_x.patches for artist in x_artists)
    assert all(artist in ax_y.patches for artist in y_artists)
    assert all(artist.get_alpha() == 0.35 and artist.get_zorder() == 0 for artist in (*x_artists, *y_artists))
    assert x_artists[0].get_facecolor() == to_rgba(diag.COLOR_CONVECTIVE_LAYER, 0.35)
    assert ax_x.get_facecolor() == ax_y.get_facecolor() == to_rgba('white')
    assert ax_x.margins()[0] == 0
    assert ax_y.margins()[1] == 0
    plt.close(fig)


def test_dependent_axes_and_colorbars_use_symbolic_labels():
    root = Path(__file__).resolve().parents[1]
    for filename in ('diagnostics.py', 'diffusion_v3.py'):
        tree = ast.parse((root/filename).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == 'set_ylabel' and node.args and isinstance(node.args[0], ast.Constant):
                label = node.args[0].value
                assert label == 'Depth' or (label.startswith('$') and label.endswith('$'))
            if node.func.attr == 'colorbar':
                label_keywords = [keyword.value for keyword in node.keywords if keyword.arg == 'label']
                for label_node in label_keywords:
                    if isinstance(label_node, ast.Constant):
                        label = label_node.value
                        assert label.startswith('$') and label.endswith('$')
