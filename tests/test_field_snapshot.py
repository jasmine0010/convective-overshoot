import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import diagnostics as diag


ROOT = Path(__file__).resolve().parents[1]


def test_field_snapshot_uses_shared_physical_layout_and_preserves_inputs():
    x = np.linspace(0, 2, 5)
    z = np.linspace(0, 3, 7)
    xx, zz = np.meshgrid(x, z, indexing='ij')
    velocity = np.stack((np.cos(xx), np.sin(xx)*np.cos(zz)))
    theta = np.sin(xx)*np.sin(zz)
    tracer = 0.5*(1+np.tanh((zz-2)/0.1))
    originals = [array.copy() for array in (velocity, theta, tracer)]

    fig, axes = diag.plot_field_snapshot(x, z, 2.0, 1400.0, velocity, theta, tracer)

    assert len(axes) == 3
    assert [ax.get_ylabel() for ax in axes] == [r'$z$']*3
    assert axes[-1].get_xlabel() == r'$x$'
    assert all(np.isclose(ax.get_box_aspect(), 3/5) for ax in axes)
    assert all(not ax.yaxis_inverted() for ax in axes)
    assert all(any(np.allclose(line.get_ydata(), 2.0) for line in ax.lines) for ax in axes)
    header_text = {text.get_text(): text for text in axes[0].texts}
    assert set(header_text) == {'(a)', r'$t=1400.0$', 'Convective boundary'}
    assert header_text[r'$t=1400.0$'].get_position()[0] == 0
    assert header_text['Convective boundary'].get_position()[0] < 1
    assert not fig.legends
    assert [[text.get_text() for text in ax.texts if text.get_text().startswith('(')] for ax in axes] == [
        ['(a)'], ['(b)'], ['(c)']]
    for ax in axes[:2]:
        vmin, vmax = ax.collections[0].get_clim()
        assert np.isclose(vmin, -vmax)
    assert fig._suptitle is None
    for array, original in zip((velocity, theta, tracer), originals):
        np.testing.assert_array_equal(array, original)
    plt.close(fig)

    fig, axes = diag.plot_field_snapshot(x, z, 2.0, 1400.0, velocity, theta)
    assert len(axes) == 2
    assert all(not ax.yaxis_inverted() for ax in axes)
    plt.close(fig)


def test_plot_run_saves_field_snapshot_as_pdf():
    source = (ROOT / 'plot_run.py').read_text()
    ast.parse(source)
    assert 'diag.load_snapshot_at_time(run_folder, snapshot_tasks, 1300)' in source
    assert 'diag.plot_field_snapshot' in source
    assert "_snapshot_fields.pdf'" in source
    assert "_snapshot_fields.png'" in source
    assert "diag.animate_fields" in source
    assert "_fields.mp4'" in source
    assert "back=False" in source
