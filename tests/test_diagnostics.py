import ast
import glob
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import h5py
import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import diagnostics as diagnostics_module


ROOT = Path(__file__).resolve().parents[1]


def write_field_snapshot(run_folder, index, times):
    folder = run_folder / 'field_snapshots'
    folder.mkdir(exist_ok=True)
    times = np.asarray(times, dtype=float)
    x = np.arange(3)[None, :, None]
    z = np.arange(4)[None, None, :]
    ux = 1.0 + times[:, None, None] + 0.2*x + 0.03*z
    w = -0.4 + 0.5*times[:, None, None] - 0.1*x + 0.07*z
    tracer = 0.3 + 0.2*times[:, None, None] + 0.04*x**2 + 0.05*z
    velocity = np.stack((ux, w), axis=1)

    with h5py.File(folder / f'field_snapshots_s{index}.h5', 'w') as file:
        tasks = file.create_group('tasks')
        scales = file.create_group('scales')
        time_scale = scales.create_dataset('sim_time', data=times)
        time_scale.make_scale('sim_time')
        velocity_dataset = tasks.create_dataset('velocity', data=velocity)
        tracer_dataset = tasks.create_dataset('C1_concentration', data=tracer[:, None, :, :])
        velocity_dataset.dims[0].attach_scale(time_scale)
        tracer_dataset.dims[0].attach_scale(time_scale)


def load_diagnostics_functions(*names, model=None):
    tree = ast.parse((ROOT / 'diagnostics.py').read_text())
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in {'model', 'TASKS'} for target in node.targets)
        or isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {'glob': glob, 'h5py': h5py, 'np': np, 'PchipInterpolator': PchipInterpolator, 'plt': plt}
    exec(compile(module, str(ROOT / 'diagnostics.py'), 'exec'), namespace)
    if model is not None:
        namespace['model'] = model
    return types.SimpleNamespace(**namespace)


class DiagnosticsTests(unittest.TestCase):
    def test_get_run_folder_requires_one_match(self):
        diag = load_diagnostics_functions('get_run_folder')
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs = root / 'data' / 'runs'
            runs.mkdir(parents=True)
            os.chdir(root)
            try:
                with self.assertRaisesRegex(FileNotFoundError, 'No run folder'):
                    diag.get_run_folder('missing', 'run')

                (runs / 'sample_a').mkdir()
                self.assertEqual(diag.get_run_folder('sample', 'run').replace('\\', '/'), 'data/runs/sample_a')

                (runs / 'sample_b').mkdir()
                with self.assertRaisesRegex(RuntimeError, 'Ambiguous run folder'):
                    diag.get_run_folder('sample', 'run')
            finally:
                os.chdir(original_cwd)

    def test_compute_nu_uses_perturbation_gradient_only(self):
        diag = load_diagnostics_functions('compute_Nu', model='boussinesq')
        z = np.array([0.0, 1.0, 2.0])
        theta = np.array([[0.0, 1.0, 4.0], [0.0, 2.0, 8.0]])
        p = {'theta_havg': theta, 'F_conv_havg': np.full_like(theta, 3.0)}
        Nu, Nu_havg, F_cond = diag.compute_Nu(p, -2.0, z, 0.5, np.array([False, True, True]))

        expected_cond = -0.5 * np.gradient(theta, z, axis=1)
        expected_nu_havg = p['F_conv_havg'] + expected_cond
        np.testing.assert_allclose(F_cond, expected_cond)
        np.testing.assert_allclose(Nu_havg, expected_nu_havg)
        np.testing.assert_allclose(Nu, expected_nu_havg[:, 1:].mean(axis=1))

    def test_tracer_release_indices_are_independent(self):
        diag = load_diagnostics_functions('check_mass_conservation', 'get_release_index')
        C1_total = np.array([0.0, 0.0, 2.0, 2.01])
        C2_total = np.array([0.0, 0.0, 0.0, 4.0, 4.01])

        C1_release = diag.get_release_index(C1_total)
        C2_release = diag.get_release_index(C2_total)

        self.assertEqual(C1_release, 2)
        self.assertEqual(C2_release, 3)
        self.assertTrue(diag.check_mass_conservation(C1_total, C1_release, threshold=0.01)[1])
        self.assertTrue(diag.check_mass_conservation(C2_total, C2_release, threshold=0.01)[1])
        self.assertFalse(diag.check_mass_conservation(np.array([0.0, 1.0, 1.1]), 1)[1])
        self.assertIsNone(diag.get_release_index(np.zeros(4)))

    def test_driver_stores_depth_and_uses_independent_releases(self):
        plot_run_source = (ROOT / 'plot_run.py').read_text()
        diffusion_source = (ROOT / 'diffusion_v3.py').read_text()
        convection_source = (ROOT / 'convection.py').read_text()

        self.assertNotIn('load_field_snapshots', plot_run_source)
        self.assertNotIn('load_field_snapshots', diffusion_source)
        self.assertIn('diag.load_velocity_moments', plot_run_source)
        self.assertIn('diag.load_tracer_moments', plot_run_source)
        self.assertIn('diag.compute_u_rms_from_moments', plot_run_source)
        self.assertIn('diag.load_tracer_moments', diffusion_source)
        self.assertIn("'var_5pct': float(var_depth_mean)", plot_run_source)
        self.assertIn('var_depth_mean = np.nanmean(var_depth_t)', plot_run_source)
        self.assertIn("diag.get_release_index(p['C1_total'])", plot_run_source)
        self.assertIn("diag.get_release_index(p['C2_total'])", plot_run_source)
        self.assertIn("background_profiles(z, zb, args.S, tw)", convection_source)
        self.assertIn("background_profiles(z_global, zb, args.S, tw)", convection_source)


    def test_snapshot_loader_validates_inputs_and_preserves_time_axis(self):
        diag = load_diagnostics_functions('get_h5_files', '_selected_snapshot_files', 'iter_snapshot_chunks', '_load_snapshots', 'load_profile_snapshots')
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_folder = Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, 'No profile_snapshots'):
                diag.load_profile_snapshots(run_folder, ['sample'])
            with self.assertRaisesRegex(ValueError, 'No snapshot tasks'):
                diag.load_profile_snapshots(run_folder, [])
            with self.assertRaisesRegex(ValueError, 'frac'):
                diag.load_profile_snapshots(run_folder, ['sample'], frac=0)
            with self.assertRaisesRegex(ValueError, 'frac'):
                diag.load_profile_snapshots(run_folder, ['sample'], frac=1.1)
            with self.assertRaisesRegex(ValueError, 'step'):
                diag.load_profile_snapshots(run_folder, ['sample'], step=0)

            snapshot_folder = run_folder / 'profile_snapshots'
            snapshot_folder.mkdir()
            for index, sim_time in ((1, 1.0), (2, 2.0)):
                path = snapshot_folder / f'profile_snapshots_s{index}.h5'
                with h5py.File(path, 'w') as file:
                    tasks = file.create_group('tasks')
                    scales = file.create_group('scales')
                    time_scale = scales.create_dataset('sim_time', data=np.array([sim_time]))
                    time_scale.make_scale('sim_time')
                    dataset = tasks.create_dataset('sample', data=np.array([[[[index, index + 1, index + 2]]]]))
                    dataset.dims[0].attach_scale(time_scale)

            with self.assertRaisesRegex(KeyError, 'Missing HDF5 tasks'):
                diag.load_profile_snapshots(run_folder, ['missing'])

            loaded = diag.load_profile_snapshots(run_folder, ['sample'], frac=0.1, back=True)
            self.assertEqual(loaded['sample'].shape, (1, 3))
            self.assertEqual(loaded['time'].shape, (1,))
            np.testing.assert_array_equal(loaded['sample'][0], [2, 3, 4])
            np.testing.assert_array_equal(loaded['time'], [2.0])

    def test_snapshot_iterator_yields_sorted_per_file_chunks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_folder = Path(tmp_dir)
            write_field_snapshot(run_folder, 10, [3.2])
            write_field_snapshot(run_folder, 2, [0.1, 0.4])
            write_field_snapshot(run_folder, 3, [1.2, 1.9])

            chunks = list(diagnostics_module.iter_snapshot_chunks(
                run_folder, ['velocity', 'C1_concentration']))

            self.assertEqual(len(chunks), 3)
            np.testing.assert_array_equal(chunks[0]['time'], [0.1, 0.4])
            np.testing.assert_array_equal(chunks[1]['time'], [1.2, 1.9])
            np.testing.assert_array_equal(chunks[2]['time'], [3.2])
            self.assertEqual(chunks[2]['velocity'].shape, (1, 2, 3, 4))
            self.assertEqual(chunks[2]['C1_concentration'].shape, (1, 1, 3, 4))
            self.assertEqual(chunks[2]['time'].shape, (1,))

    def test_streaming_reductions_match_full_loader(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_folder = Path(tmp_dir)
            write_field_snapshot(run_folder, 10, [3.2])
            write_field_snapshot(run_folder, 2, [0.1, 0.4])
            write_field_snapshot(run_folder, 3, [1.2, 1.9])

            full = diagnostics_module.load_field_snapshots(
                run_folder, ['velocity', 'C1_concentration'])
            x_grid = np.array([0.0, 1.0, 2.0])
            z_grid = np.array([0.0, 0.2, 0.55, 1.0])
            nu = 0.2
            scalar = diagnostics_module.load_scalar_havg(run_folder, 'C1_concentration')
            velocity = diagnostics_module.load_velocity_moments(run_folder, nu, x_grid, z_grid, 0.55)
            tracer = diagnostics_module.load_tracer_moments(run_folder, 'C1_concentration', nu, x_grid, z_grid, 0.55)

            ux = full['velocity'][:, 0]
            w = full['velocity'][:, 1]
            expected_flux, expected_tracer_havg = diagnostics_module.compute_turbulent_flux(
                full['C1_concentration'], w)
            expected_speed_sq = np.mean(ux**2 + w**2, axis=1)
            field_arrays = diagnostics_module.compute_field_arrays(full, nu)
            expected_dissipation = diagnostics_module.compute_KE_dissipation_rate_field(
                nu, full, x_grid, z_grid).mean(axis=(1, 2))

            for reduced in (scalar, velocity, tracer):
                np.testing.assert_array_equal(reduced['time'], full['time'])
            np.testing.assert_allclose(scalar['havg'], expected_tracer_havg, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['ux_havg'], np.mean(ux, axis=1), rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['w_havg'], np.mean(w, axis=1), rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['w_sq_havg'], np.mean(w**2, axis=1), rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['w_var'], diagnostics_module.compute_w_variance(w), rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['speed_sq_havg'], expected_speed_sq, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['KE_havg'], field_arrays['KE_havg'], rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['KE_domavg'], field_arrays['KE_domavg'], rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['max_Re'], field_arrays['max_Re'], rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['Re_havg'], field_arrays['Re'].mean(axis=1), rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['KE_diss_rate'], expected_dissipation, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(velocity['w_mid'], w[:, :, 2], rtol=1e-14, atol=1e-14)
            full_ratio = diagnostics_module.compute_KE_ratio(field_arrays['KE'], z_grid, 0.55)
            reduced_ratio = diagnostics_module.compute_KE_ratio_from_havg(velocity['KE_havg'], z_grid, 0.55)
            for full_value, reduced_value in zip(full_ratio, reduced_ratio):
                np.testing.assert_allclose(reduced_value, full_value, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(tracer['tracer_havg'], expected_tracer_havg, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(tracer['tracer_flux'], expected_flux, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(tracer['speed_sq_havg'], expected_speed_sq, rtol=1e-14, atol=1e-14)

            time_mask = (full['time'] > 0.35) & (full['time'] <= 1.9)
            expected_u_rms = diagnostics_module.compute_u_rms(
                full['velocity'][time_mask], z_grid, (0.15, 0.9))
            streamed_u_rms = diagnostics_module.compute_u_rms_from_moments(
                velocity['speed_sq_havg'], z_grid, (0.15, 0.9), time_mask)
            self.assertAlmostEqual(streamed_u_rms, expected_u_rms, places=14)

    def test_snapshot_iterator_preserves_frac_step_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_folder = Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, 'No field_snapshots'):
                list(diagnostics_module.iter_snapshot_chunks(run_folder, ['velocity']))
            with self.assertRaisesRegex(ValueError, 'No snapshot tasks'):
                list(diagnostics_module.iter_snapshot_chunks(run_folder, []))
            with self.assertRaisesRegex(ValueError, 'frac'):
                list(diagnostics_module.iter_snapshot_chunks(run_folder, ['velocity'], frac=0))
            with self.assertRaisesRegex(ValueError, 'step'):
                list(diagnostics_module.iter_snapshot_chunks(run_folder, ['velocity'], step=0))

            write_field_snapshot(run_folder, 10, [3.2])
            write_field_snapshot(run_folder, 2, [0.1, 0.4])
            write_field_snapshot(run_folder, 3, [1.2, 1.9])

            chunks = list(diagnostics_module.iter_snapshot_chunks(
                run_folder, ['velocity'], frac=0.67, back=False, step=2))
            full = diagnostics_module.load_field_snapshots(
                run_folder, ['velocity'], frac=0.67, back=False, step=2)
            np.testing.assert_array_equal(np.concatenate([chunk['time'] for chunk in chunks]), full['time'])
            np.testing.assert_array_equal(full['time'], [0.1, 1.2])

            tail = list(diagnostics_module.iter_snapshot_chunks(
                run_folder, ['velocity'], frac=0.1, back=True))
            self.assertEqual(len(tail), 1)
            np.testing.assert_array_equal(tail[0]['time'], [3.2])

            with self.assertRaisesRegex(KeyError, 'Missing HDF5 tasks'):
                list(diagnostics_module.iter_snapshot_chunks(run_folder, ['missing']))

    def test_ke_ratio_is_one_value_per_frame_and_finite_safe(self):
        diag = load_diagnostics_functions('compute_KE_ratio')
        KE = np.array([[[0.0, 0.0, 0.0, 0.0]], [[2.0, 2.0, 4.0, 4.0]]])
        ratio, KE_cz, KE_rz = diag.compute_KE_ratio(KE, np.array([0.0, 0.5, 1.0, 1.5]), 1.0)

        self.assertEqual(ratio.shape, (2,))
        self.assertTrue(np.isnan(ratio[0]))
        self.assertEqual(ratio[1], 0.5)
        self.assertEqual(KE_cz.shape, (2,))
        self.assertEqual(KE_rz.shape, (2,))

    def test_compute_u_rms_uses_only_bounded_layer(self):
        diag = load_diagnostics_functions('compute_u_rms')
        z = np.array([0.0, 0.4, 0.7, 1.0])
        velocity = np.zeros((2, 2, 3, z.size))
        velocity[:, 0, :, :] = 3.0
        velocity[:, 1, :, :] = 4.0
        baseline = diag.compute_u_rms(velocity, z, (0.4, 1.0))

        velocity[:, :, :, z < 0.4] = 1000.0
        changed_rz = diag.compute_u_rms(velocity, z, (0.4, 1.0))

        self.assertAlmostEqual(baseline, 5.0)
        self.assertAlmostEqual(changed_rz, baseline)

    def test_compute_u_rms_interpolates_exact_layer_bounds(self):
        diag = load_diagnostics_functions('compute_u_rms')
        z = np.array([0.0, 0.1, 0.4, 1.0])
        velocity = np.zeros((1, 2, 1, z.size))
        velocity[0, 0, 0] = np.sqrt(1.0 + z)

        result = diag.compute_u_rms(velocity, z, (0.2, 0.8))

        self.assertAlmostEqual(result, np.sqrt(1.5))

    def test_estimate_tau_conv_keeps_elliptical_roll_formula(self):
        diag = load_diagnostics_functions('estimate_tau_conv')
        result = diag.estimate_tau_conv(2.5, 0.15, 1.0)
        expected = np.pi * (2.5 + 2.0) / (4.0 * 0.15)
        self.assertAlmostEqual(result, expected)

    def test_mixing_depth_without_crossing_is_unavailable(self):
        z = np.linspace(0, 2, 5)
        tracer = np.stack((z, z))
        fig, depth, _, _ = diagnostics_module.compute_mixing_depth(np.zeros((2, 5)), tracer, 1.0, z, 1.0, 1.0)
        self.assertTrue(np.isnan(depth))
        plt.close(fig)


    def test_historical_reference_gradient_uses_unit_defaults(self):
        diag = load_diagnostics_functions('get_boussinesq_cz_reference_gradient')
        self.assertEqual(diag.get_boussinesq_cz_reference_gradient({'N2_cz': -2.0}), -2.0)
        self.assertEqual(diag.get_boussinesq_cz_reference_gradient({'N2_cz': -2.0, 'g': 4.0, 'alpha': 0.5}), -1.0)

    def test_driver_keeps_gaussian_width_and_adds_five_percent_depth(self):
        source = (ROOT / 'plot_run.py').read_text()
        self.assertIn("'KE': float(d_width_mean)", source)
        self.assertIn("'KE_5pct': float(d_5_mean)", source)
        self.assertIn(
            "None if mass_conserved_C2 is None else bool(mass_conserved_C2)",
            (ROOT / 'diagnostics.py').read_text(),
        )


    def test_derived_conservation_metadata_is_tri_state(self):
        info = {
            'params': {'L_cz': 1.0},
            'background': {'x': [0.0, 1.0], 'z': [0.0, 0.5, 1.0, 1.5], 'zb': 1.0, 'nu': 1.0},
        }
        f = {'velocity': np.zeros((2, 2, 2, 4))}
        p = {'C1_total': np.array([0.0, 2.0, 2.0]), 'C2_total': np.zeros(3)}
        field_arrays = {'w': np.ones((2, 2, 4)), 'KE_domavg': np.ones(2), 'KE': np.ones((2, 2, 4))}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(diagnostics_module, 'compute_lambda_E', return_value=1.0),
                mock.patch.object(diagnostics_module, 'estimate_tau_conv', return_value=1.0),
                mock.patch.object(
                    diagnostics_module,
                    'compute_KE_dissipation_rate_field',
                    return_value=np.ones((2, 2, 4)),
                ),
            ):
                derived = diagnostics_module.compute_and_store_derived(
                    tmp_dir,
                    info,
                    f,
                    p,
                    field_arrays,
                    1.0,
                    np.ones((2, 4)),
                    np.ones(2),
                    np.ones(2),
                    1,
                    None,
                    np.array([True, True]),
                    np.array([True, True]),
                )

        self.assertIs(type(derived['mass_conserved_C1']), bool)
        self.assertTrue(derived['mass_conserved_C1'])
        self.assertEqual(derived['max_drift_C1'], 0.0)
        self.assertIsNone(derived['mass_conserved_C2'])
        self.assertIsNone(derived['max_drift_C2'])


if __name__ == '__main__':
    unittest.main()
