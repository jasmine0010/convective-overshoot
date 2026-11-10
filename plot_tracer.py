import os
import argparse

import numpy as np

import diagnostics as diag
from metadata_io import load_run_metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)
    return p.parse_args()


def _release_index(concentration):
    signal_axes = tuple(range(1, concentration.ndim))
    signal = np.max(np.abs(concentration), axis=signal_axes)
    return diag.get_release_index(signal)


def main():
    args = parse_args()
    run_folder = diag.get_run_folder(args.run_id, args.mode)
    out_folder = f'analysis/{args.mode}s/figures'
    os.makedirs(out_folder, exist_ok=True)

    flow_mask_frac = 0.7
    tracer_mask_frac = 0.6

    info, normalized = load_run_metadata(f'{run_folder}/info.json')
    params = normalized['params']
    bg = normalized['background']
    timing = normalized['timing']

    z_grid = np.asarray(bg['z'])
    x_grid = np.asarray(bg['x'])
    nu, zb, kappa_C = bg['nu'], bg['zb'], bg['kappa_C']
    L_cz = params['L_cz']

    profile_tasks_available = diag.get_snapshot_tasks(run_folder, 'profile_snapshots')
    field_tasks_available = diag.get_snapshot_tasks(run_folder, 'field_snapshots')

    tracers = []
    for tracer_name in ('C1', 'C2', 'C3'):
        if f'{tracer_name}_havg' in profile_tasks_available and f'{tracer_name}_concentration' in field_tasks_available:
            tracers.append(tracer_name)
        elif tracer_name != 'C1':
            print(f'Skipping {tracer_name}: required datasets are absent')

    if 'C1' not in tracers:
        raise KeyError('C1 datasets are required')

    release_times = {
        tracer_name: float(timing.get(f'release_time_{tracer_name}') or 0.0)
        for tracer_name in tracers
    }

    stop_sim_time = float(info.get('stop_sim_time', 0.0))
    release_time_C1 = release_times['C1']
    mask_frac = min(1.0, 1.0-release_time_C1/stop_sim_time) if stop_sim_time > release_time_C1 else 1.0

    profile_tasks = [f'{name}_havg' for name in tracers]
    field_tasks = ['velocity'] + [f'{name}_concentration' for name in tracers]
    p = diag.load_profile_snapshots(run_folder, tasks=profile_tasks, frac=mask_frac, back=True)
    f = {task: [] for task in field_tasks + ['time']}
    for chunk in diag.iter_snapshot_chunks(run_folder, field_tasks, frac=mask_frac, back=True):
        keep = np.arange(len(chunk['time'])) % 10 != 5  # Omit one in ten field snapshots.
        for task in f:
            f[task].append(chunk[task][keep])
    f = {task: np.concatenate(chunks) for task, chunks in f.items()}
    field_arrays = diag.compute_field_arrays(f, nu)

    release_indices = {name: _release_index(f[f'{name}_concentration']) for name in tracers}
    released_tracers = [name for name in tracers if release_indices[name] is not None]

    for tracer_name in tracers:
        if release_indices[tracer_name] is None and tracer_name != 'C1':
            print(f'Skipping {tracer_name}: tracer was not released')

    if release_indices['C1'] is None:
        raise ValueError('C1 was not released')

    actual_release_times = {name: float(f['time'][release_indices[name]]) for name in released_tracers}
    mature_start = release_time_C1 + tracer_mask_frac*(f['time'][-1]-release_time_C1)
    t_mask_tracer_f = (f['time'] >= mature_start) & (np.arange(len(f['time'])) >= release_indices['C1'])
    if not np.any(t_mask_tracer_f):
        raise ValueError('No mature C1 outputs')

    F_C1, C1_havg = diag.compute_turbulent_flux(f['C1_concentration'][t_mask_tracer_f], field_arrays['w'][t_mask_tracer_f])
    F_C1_avg, C1_avg, dC1dz_avg, Dt_tracer_C1 = diag.compute_turbulent_tracer_diffusivity(F_C1, C1_havg, z_grid, zb)
    Dt_C1_mean, Dt_C1_std, _ = diag.compute_turbulent_tracer_diffusivity_mean_after(F_C1, C1_havg, z_grid, zb)

    fig, _ = diag.tracer_plots(z_grid, C1_avg, F_C1_avg, dC1dz_avg, Dt_tracer_C1, zb, L_cz, kappa_C)
    fig.savefig(f'{out_folder}/{args.run_id}_C1_diag.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_C1_diag.png', dpi=180)

    fig, _ = diag.plot_turbulent_tracer_diffusivity_comparison(Dt_tracer_C1, Dt_C1_mean, Dt_C1_std, z_grid, zb, L_cz, kappa_C)
    fig.savefig(f'{out_folder}/{args.run_id}_C1_dt_comparison.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_C1_dt_comparison.png', dpi=180)

    for tracer_name in released_tracers:
        release_time = actual_release_times[tracer_name]
        times = release_time + np.array([0.05, 0.25, 0.5, 1.0])*(f['time'][-1]-release_time)
        print(f'{tracer_name} profile times:', times)

        fig, _ = diag.plot_tracer_profiles_times(f['time'], f[f'{tracer_name}_concentration'], field_arrays['w'], z_grid, zb, L_cz, times, kappa_C, tracer_name)
        fig.savefig(f'{out_folder}/{args.run_id}_{tracer_name}_profiles_time.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_{tracer_name}_profiles_time.png', dpi=180)

        if tracer_name == 'C3':
            continue

        fig, _ = diag.plot_C(p, z_grid, zb, L_cz, release_time+1, tracer_name)
        fig.savefig(f'{out_folder}/{args.run_id}_{tracer_name}.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_{tracer_name}.png', dpi=180)

        post_release = np.arange(len(f['time'])) >= release_indices[tracer_name]
        if np.count_nonzero(post_release) >= 2:
            fig, _ = diag.plot_C_diff(f[f'{tracer_name}_concentration'][post_release], x_grid, z_grid, zb, L_cz, tracer_name)
            fig.savefig(f'{out_folder}/{args.run_id}_{tracer_name}_diff.png', dpi=300)
        else:
            print(f'Skipping {tracer_name} difference plot: fewer than two post-release outputs')

    fig, _ = diag.plot_Dt_time_windows(f['time'], f['C1_concentration'], field_arrays['w'], z_grid, zb, L_cz, actual_release_times['C1'], kappa_C, windows=[(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)])
    fig.savefig(f'{out_folder}/{args.run_id}_C1_Dt_time_windows.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_C1_Dt_time_windows.png', dpi=180)

    print(f'Wrote figures to {out_folder}/{args.run_id}_*.pdf and {out_folder}/{args.run_id}_*_diff.png')



if __name__ == '__main__':
    main()
