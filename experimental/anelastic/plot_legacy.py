"""Unsupported legacy anelastic analysis retained only as a repair reference.

This module is intentionally not runnable until its stale diagnostics calls and
anelastic heat-flux semantics are reviewed.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import diagnostics as shared_diagnostics
from experimental.anelastic import diagnostics as anelastic_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)
    p.add_argument('--t-mask-frac', type=float, default=0.5)
    p.add_argument('--fields', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    run_folder = anelastic_diagnostics.get_run_folder(args.run_id, args.mode)
    out_folder = PROJECT_ROOT / 'analysis' / f'{args.mode}s' / 'figures'
    out_folder.mkdir(parents=True, exist_ok=True)

    with open(f'{run_folder}/info.json') as file:
        info = json.load(file)
    params = info['params']
    bg = info['background']

    z_grid, x_grid = np.asarray(bg['z']), np.asarray(bg['x'])
    T0, rho0, p0, ds0dz = np.squeeze(np.asarray(bg['T0'])), np.squeeze(np.asarray(bg['rho0'])), np.squeeze(np.asarray(bg['p0'])), np.squeeze(np.asarray(bg['ds0dz']))
    nu, chi, zb = bg['nu'], bg['chi'], bg['zb']

    # profiles
    p = shared_diagnostics.load_profile_snapshots(run_folder, tasks=['F_conv_havg', 'KE_havg', 'KE_flux_havg', 's_havg', 'w_sq_havg', 'C1_havg', 'C1_total', 'C2_havg', 'C2_total'])
    w_rms = shared_diagnostics.compute_w_rms(p)
    w_rms_interface = shared_diagnostics.compute_w_rms_interface(w_rms, z_grid, zb)

    ds0dz_cz_ref = bg['N2_cz']
    Nu, Nu_havg, F_cond = anelastic_diagnostics.compute_Nu_anelastic(p, ds0dz_cz_ref, z_grid, z_grid >= zb, rho0, T0, chi, ds0dz)
    kappa_turb = anelastic_diagnostics.compute_kappa_turb(p, z_grid, rho0, T0, chi, ds0dz)

    fig, _ = shared_diagnostics.plot_bg_stratification(z_grid, ds0dz, zb)
    fig.savefig(f'{out_folder}/{args.run_id}_stratification.png', dpi=150)

    fig, _ = shared_diagnostics.plot_profiles(p, w_rms, z_grid, zb)
    fig.savefig(f'{out_folder}/{args.run_id}_profiles.png', dpi=150)

    fig, _ = anelastic_diagnostics.plot_kappa_turb(p, kappa_turb, z_grid, zb)
    fig.savefig(f'{out_folder}/{args.run_id}_kappa_turb.png', dpi=150)

    fig, _ = shared_diagnostics.plot_Nu(p['time'], Nu)
    fig.savefig(f'{out_folder}/{args.run_id}_Nu.png', dpi=150)

    fig, _ = shared_diagnostics.plot_w_rms_interface(p, w_rms_interface, z_grid, zb)
    fig.savefig(f'{out_folder}/{args.run_id}_w_rms_inter.png', dpi=150)

    print(f'Wrote profile figures to {out_folder}/{args.run_id}_*.png')

    if args.fields:
        f = shared_diagnostics.load_field_snapshots(run_folder, tasks=['velocity', 'entropy_perturbation', 'C1_concentration', 'C2_concentration'], frac=0.5, back=False)
        field_arrays = shared_diagnostics.compute_field_arrays(f, nu, rho0)

        ratio, KE_cz, KE_rz = shared_diagnostics.compute_KE_ratio(field_arrays['KE'], z_grid, zb)

        t_mask_f = f['time'] >= args.t_mask_frac * f['time'][-1]
        t_mask_p = p['time'] >= args.t_mask_frac * p['time'][-1]
        w_var_t = shared_diagnostics.compute_w_variance(field_arrays['w'])
        w_var = w_var_t[t_mask_f].mean(axis=0)
        KE_diss_rate = shared_diagnostics.compute_KE_dissipation_rate_field(nu, f, x_grid, z_grid, rho0).mean(axis=(1, 2))
        depth_from_var_5pct = shared_diagnostics.compute_depth_from_var_5pct_frames(w_var_t, z_grid, zb)

        fig, depth_from_KE = shared_diagnostics.compute_depth_from_KE(params['L_cz'], z_grid, zb, p['KE_havg'], t_mask_p)
        fig.savefig(f'{out_folder}/{args.run_id}_half_gaussian.png', dpi=150)
        print(depth_from_KE)
        u_rms = shared_diagnostics.compute_u_rms(f)
        derived = shared_diagnostics.compute_and_store_derived(run_folder, info, f, p, field_arrays, u_rms, w_rms, w_rms_interface, Nu, args.t_mask_frac)
        print(f'Stored derived values to {run_folder}/info.json')

        # Plots
        fig, _ = shared_diagnostics.plot_Re_KE(f['time'], field_arrays['max_Re'], field_arrays['KE_domavg'])
        fig.savefig(f'{out_folder}/{args.run_id}_Re_KE.png', dpi=150)

        fig, _ = shared_diagnostics.plot_KE_ratio(f['time'], ratio)
        fig.savefig(f'{out_folder}/{args.run_id}_ratio.png', dpi=150)

        fig, _ = shared_diagnostics.plot_KE(f['time'], KE_cz, KE_rz)
        fig.savefig(f'{out_folder}/{args.run_id}_KE_layers.png', dpi=150)

        fig, _ = shared_diagnostics.plot_KE_dissipation_rate(f['time'], KE_diss_rate)
        fig.savefig(f'{out_folder}/{args.run_id}_KE_dissipation_rate.png', dpi=150)

        fig, _ = shared_diagnostics.plot_w_variance(w_var, z_grid, zb)
        fig.savefig(f'{out_folder}/{args.run_id}_w_var.png', dpi=150)

        fig, _ = shared_diagnostics.plot_w_variance_symmetry(params['Lz'], params['L_cz'], w_var, z_grid)
        fig.savefig(f'{out_folder}/{args.run_id}_w_var_sym.png', dpi=150)

        fig, _ = shared_diagnostics.plot_C_diff(f['C1_concentration'], p, x_grid, z_grid)
        fig.savefig(f'{out_folder}/{args.run_id}_C_diff.png', dpi=150)

        fig = shared_diagnostics.plot_diagnostics(info, f, p, field_arrays, KE_cz, KE_rz, w_rms_interface, depth_from_var_5pct)
        fig.savefig(f'{out_folder}/{args.run_id}_diagnostics.png', dpi=150)

        #print(params['L_cz'] / shared_diagnostics.compute_depth_from_KE(p, z_grid, zb, t_mask_p, 0.05))

        print(f'Wrote field figures to {out_folder}/{args.run_id}_*.png')

        # Animations
        shared_diagnostics.animate_fields(
            panels=[
                {'data': f['velocity'][:, 1], 'label': 'velocity', 'clim': 'symmetric'},
                {'data': f['entropy_perturbation'], 'label': 'entropy_perturbation', 'clim': 'symmetric'}
            ],
            x=x_grid, z=z_grid, time=f['time'],
            save_path=f'{out_folder}/{args.run_id}_snapshot_vel_tem_the.mp4'
        )
        shared_diagnostics.animate_fields(
            panels=[
                {'data': field_arrays['KE'], 'label': 'kinetic energy', 'clim': 'minmax'},
                {'data': field_arrays['KE_flux'], 'label': 'kinetic energy flux', 'clim': 'minmax'}
            ],
            x=x_grid, z=z_grid, time=f['time'],
            save_path=f'{out_folder}/{args.run_id}_snapshot_ke_kf.mp4'
        )
        shared_diagnostics.animate_fields(
            panels=[
                {'data': f['C1_concentration'], 'label': 'concentration', 'clim': 'minmax'},
                #{'data': f['C2_concentration'], 'label': 'concentration', 'clim': 'minmax'}
            ],
            x=x_grid, z=z_grid, time=f['time'],
            save_path=f'{out_folder}/{args.run_id}_snapshot_c.mp4'
        )
        print(f'Wrote field figures to {out_folder}/{args.run_id}_snapshot*.mp4')


if __name__ == '__main__':
    main()