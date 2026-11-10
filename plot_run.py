import os
import argparse
import numpy as np
import diagnostics as diag
from metadata_io import load_run_metadata, write_json_atomic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)
    p.add_argument('--fields', action='store_true')
    return p.parse_args()


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
    release_time_C1 = normalized['timing']['release_time_C1']

    z_grid, x_grid, dT0dz = np.asarray(bg['z']), np.asarray(bg['x']), np.asarray(bg['dT0dz'])
    nu, kappa, zb = bg['nu'], bg['kappa'], bg['zb']
    kappa_C = bg['kappa_C']

    # Profiles.
    profile_tasks = ['F_conv_havg', 'KE_havg', 'KE_flux_havg', 'theta_havg', 'w_sq_havg']
    profile_tasks_available = diag.get_snapshot_tasks(run_folder, 'profile_snapshots')
    if args.fields:
        profile_tasks.extend(task for task in ('C1_total', 'C2_total') if task in profile_tasks_available)
    p = diag.load_profile_snapshots(run_folder, tasks=profile_tasks)
    flow_mask_p = p['time'] > p['time'][-1]*flow_mask_frac

    L_cz = params['L_cz']
    z_cz_mid = zb + L_cz/2
    f = None
    flow_mask_f = None
    u_rms = None
    tau_conv = info.get('derived', {}).get('tau_conv')
    if args.fields or tau_conv is None:
        f = diag.load_velocity_moments(run_folder, nu, x_grid, z_grid, z_cz_mid, step=20)
        flow_mask_f = f['time'] > f['time'][-1]*flow_mask_frac
        u_rms = diag.compute_u_rms_from_moments(f['speed_sq_havg'], z_grid, (zb, zb + L_cz), flow_mask_f)
        if tau_conv is None:
            lambda_E = diag.compute_lambda_E(f['w_mid'][flow_mask_f], x_grid)
            tau_conv = diag.estimate_tau_conv(lambda_E, u_rms, L_cz)

    w_rms = diag.compute_w_rms(p['w_sq_havg'])
    w_rms_interface = diag.compute_w_rms_interface(w_rms, z_grid, zb)

    dT0dz_cz_ref = diag.get_boussinesq_cz_reference_gradient(bg)
    Nu, _, _ = diag.compute_Nu(p, dT0dz_cz_ref, z_grid, kappa, z_grid >= zb)

    fig, _ = diag.plot_bg_stratification(z_grid, dT0dz, zb, L_cz, params['tw'])
    fig.savefig(f'{out_folder}/{args.run_id}_stratification.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_stratification.png', dpi=180)

    fig, _ = diag.plot_profiles(p, w_rms, z_grid, zb, L_cz)
    fig.savefig(f'{out_folder}/{args.run_id}_profiles.png', dpi=300)

    fig, _ = diag.plot_Nu(p['time'], Nu, flow_mask_p)
    fig.savefig(f'{out_folder}/{args.run_id}_Nu.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_Nu.png', dpi=180)

    fig, _ = diag.plot_w_rms_interface(p, w_rms_interface, z_grid, zb, flow_mask_p)
    fig.savefig(f'{out_folder}/{args.run_id}_w_rms_inter.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_w_rms_inter.png', dpi=180)

    fig, _ = diag.plot_heat_fluxes(z_grid, zb, L_cz, p['F_conv_havg'][flow_mask_p], p['theta_havg'][flow_mask_p], kappa, dT0dz)
    fig.savefig(f'{out_folder}/{args.run_id}_heat_fluxes.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_heat_fluxes.png', dpi=180)

    print(f'Wrote profile figures to {out_folder}/{args.run_id}_*.pdf and {args.run_id}_profiles.png')

    if args.fields:
        ratio, KE_cz, KE_rz = diag.compute_KE_ratio_from_havg(f['KE_havg'], z_grid, zb)
        KE_diss_rate = f['KE_diss_rate']

        release_idx_C1 = diag.get_release_index(p['C1_total']) if 'C1_total' in p else None
        release_idx_C2 = diag.get_release_index(p['C2_total']) if 'C2_total' in p else None

        diag.compute_and_store_derived_from_moments(run_folder, info, p, f, u_rms, w_rms, w_rms_interface, Nu, release_idx_C1, release_idx_C2, flow_mask_p, flow_mask_f)
        print(f'Stored derived values to {run_folder}/info.json')

        # Plots.
        Re_cz = f['Re_havg'][:, z_grid >= zb].mean(axis=1)
        fig, _ = diag.plot_stationarity(
            f['time'], Re_cz, p['time'], flow_mask_f, flow_mask_p, p['w_sq_havg'], z_grid, zb, L_cz)
        fig.savefig(f'{out_folder}/{args.run_id}_stationarity.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_stationarity.png', dpi=180)

        fig, _ = diag.plot_u_components_from_moments(
            f['time'], f['ux_havg'], f['w_havg'], z_grid, zb, L_cz, flow_mask_f)
        fig.savefig(f'{out_folder}/{args.run_id}_u.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_u.png', dpi=180)

        fig, _ = diag.plot_Re_KE(f['time'], f['max_Re'], f['KE_domavg'], flow_mask_f)
        fig.savefig(f'{out_folder}/{args.run_id}_Re_KE.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_Re_KE.png', dpi=180)

        fig, _ = diag.plot_Re_cz_from_moments(
            f['time'], f['Re_havg'], z_grid, zb, L_cz, flow_mask_f)
        fig.savefig(f'{out_folder}/{args.run_id}_Re_CZ.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_Re_CZ.png', dpi=180)

        fig, _ = diag.plot_KE_ratio(f['time'], ratio, flow_mask_f)
        fig.savefig(f'{out_folder}/{args.run_id}_ratio.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_ratio.png', dpi=180)

        fig, _ = diag.plot_KE(f['time'], KE_cz, KE_rz, flow_mask_f)
        fig.savefig(f'{out_folder}/{args.run_id}_KE_layers.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_KE_layers.png', dpi=180)

        fig, _ = diag.plot_KE_dissipation_rate(f['time'], KE_diss_rate, flow_mask_f)
        fig.savefig(f'{out_folder}/{args.run_id}_KE_dissipation_rate.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_KE_dissipation_rate.png', dpi=180)

        fig = diag.plot_diagnostics(
            info, f, p, f, KE_cz, KE_rz, w_rms_interface, flow_mask_f, flow_mask_p)
        fig.savefig(f'{out_folder}/{args.run_id}_diagnostics.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_diagnostics.png', dpi=180)

        # Steady-state diagnostics.
        steady_profile_tasks = ['F_conv_havg', 'KE_havg', 'KE_flux_havg', 'theta_havg', 'w_sq_havg']
        field_tasks_available = diag.get_snapshot_tasks(run_folder, 'field_snapshots')
        snapshot_tasks = ['velocity', 'theta']
        if 'C1_concentration' in field_tasks_available:
            snapshot_tasks.append('C1_concentration')
        snapshot = diag.load_snapshot_at_time(run_folder, snapshot_tasks, 1820)
        fig, _ = diag.plot_field_snapshot(
            x_grid, z_grid, zb, snapshot['time'], snapshot['velocity'], snapshot['theta'],
            snapshot.get('C1_concentration'), 'C1')
        fig.savefig(f'{out_folder}/{args.run_id}_snapshot_fields.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_snapshot_fields.png', dpi=180)

        has_C1_fields = (
            release_idx_C1 is not None
            and 'C1_concentration' in field_tasks_available
            and 'C1_havg' in profile_tasks_available
        )
        if has_C1_fields:
            steady_profile_tasks.append('C1_havg')
            f = diag.load_tracer_moments(run_folder, 'C1_concentration', nu, x_grid, z_grid, z_cz_mid, frac=1-flow_mask_frac, back=True, step=5)
        else:
            f = diag.load_velocity_moments(run_folder, nu, x_grid, z_grid, z_cz_mid, frac=1-flow_mask_frac, back=True, step=5)
        p = diag.load_profile_snapshots(run_folder, tasks=steady_profile_tasks, frac=1-flow_mask_frac, back=True)

        if has_C1_fields:
            t_steady = flow_mask_frac*f['time'][-1]
            t_tracer_start = max(t_steady, release_time_C1 + tracer_mask_frac*(f['time'][-1]-release_time_C1))
            t_mask_tracer_f = f['time'] >= t_tracer_start

        w_var_t = f['w_var']
        w_var = w_var_t.mean(axis=0)

        fig, _ = diag.plot_w_variance_symmetry(params['Lz'], L_cz, w_var, z_grid, zb)
        fig.savefig(f'{out_folder}/{args.run_id}_w_var_sym.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_w_var_sym.png', dpi=180)

        # Depths.
        var_depth_t = diag.compute_depth_from_var_5pct_frames(w_var_t, z_grid, zb)
        var_depth_mean = np.nanmean(var_depth_t)
        var_depth_std = np.nanstd(var_depth_t, ddof=1)

        print(f'variance 5% depth: {var_depth_mean:.3f} ± {var_depth_std:.3f}')

        var_symmetry_depth_mean, var_symmetry_depth_std = diag.compute_variance_symmetry_depth(w_var_t, z_grid, zb)
        print(f'variance-symmetry residual: {var_symmetry_depth_mean:.3f} ± {var_symmetry_depth_std:.3f}')

        fig, _ = diag.plot_w_variance(w_var, z_grid, zb, L_cz, var_depth_mean, var_depth_std)
        fig.savefig(f'{out_folder}/{args.run_id}_w_var.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_w_var.png', dpi=180)

        fig, _, d_width_mean, d_width_std, _, d_5_mean, d_5_std = diag.compute_depth_from_KE_frames(L_cz, z_grid, zb, p['KE_havg'])
        fig.savefig(f'{out_folder}/{args.run_id}_depth_half_gaussian.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_depth_half_gaussian.png', dpi=180)

        print(f'KE half-Gaussian width: {d_width_mean:.3f} ± {d_width_std:.3f}')
        print(f'KE 5% depth: {d_5_mean:.3f} ± {d_5_std:.3f}')

        fig, _, KE_flux_depth_5_mean, KE_flux_depth_5_std = diag.compute_depth_from_KE_flux_frames(
            p['KE_flux_havg'], z_grid, zb, L_cz)
        fig.savefig(f'{out_folder}/{args.run_id}_depth_KE_flux.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_depth_KE_flux.png', dpi=180)

        print(f'KE flux depth: {KE_flux_depth_5_mean:.3f} ± {KE_flux_depth_5_std:.3f}')

        fig, _, F_conv_depth_5_mean, F_conv_depth_5_std = diag.compute_depth_from_F_conv_frames(
            p['F_conv_havg'], z_grid, zb, L_cz)
        fig.savefig(f'{out_folder}/{args.run_id}_depth_F_conv.pdf')
        fig.savefig(f'{out_folder}/{args.run_id}_depth_F_conv.png', dpi=180)

        print(f'convective flux: {F_conv_depth_5_mean:.3f} ± {F_conv_depth_5_std:.3f}')

        # C1 remains the material-mixing diagnostic used to calibrate the 1D model.
        d_mix_1 = np.nan
        if has_C1_fields:
            F_C1 = f['tracer_flux'][t_mask_tracer_f]
            C1_havg = f['tracer_havg'][t_mask_tracer_f]
            fig, d_mix_1, d_mix_1_std, Dt_tracer_C1 = diag.compute_mixing_depth(F_C1, C1_havg, kappa_C, z_grid, zb, L_cz)
            fig.savefig(f'{out_folder}/{args.run_id}_mixing_C1.pdf')
            fig.savefig(f'{out_folder}/{args.run_id}_mixing_C1.png', dpi=180)

            table_depths = {
                'var': var_depth_mean,
                'var_symmetry': var_symmetry_depth_mean,
                'KE': d_5_mean,
                'KE_flux': KE_flux_depth_5_mean,
                'conv_flux': F_conv_depth_5_mean,
                'mixing': d_mix_1,
            }
            fig, _ = diag.plot_depth_diagnostics(w_var, p['KE_havg'], p['KE_flux_havg'], p['F_conv_havg'], Dt_tracer_C1, kappa_C, z_grid, zb, L_cz, table_depths)
            fig.savefig(f'{out_folder}/{args.run_id}_depth_diagnostics.pdf')
            fig.savefig(f'{out_folder}/{args.run_id}_depth_diagnostics.png', dpi=180)

            if np.isfinite(d_mix_1):
                print(f'C1 mixing depth: {d_mix_1:.3f} ± {d_mix_1_std:.3f}')
            else:
                print('C1 mixing depth: unavailable')
        else:
            print('C1 mixing depth: unavailable (C1 data was absent or unreleased)')


        info['depths'] = {
            'var_5pct': float(var_depth_mean),
            'var_symmetry': float(var_symmetry_depth_mean),
            'KE': float(d_width_mean),
            'KE_5pct': float(d_5_mean),
            'KE_flux': float(KE_flux_depth_5_mean),
            'conv_flux': float(F_conv_depth_5_mean),
            'mixing_C1': float(d_mix_1)
        }

        info = diag.clean_nans(info)

        write_json_atomic(f'{run_folder}/info.json', info)

        animation = diag.load_field_snapshots(run_folder, tasks=['velocity', 'theta'], back=False, step=5)
        temperature = animation['theta'] + np.asarray(bg['T0'])
        diag.animate_fields([
            {'data': animation['velocity'][:, 1], 'label': 'Vertical velocity', 'clim': 'symmetric'},
            {'data': temperature, 'label': 'Temperature', 'clim': 'minmax'},
            {'data': animation['theta'], 'label': 'Temperature perturbation', 'clim': 'symmetric'}
        ], x_grid, z_grid, animation['time'], f'{out_folder}/{args.run_id}_fields.mp4')

        print(f'Wrote field figures to {out_folder}/{args.run_id}_*.pdf and {args.run_id}_fields.mp4')


if __name__ == '__main__':
    main()
