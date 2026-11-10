import os
import json
import argparse
import numpy as np
import diagnostics as diag


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)
    p.add_argument('--t-mask-frac', type=float, default=0.5)
    p.add_argument('--fields', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    run_folder = diag.get_run_folder(args.run_id, args.mode)
    out_folder = f'analysis/{args.mode}s/figures'
    os.makedirs(out_folder, exist_ok=True)
    
    with open(f'{run_folder}/info.json') as file:
        info = json.load(file)
    params = info['params']
    bg = info['background']
    
    z_grid, x_grid, dT0dz = np.asarray(bg['z']), np.asarray(bg['x']), np.asarray(bg['dT0dz'])
    nu, kappa, zb = bg['nu'], bg['kappa'], bg['zb']

    # scalars
    s = diag.load_scalar_snapshots(run_folder, tasks=['KE', 'w_sq'])
    fig, _ = diag.plot_scalar_time(s)
    fig.savefig(f'{out_folder}/{args.run_id}_scalars.png', dpi=150)
    
    # profiles
    p = diag.load_profile_snapshots(run_folder, tasks=['F_conv_havg', 'KE_havg', 'KE_flux_havg', 'theta_havg', 'w_sq_havg'])
    w_rms = diag.compute_w_rms(p['w_sq_havg'])
    w_rms_interface = diag.compute_w_rms_interface(w_rms, z_grid, zb)

    fig, _ = diag.plot_bg_stratification(z_grid, dT0dz, zb)
    fig.savefig(f'{out_folder}/{args.run_id}_stratification.png', dpi=150)
 
    fig, _ = diag.plot_profiles(p, w_rms, z_grid, zb)
    fig.savefig(f'{out_folder}/{args.run_id}_profiles.png', dpi=150)

    print(f'Wrote profile figures to {out_folder}/{args.run_id}_*.png')

    if args.fields:
        f = diag.load_field_snapshots(run_folder, tasks=['velocity', 'temperature_perturbation'], step=2)
        
        field_arrays = diag.compute_field_arrays(f, nu)
        ratio, KE_cz, KE_rz = diag.compute_KE_ratio(field_arrays['KE'], z_grid, zb)
        
        t_mask_f = f['time'] >= args.t_mask_frac * f['time'][-1]
        w_var = diag.compute_w_variance(field_arrays['w'])[t_mask_f].mean(axis=0)
        
        # Animations
        #f = diag.load_field_snapshots(run_folder, tasks=['velocity', 'theta'], frac=0.4, back=True, step=2)
        #field_arrays = diag.compute_field_arrays(f, nu)

        temperature = f['temperature_perturbation'] + bg['T0']
        
        diag.animate_fields(
            panels=[
                {'data': f['velocity'][:, 1], 'label': 'velocity', 'clim': 'symmetric'},
                {'data': temperature, 'label': 'temperature', 'clim': 'minmax'},
                {'data': f['temperature_perturbation'], 'label': 'temperature perturbation', 'clim': 'symmetric'}
            ],
            x=x_grid, z=z_grid, time=f['time'],
            save_path=f'{out_folder}/{args.run_id}_snapshot_vel_tem_the.mp4'
        )
        diag.animate_fields(
            panels=[
                {'data': field_arrays['KE'], 'label': 'kinetic energy', 'clim': 'minmax'},
                {'data': field_arrays['KE_flux'], 'label': 'kinetic energy flux', 'clim': 'minmax'}
            ],
            x=x_grid, z=z_grid, time=f['time'],
            save_path=f'{out_folder}/{args.run_id}_snapshot_ke_kf.mp4'
        )
        print(f'Wrote field figures to {out_folder}/{args.run_id}_snapshot*.mp4')


if __name__ == '__main__':
    main()