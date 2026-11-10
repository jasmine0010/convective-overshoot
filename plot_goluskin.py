import os
import json
import argparse
import numpy as np
import diagnostics as diag
from metadata_io import write_json_atomic


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
    
    z_grid, x_grid, dT0dz = np.asarray(bg['z']), np.asarray(bg['x']), np.asarray(bg['dTdz'])
    nu, kappa = bg['nu'], bg['kappa']

    # scalars
    s = diag.load_scalar_snapshots(run_folder, tasks=['KE', 'w_sq', 'wT', 'T'])
    t_mask_s = s['time'] >= args.t_mask_frac * s['time'][-1]
    wT = np.mean(s['wT'][t_mask_s]) * np.sqrt(params['Ra']*params['Pr'])
    T = np.mean(s['T'][t_mask_s])
    
    info['derived'] = {
        'wT': float(wT),
        'T': float(T)
    }
    write_json_atomic(f'{run_folder}/info.json', info)

    fig, _ = diag.plot_wT(s)
    fig.savefig(f'{out_folder}/{args.run_id}_wT_T.png', dpi=150)
    
    fig, _ = diag.plot_scalar_time(s)
    fig.savefig(f'{out_folder}/{args.run_id}_scalars.png', dpi=150)

if __name__ == '__main__':
    main()