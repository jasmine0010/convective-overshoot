import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import diagnostics as diag


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)
    return p.parse_args()


def build_measured_Dt_1d(Dt, z_grid, zb, Lz, Nz=300):
    valid = np.isfinite(Dt)

    if np.sum(valid) < 2:
        raise ValueError('Not enough valid Dt points for 1D model')

    valid_idx = np.where(valid)[0]
    i0 = valid_idx[np.argmin(np.abs(z_grid[valid_idx]-zb))]

    # Keep the continuous measured-Dt region connected to zb.
    i_lo = i0
    while i_lo > 0 and valid[i_lo-1]:
        i_lo -= 1

    i_hi = i0
    while i_hi < len(z_grid)-1 and valid[i_hi+1]:
        i_hi += 1

    z_valid = z_grid[i_lo:i_hi+1]
    Dt_valid = Dt[i_lo:i_hi+1]

    if len(z_valid) < 2:
        raise ValueError('Dt region connected to zb has fewer than two points')

    if not (np.any(z_valid < zb) and np.any(z_valid > zb)):
        raise ValueError('Measured Dt is not constrained on both sides of zb')

    dz = Lz/Nz
    z_1d = (np.arange(Nz)+0.5)*dz

    # No extrapolated turbulent mixing outside the measured region.
    Dt_1d = np.zeros(Nz)

    measured = (z_1d >= z_valid[0]) & (z_1d <= z_valid[-1])
    Dt_1d[measured] = np.interp(z_1d[measured], z_valid, Dt_valid)

    # A simple diffusion model cannot use a negative turbulent diffusivity.
    n_clipped = np.sum(Dt_1d < 0)
    Dt_1d = np.maximum(Dt_1d, 0)

    return z_1d, Dt_1d, z_valid[0], z_valid[-1], n_clipped


def solve_tracer_diffusion_full(z_1d, Dt_1d, kappa_C, z_dns, time_dns, C_dns):
    if len(time_dns) < 2:
        raise ValueError('Need at least two prediction times')

    if np.any(np.diff(time_dns) <= 0):
        raise ValueError('Prediction times must be strictly increasing')

    dz = z_1d[1] - z_1d[0]

    # Total diffusivity = molecular + measured turbulent diffusivity.
    K = kappa_C + Dt_1d

    if np.any(K <= 0):
        raise ValueError('Total diffusivity must be positive')

    # Face diffusivity for conservative finite-volume fluxes.
    K_face = 2*K[:-1]*K[1:]/(K[:-1]+K[1:])

    # Only DNS information used during prediction: initial profile.
    C = np.interp(z_1d, z_dns, C_dns[0])

    C_model = np.empty((len(time_dns), len(z_1d)))
    C_model[0] = C

    # Backward-Euler diffusion solve with zero-flux physical boundaries.
    for n in range(len(time_dns)-1):
        dt = time_dns[n+1] - time_dns[n]
        a = dt*K_face/dz**2

        main = np.ones(len(z_1d))
        main[:-1] += a
        main[1:] += a

        A = diags([-a, main, -a], [-1, 0, 1], format='csc')
        C = spsolve(A, C)

        C_model[n+1] = C

    return time_dns, C_model


def region_weights(z, dz, z_min, z_max):
    left = z-dz/2
    right = z+dz/2
    return np.maximum(0, np.minimum(right, z_max)-np.maximum(left, z_min))


def plot_Dt_model(z_grid, Dt, z_1d, Dt_1d, zb, z_Dt_min, z_Dt_max):
    fig, ax = plt.subplots(figsize=(5,3))

    ax.plot(z_grid, Dt, label='measured Dt')
    ax.plot(z_1d, Dt_1d, '--', label='Dt used by 1D model')

    ax.axvline(zb, color='r', linestyle='--', label='zb')
    ax.axvline(z_Dt_min, color='black', linestyle=':', linewidth=0.8)
    ax.axvline(z_Dt_max, color='black', linestyle=':', label='measured Dt extent')
    ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xlabel('z')
    ax.set_ylabel('turbulent diffusivity')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.close(fig)

    return fig, ax


def plot_1d_tracer_prediction(time, z, C_model, C_dns, zb, Lz):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,7))

    dz = z[1]-z[0]
    rz_weights = region_weights(z, dz, 0, zb)
    full_weights = region_weights(z, dz, 0, Lz)

    indices = np.unique([0, len(time)//2, len(time)-1])

    for i in indices:
        ax1.plot(z, C_dns[i], label=f'DNS t={time[i]:.0f}')
        ax1.plot(z, C_model[i], '--', label=f'1D t={time[i]:.0f}')

    error_sq = (C_model-C_dns)**2

    rmse_full = np.sqrt(np.sum(error_sq*full_weights, axis=1)/np.sum(full_weights))
    rmse_rz = np.sqrt(np.sum(error_sq*rz_weights, axis=1)/np.sum(rz_weights))

    mass_model = np.sum(C_model*rz_weights, axis=1)
    mass_dns = np.sum(C_dns*rz_weights, axis=1)

    ax1.axvline(zb, color='r', linestyle='--')
    ax1.set_xlabel('z')
    ax1.set_ylabel('mean concentration')
    ax1.legend(fontsize=6, ncol=2)

    ax2.plot(time, rmse_full, label='full domain')
    ax2.plot(time, rmse_rz, label='RZ')
    ax2.set_xlabel('t')
    ax2.set_ylabel('profile RMSE')
    ax2.legend()

    ax3.plot(time, mass_dns, label='DNS')
    ax3.plot(time, mass_model, '--', label='1D diffusion')
    ax3.set_xlabel('t')
    ax3.set_ylabel('RZ tracer amount')
    ax3.legend()

    plt.tight_layout()
    plt.close(fig)

    return fig, (ax1, ax2, ax3)


def main():
    args = parse_args()

    run_folder = diag.get_run_folder(args.run_id, args.mode)
    out_folder = f'analysis/{args.mode}s/figures'
    os.makedirs(out_folder, exist_ok=True)

    diag.model = 'boussinesq'

    train_start_frac = 0.6
    train_end_frac = 0.8
    Nz_1d = 300

    with open(f'{run_folder}/info.json') as file:
        info = json.load(file)

    bg = info['background']
    release_time = info['timing']['release_time']

    z_grid = np.asarray(bg['z'])
    zb = bg['zb']
    kappa_C = bg['kappa_C']
    Lz = info['params']['Lz']

    if not np.all(np.diff(z_grid) > 0):
        raise ValueError('z_grid must be strictly increasing')

    # Load enough field data to contain the entire training interval.
    mask_frac = min(1, 1.1*(1-release_time/info['stop_sim_time']))
    f = diag.load_field_snapshots(run_folder, tasks=['velocity', 'C1_concentration'], frac=mask_frac, back=True, step=5)
    field_arrays = diag.compute_field_arrays(f, bg['nu'])

    # Profiles are inexpensive, so keep the full C1 profile history.
    p = diag.load_profile_snapshots(run_folder, tasks=['C1_havg'])

    # Use the latest time available in both datasets.
    t_end = min(f['time'][-1], p['time'][-1])

    # Training interval: 60-80% of post-release tracer evolution.
    t_train_start = release_time + train_start_frac*(t_end-release_time)
    t_train_end = release_time + train_end_frac*(t_end-release_time)

    if f['time'][0] > t_train_start:
        raise ValueError('Loaded field data begin after the training interval; increase mask_frac')

    train_mask_f = (f['time'] >= t_train_start) & (f['time'] < t_train_end)

    if np.sum(train_mask_f) < 2:
        raise ValueError('Not enough field snapshots in Dt training interval')

    print(f'Dt training interval: {t_train_start:.1f} - {t_train_end:.1f}')
    print(f'Dt training snapshots: {np.sum(train_mask_f)}')

    # Infer Dt from the training interval only.
    F_C1, C1_havg = diag.compute_turbulent_flux(f['C1_concentration'][train_mask_f], field_arrays['w'][train_mask_f])
    F_C1_avg, C1_avg, dC1dz_avg, Dt_train = diag.compute_turbulent_tracer_diffusivity(F_C1, C1_havg, z_grid, zb)

    # Full-domain model using ONLY the measured turbulent diffusivity.
    # This includes the measured RZ and CZ portions of Dt.
    # Outside the valid measured region, turbulent Dt = 0 and only kappa_C remains.
    z_1d, Dt_1d, z_Dt_min, z_Dt_max, n_clipped = build_measured_Dt_1d(Dt_train, z_grid, zb, Lz, Nz=Nz_1d)

    print(f'Measured Dt region used: {z_Dt_min:.3f} - {z_Dt_max:.3f}')
    print(f'Negative 1D Dt cells clipped to zero: {n_clipped}')

    fig, _ = plot_Dt_model(z_grid, Dt_train, z_1d, Dt_1d, zb, z_Dt_min, z_Dt_max)
    fig.savefig(f'{out_folder}/{args.run_id}_p2_C1_1d_Dt_only_Dt.png', dpi=150)

    # Test interval: 80-100%, excluded completely from Dt inference.
    t_test_start = t_train_end
    test_mask_p = (p['time'] >= t_test_start) & (p['time'] <= t_end)

    time_test = p['time'][test_mask_p]
    C1_test = p['C1_havg'][test_mask_p]

    if len(time_test) < 2:
        raise ValueError('Not enough profile snapshots in prediction interval')

    print(f'Prediction interval: {time_test[0]:.1f} - {time_test[-1]:.1f}')
    print(f'Prediction snapshots: {len(time_test)}')

    # Autonomous full-domain prediction.
    # No future DNS value at zb is supplied.
    time_1d, C1_model = solve_tracer_diffusion_full(z_1d, Dt_1d, kappa_C, z_grid, time_test, C1_test)

    # Held-out DNS profiles are used only after integration for comparison.
    C1_dns = np.array([np.interp(z_1d, z_grid, C) for C in C1_test])

    dz_1d = z_1d[1]-z_1d[0]
    rz_weights = region_weights(z_1d, dz_1d, 0, zb)
    full_weights = region_weights(z_1d, dz_1d, 0, Lz)

    error_sq = (C1_model-C1_dns)**2

    rmse_full = np.sqrt(np.sum(error_sq*full_weights, axis=1)/np.sum(full_weights))
    rmse_rz = np.sqrt(np.sum(error_sq*rz_weights, axis=1)/np.sum(rz_weights))

    mass_model_rz = np.sum(C1_model*rz_weights, axis=1)
    mass_dns_rz = np.sum(C1_dns*rz_weights, axis=1)
    mass_error_rz = (mass_model_rz-mass_dns_rz)/mass_dns_rz

    total_mass_model = np.sum(C1_model*full_weights, axis=1)
    total_mass_dns = np.sum(C1_dns*full_weights, axis=1)

    model_mass_drift = np.max(np.abs(total_mass_model-total_mass_model[0])/np.abs(total_mass_model[0]))
    dns_mass_drift = np.max(np.abs(total_mass_dns-total_mass_dns[0])/np.abs(total_mass_dns[0]))

    print(f'Mean full-domain RMSE: {np.mean(rmse_full):.6e}')
    print(f'Final full-domain RMSE: {rmse_full[-1]:.6e}')
    print(f'Mean RZ RMSE: {np.mean(rmse_rz):.6e}')
    print(f'Final RZ RMSE: {rmse_rz[-1]:.6e}')
    print(f'Final RZ tracer mass error: {mass_error_rz[-1]:.3%}')
    print(f'1D total tracer mass drift: {model_mass_drift:.3e}')
    print(f'Interpolated DNS total tracer mass drift: {dns_mass_drift:.3e}')

    fig, _ = plot_1d_tracer_prediction(time_1d, z_1d, C1_model, C1_dns, zb, Lz)
    fig.savefig(f'{out_folder}/{args.run_id}_p2_C1_1d_Dt_only_prediction.png', dpi=150)

    print(f'Wrote measured-Dt-only validation figures to {out_folder}/{args.run_id}_p_C1_1d_Dt_only_*.png')


if __name__ == '__main__':
    main()