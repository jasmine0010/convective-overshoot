import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import diagnostics as diag


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)
    return p.parse_args()


def interpolate_Dt(Dt, z_grid, zb, Nz=300):
    valid = (z_grid < zb) & np.isfinite(Dt)

    if np.sum(valid) < 2:
        raise ValueError('Not enough valid Dt points for 1D model')

    z_valid = z_grid[valid]
    Dt_valid = Dt[valid]

    z_1d = np.linspace(0, zb, Nz)
    Dt_1d = np.interp(z_1d, z_valid, Dt_valid, left=0, right=Dt_valid[-1])
    Dt_1d = np.maximum(Dt_1d, 0)

    return z_1d, Dt_1d


def solve_tracer_diffusion_rz(z_1d, Dt_1d, kappa_C, z_dns, time_dns, C_dns):
    dz = z_1d[1] - z_1d[0]
    K = kappa_C + Dt_1d
    K_face = 0.5*(K[:-1] + K[1:])

    C0 = np.interp(z_1d, z_dns, C_dns[0])
    C_interface_dns = np.array([np.interp(z_1d[-1], z_dns, C) for C in C_dns])

    def C_interface(t):
        return np.interp(t, time_dns, C_interface_dns)

    def rhs(t, y):
        C = np.empty(len(z_1d))
        C[:-1] = y
        C[-1] = C_interface(t)

        dCdt = np.empty(len(y))
        dCdt[0] = 2*K_face[0]*(C[1] - C[0])/dz**2
        dCdt[1:] = (K_face[1:]*(C[2:] - C[1:-1]) - K_face[:-1]*(C[1:-1] - C[:-2]))/dz**2

        return dCdt

    sol = solve_ivp(rhs, (time_dns[0], time_dns[-1]), C0[:-1], t_eval=time_dns, method='BDF', rtol=1e-6, atol=1e-8)

    C_model = np.empty((len(sol.t), len(z_1d)))
    C_model[:,:-1] = sol.y.T
    C_model[:,-1] = np.array([C_interface(t) for t in sol.t])

    return sol.t, C_model


def plot_Dt_model(z_grid, Dt, z_1d, Dt_1d, zb):
    rz = z_grid < zb

    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(z_grid[rz], Dt[rz], label='measured Dt')
    ax.plot(z_1d, Dt_1d, '--', label='Dt used by 1D model')
    ax.axvline(zb, color='r', linestyle='--')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('z')
    ax.set_ylabel('turbulent diffusivity')
    ax.legend()
    plt.tight_layout()
    plt.close(fig)

    return fig, ax


def plot_1d_tracer_prediction(time, z, C_model, C_dns):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,7))

    indices = np.unique([0, len(time)//2, len(time)-1])

    for i in indices:
        ax1.plot(z, C_dns[i], label=f'DNS t={time[i]:.0f}')
        ax1.plot(z, C_model[i], '--', label=f'1D t={time[i]:.0f}')

    rmse = np.sqrt(np.mean((C_model-C_dns)**2, axis=1))
    mass_model = np.trapezoid(C_model, z, axis=1)
    mass_dns = np.trapezoid(C_dns, z, axis=1)

    ax2.plot(time, rmse)
    ax3.plot(time, mass_dns, label='DNS')
    ax3.plot(time, mass_model, '--', label='1D diffusion')

    ax1.set_xlabel('z')
    ax1.set_ylabel('mean concentration')
    ax1.legend(fontsize=6, ncol=2)

    ax2.set_xlabel('t')
    ax2.set_ylabel('profile RMSE')

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

    with open(f'{run_folder}/info.json') as file:
        info = json.load(file)

    bg = info['background']
    release_time = info['timing']['release_time']

    z_grid = np.asarray(bg['z'])
    zb = bg['zb']
    kappa_C = bg['kappa_C']

    # Load enough field data to cover tracer release -> end
    mask_frac = min(1, 1 - release_time/info['stop_sim_time'])
    f = diag.load_field_snapshots(run_folder, tasks=['velocity', 'C1_concentration'], frac=mask_frac, back=True, step=5)
    field_arrays = diag.compute_field_arrays(f, bg['nu'])
    p = diag.load_profile_snapshots(run_folder, tasks=['C1_havg'])

    # Use common available end time
    t_end = min(f['time'][-1], p['time'][-1])

    # Training interval: 60-80% of post-release tracer evolution
    t_train_start = release_time + train_start_frac*(t_end-release_time)
    t_train_end = release_time + train_end_frac*(t_end-release_time)
    train_mask_f = (f['time'] >= t_train_start) & (f['time'] < t_train_end)

    print(f'Dt training interval: {t_train_start:.1f} - {t_train_end:.1f}')
    print(f'Dt training snapshots: {np.sum(train_mask_f)}')

    # Infer Dt using training data only
    F_C1, C1_havg = diag.compute_turbulent_flux(f['C1_concentration'][train_mask_f], field_arrays['w'][train_mask_f])
    F_C1_avg, C1_avg, dC1dz_avg, Dt_train = diag.compute_turbulent_tracer_diffusivity(F_C1, C1_havg, z_grid, zb)

    # Freeze Dt and put it on a 1D RZ grid
    z_1d, Dt_1d = interpolate_Dt(Dt_train, z_grid, zb)

    fig, _ = plot_Dt_model(z_grid, Dt_train, z_1d, Dt_1d, zb)
    fig.savefig(f'{out_folder}/{args.run_id}_p_C1_1d_Dt.png', dpi=150)

    # Test interval: 80-100%, completely excluded from Dt inference
    t_test_start = t_train_end
    test_mask_p = (p['time'] >= t_test_start) & (p['time'] <= t_end)

    time_test = p['time'][test_mask_p]
    C1_test = p['C1_havg'][test_mask_p]

    print(f'Prediction interval: {time_test[0]:.1f} - {time_test[-1]:.1f}')
    print(f'Prediction snapshots: {len(time_test)}')

    # Evolve 1D model using DNS concentration at z=zb as the upper RZ boundary
    time_1d, C1_model = solve_tracer_diffusion_rz(z_1d, Dt_1d, kappa_C, z_grid, time_test, C1_test)

    # Put DNS profiles on the same 1D grid
    C1_dns = np.array([np.interp(z_1d, z_grid, C) for C in C1_test])

    # Quantitative errors
    rmse = np.sqrt(np.mean((C1_model-C1_dns)**2, axis=1))
    mass_model = np.trapezoid(C1_model, z_1d, axis=1)
    mass_dns = np.trapezoid(C1_dns, z_1d, axis=1)
    mass_error = (mass_model-mass_dns)/mass_dns

    print(f'Mean profile RMSE: {np.mean(rmse):.6e}')
    print(f'Final profile RMSE: {rmse[-1]:.6e}')
    print(f'Final RZ tracer mass error: {mass_error[-1]:.3%}')

    fig, _ = plot_1d_tracer_prediction(time_1d, z_1d, C1_model, C1_dns)
    fig.savefig(f'{out_folder}/{args.run_id}_p_C1_1d_prediction.png', dpi=150)

    print(f'Wrote diffusion validation figures to {out_folder}/{args.run_id}_p_C1_1d_*.png')


if __name__ == '__main__':
    main()