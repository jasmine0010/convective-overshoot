import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
import diagnostics as diag
from metadata_io import load_run_metadata, write_json_atomic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)
    return p.parse_args()


def estimate_Dconv(field_moments, train_mask, x_grid, z_grid, zb, L_cz):
    cz = z_grid >= zb
    w_rms_cz = np.sqrt(np.mean(field_moments['w_sq_havg'][train_mask][:, cz]))
    lambda_E = diag.compute_lambda_E(field_moments['w_mid'][train_mask], x_grid)

    ell_mix = min(lambda_E, L_cz)
    D_conv = w_rms_cz*ell_mix/3

    return D_conv, w_rms_cz, lambda_E, ell_mix


def get_optional_tracer_tasks(available_tasks):
    tasks = ['C1_havg']
    for tracer in ('C2', 'C3'):
        profile_task = f'{tracer}_havg'
        total_task = f'{tracer}_total'
        if profile_task in available_tasks:
            tasks.append(profile_task)
            if total_task in available_tasks:
                tasks.append(total_task)
    return tasks


def get_optional_tracer_release(profiles, tracer, metadata_release_time):
    profile_task = f'{tracer}_havg'
    if profile_task not in profiles:
        return None

    time = np.asarray(profiles['time'])
    profile = np.asarray(profiles[profile_task]).reshape(len(time), -1)
    released = np.any(profile != 0, axis=1)
    total_task = f'{tracer}_total'
    if total_task in profiles:
        total = np.asarray(profiles[total_task]).reshape(len(time), -1)
        released |= np.any(total != 0, axis=1)

    release_indices = np.flatnonzero(released)
    if len(release_indices) == 0:
        return None
    if metadata_release_time is not None:
        return float(metadata_release_time)
    return float(time[release_indices[0]])


def build_mesa_Dmix_1d(zb, L_cz, Lz, D_conv, q0, lambda_ov, Nz=300):
    """Build a MESA-style exponential overshoot profile on a cell-centred grid."""
    if L_cz <= 0 or Lz <= 0:
        raise ValueError('L_cz and Lz must be positive')
    if not 0 < zb < Lz:
        raise ValueError('zb must be inside domain')
    if D_conv <= 0:
        raise ValueError('D_conv must be positive')
    if q0 < 0:
        raise ValueError('q0 must be non-negative')
    if lambda_ov <= 0:
        raise ValueError('lambda_ov must be positive')
    if not isinstance(Nz, int) or Nz < 2:
        raise ValueError('Nz must be an integer > 1')

    dz = Lz/Nz
    z_1d = (np.arange(Nz)+0.5)*dz
    q = (zb-z_1d)/L_cz

    Dmix_1d = np.full(Nz, D_conv, dtype=float)
    exponential = q > -q0
    Dmix_1d[exponential] = D_conv*np.exp(-(q[exponential]+q0)/lambda_ov)
    Dmix_1d = np.maximum(Dmix_1d, np.finfo(float).tiny)

    return z_1d, Dmix_1d


def mesa_overshoot_parameters(q0, lambda_ov, L_cz, H_P):
    """Convert the dimensionless fit to MESA's exponential f and f0."""
    if q0 < 0:
        raise ValueError('q0 must be non-negative')
    if lambda_ov <= 0 or L_cz <= 0 or H_P <= 0:
        raise ValueError('lambda_ov, L_cz, and H_P must be positive')

    return {'f': 2*lambda_ov*L_cz/H_P, 'f0': q0*L_cz/H_P}


def _prepare_mesa_fit_data(z_grid, C_profiles, time, Lz, Nz):
    z_grid = np.asarray(z_grid, dtype=float)
    C_profiles = np.asarray(C_profiles, dtype=float)
    time = np.asarray(time, dtype=float)

    if z_grid.ndim != 1 or len(z_grid) < 2:
        raise ValueError('z_grid must be one-dimensional')
    if not np.all(np.diff(z_grid) > 0):
        raise ValueError('z_grid must be strictly increasing')
    if C_profiles.ndim != 2 or C_profiles.shape != (len(time), len(z_grid)):
        raise ValueError('C_profiles must have shape (len(time), len(z_grid))')
    if len(time) < 2:
        raise ValueError('Need at least two tracer profiles')
    if not np.all(np.diff(time) > 0):
        raise ValueError('Fit times must be strictly increasing')
    if not np.all(np.isfinite(C_profiles)):
        raise ValueError('C_profiles must be finite')

    dz = Lz/Nz
    z_1d = (np.arange(Nz)+0.5)*dz
    C_dns_all = np.array([np.interp(z_1d, z_grid, C) for C in C_profiles])
    C0 = C_dns_all[0].copy()

    full_weights = region_weights(z_1d, dz, 0, Lz)
    cell_sqrt_weights = np.sqrt(full_weights/np.sum(full_weights))
    tracer_scale = max(float(np.ptp(C0)), float(np.max(np.abs(C0))), np.finfo(float).eps)

    return (z_1d, C0, time[1:], C_dns_all[1:], cell_sqrt_weights, tracer_scale)


def fit_mesa_diffusivity(z_grid, C_profiles, time, zb, L_cz, Lz, D_conv, kappa_C, Nz=300, initial_q0=0.05, initial_lambda=0.1, max_nfev=200, max_dt=None):
    """Fit q0 and lambda by forward-solving the full tracer time sequence."""
    if kappa_C < 0:
        raise ValueError('kappa_C must be non-negative')

    z_1d, C0, time_eval, C_dns, cell_sqrt_weights, tracer_scale = _prepare_mesa_fit_data(z_grid, C_profiles, time, Lz, Nz)
    t_start = float(np.asarray(time)[0])

    dz_q = (Lz/Nz)/L_cz
    q0_bounds = (max(1e-10, 1e-4*dz_q), 1.0)
    lambda_bounds = (max(1e-8, 0.5*dz_q), max(1.0, zb/L_cz))

    initial_q0 = np.clip(initial_q0, *q0_bounds)
    initial_lambda = np.clip(initial_lambda, *lambda_bounds)
    lower = np.log([q0_bounds[0], lambda_bounds[0]])
    upper = np.log([q0_bounds[1], lambda_bounds[1]])
    x0 = np.log([initial_q0, initial_lambda])

    def residuals(log_parameters):
        q0, lambda_ov = np.exp(log_parameters)
        _, Dmix_1d = build_mesa_Dmix_1d(zb, L_cz, Lz, D_conv, q0, lambda_ov, Nz=Nz)
        _, C_model = solve_tracer_diffusion_full(z_1d, Dmix_1d, kappa_C, time_eval, C0, t_start, max_dt=max_dt)
        residual = (C_model-C_dns)/tracer_scale
        residual *= cell_sqrt_weights[None, :]
        residual /= np.sqrt(len(time_eval))
        return residual.ravel()

    fit = least_squares(residuals, x0, bounds=(lower, upper), max_nfev=max_nfev, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    q0, lambda_ov = np.exp(fit.x)
    _, Dmix_1d = build_mesa_Dmix_1d(zb, L_cz, Lz, D_conv, q0, lambda_ov, Nz=Nz)
    _, C_model = solve_tracer_diffusion_full(z_1d, Dmix_1d, kappa_C, time_eval, C0, t_start, max_dt=max_dt)

    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    singular_threshold = (np.finfo(float).eps*singular_values[0] if len(singular_values) else np.inf)
    jacobian_scale = singular_values[0] if len(singular_values) else 0.0
    if len(singular_values) < 2 or singular_values[-1] <= singular_threshold:
        jacobian_condition = np.inf
    else:
        jacobian_condition = singular_values[0]/singular_values[-1]

    bound_distance = np.minimum(fit.x-lower, upper-fit.x)
    at_bounds = bool(np.any(bound_distance <= 1e-3*(upper-lower)))
    poorly_localized = bool(jacobian_scale <= 1e-6 or not np.isfinite(jacobian_condition) or jacobian_condition > 1e6)

    if not fit.success:
        warnings.warn(f'MESA fit did not converge: {fit.message}', RuntimeWarning)
    if at_bounds:
        warnings.warn('MESA fit reached a bound', RuntimeWarning)
    if poorly_localized:
        warnings.warn('MESA fit is poorly localized', RuntimeWarning)

    return {
        'q0': float(q0),
        'lambda_ov': float(lambda_ov),
        'D_boundary': float(D_conv*np.exp(-q0/lambda_ov)),
        'D_conv': float(D_conv),
        'z_1d': z_1d,
        'Dmix_1d': Dmix_1d,
        'C0': C0,
        't_start': t_start,
        'time_eval': time_eval,
        'C_dns': C_dns,
        'C_model': C_model,
        'cell_sqrt_weights': cell_sqrt_weights,
        'tracer_scale': float(tracer_scale),
        'objective_rms': float(np.sqrt(np.sum(fit.fun**2))),
        'success': bool(fit.success),
        'message': str(fit.message),
        'nfev': int(fit.nfev),
        'at_bounds': at_bounds,
        'poorly_localized': poorly_localized,
        'jacobian_scale': float(jacobian_scale),
        'jacobian_condition': float(jacobian_condition),
        'log_bounds': (lower, upper),
        'max_dt': (None if max_dt is None else float(max_dt)),
    }


def evaluate_mesa_objective_surface(fit_result, zb, L_cz, Lz, D_conv, kappa_C, points=9):
    if not isinstance(points, int) or points < 3:
        raise ValueError('points must be an integer >= 3')

    lower, upper = fit_result['log_bounds']
    q0_log = np.linspace(
        max(lower[0], np.log(fit_result['q0'])-np.log(4)),
        min(upper[0], np.log(fit_result['q0'])+np.log(4)),
        points,
    )
    lambda_log = np.linspace(
        max(lower[1], np.log(fit_result['lambda_ov'])-np.log(4)),
        min(upper[1], np.log(fit_result['lambda_ov'])+np.log(4)),
        points,
    )
    objective = np.empty((points, points))

    for i, log_lambda in enumerate(lambda_log):
        for j, log_q0 in enumerate(q0_log):
            _, Dmix_1d = build_mesa_Dmix_1d(
                zb,
                L_cz,
                Lz,
                D_conv,
                np.exp(log_q0),
                np.exp(log_lambda),
                Nz=len(fit_result['z_1d']),
            )
            _, C_model = solve_tracer_diffusion_full(
                fit_result['z_1d'],
                Dmix_1d,
                kappa_C,
                fit_result['time_eval'],
                fit_result['C0'],
                fit_result['t_start'],
                max_dt=fit_result['max_dt'],
            )
            residual = (C_model-fit_result['C_dns'])/fit_result['tracer_scale']
            residual *= fit_result['cell_sqrt_weights'][None, :]
            residual /= np.sqrt(len(fit_result['time_eval']))
            objective[i, j] = np.sqrt(np.sum(residual**2))

    return np.exp(q0_log), np.exp(lambda_log), objective


def solve_tracer_diffusion_full(z_1d, Dmix_1d, kappa_C, time_eval, C0, t_start, max_dt=None):
    if len(time_eval) < 1:
        raise ValueError('Need at least one prediction time')

    if np.any(time_eval <= t_start):
        raise ValueError('Prediction times must follow t_start')

    if np.any(np.diff(time_eval) <= 0):
        raise ValueError('Prediction times must be strictly increasing')

    dz = z_1d[1] - z_1d[0]
    K = kappa_C + Dmix_1d

    if np.any(K <= 0):
        raise ValueError('Total D must be positive')

    K_face = 2*K[:-1]*K[1:]/(K[:-1]+K[1:])
    C = C0.copy()

    times = np.concatenate(([t_start], time_eval))
    C_all = np.empty((len(times), len(z_1d)))
    C_all[0] = C

    solve_by_timestep = {}
    for n in range(len(times)-1):
        interval = times[n+1] - times[n]
        steps = 1 if max_dt is None else int(np.ceil(interval/max_dt))
        dt = interval/steps
        timestep_key = round(float(dt), 12)
        if timestep_key not in solve_by_timestep:
            a = dt*K_face/dz**2
            main = np.ones(len(z_1d))
            main[:-1] += a
            main[1:] += a
            A = diags([-a, main, -a], [-1, 0, 1], format='csc')
            solve_by_timestep[timestep_key] = factorized(A)
        for _ in range(steps):
            C = solve_by_timestep[timestep_key](C)
        C_all[n+1] = C

    return time_eval, C_all[1:]


def region_weights(z, dz, z_min, z_max):
    left = z - dz/2
    right = z + dz/2
    return np.maximum(0, np.minimum(right, z_max)-np.maximum(left, z_min))


def compute_prediction_errors(z, C_model, C_dns, zb, Lz, injected_mass=None):
    dz = z[1] - z[0]
    rz_weights = region_weights(z, dz, 0, zb)
    full_weights = region_weights(z, dz, 0, Lz)

    error_sq = (C_model-C_dns)**2

    rmse_full = np.sqrt(np.sum(error_sq*full_weights, axis=1)/np.sum(full_weights))
    rmse_rz = np.sqrt(np.sum(error_sq*rz_weights, axis=1)/np.sum(rz_weights))

    mass_model_rz = np.sum(C_model*rz_weights, axis=1)
    mass_dns_rz = np.sum(C_dns*rz_weights, axis=1)
    total_mass_model = np.sum(C_model*full_weights, axis=1)
    total_mass_dns = np.sum(C_dns*full_weights, axis=1)
    if injected_mass is None:
        injected_mass = total_mass_dns[0]
    mass_error_rz = (mass_model_rz-mass_dns_rz)/injected_mass

    model_mass_drift = np.max(np.abs(total_mass_model-injected_mass)/np.abs(injected_mass))
    dns_mass_drift = np.max(np.abs(total_mass_dns-injected_mass)/np.abs(injected_mass))

    return rmse_full, rmse_rz, mass_model_rz, mass_dns_rz, mass_error_rz, model_mass_drift, dns_mass_drift


def build_pointwise_Dmix_1d(Dt, z_grid, z_1d, zb):
    valid = np.isfinite(Dt)
    valid_indices = np.flatnonzero(valid)
    i0 = valid_indices[np.argmin(np.abs(z_grid[valid_indices]-zb))]
    i_lo = i0
    while i_lo > 0 and valid[i_lo-1]:
        i_lo -= 1
    i_hi = i0
    while i_hi < len(z_grid)-1 and valid[i_hi+1]:
        i_hi += 1
    Dmix = np.zeros_like(z_1d)
    measured = (z_1d >= z_grid[i_lo]) & (z_1d <= z_grid[i_hi])
    Dmix[measured] = np.interp(z_1d[measured], z_grid[i_lo:i_hi+1], Dt[i_lo:i_hi+1])
    return np.maximum(Dmix, 0)


def compare_diffusivity_models(z, models, kappa_C, time_eval, C0, t_start, C_dns, zb, Lz, max_dt=None):
    dz = z[1]-z[0]
    injected_mass = np.sum(C0*region_weights(z, dz, 0, Lz))
    comparisons = {}
    for name, Dmix in models.items():
        _, C_model = solve_tracer_diffusion_full(z, Dmix, kappa_C, time_eval, C0, t_start, max_dt=max_dt)
        errors = compute_prediction_errors(z, C_model, C_dns, zb, Lz, injected_mass)
        comparisons[name] = {'C_model': C_model, 'errors': errors}
    return comparisons


def comparison_summary(comparisons):
    return {name: prediction_error_summary(result['errors']) for name, result in comparisons.items()}


def numerical_convergence_check(z_grid, C0_dns, time_eval, C_dns_grid, t_start, zb, L_cz, Lz, D_conv, q0, lambda_ov, kappa_C, resolutions, timesteps):
    results = {}
    for Nz in resolutions:
        z, Dmix = build_mesa_Dmix_1d(zb, L_cz, Lz, D_conv, q0, lambda_ov, Nz=Nz)
        C0 = np.interp(z, z_grid, C0_dns)
        C_dns = np.array([np.interp(z, z_grid, C) for C in C_dns_grid])
        for max_dt in timesteps:
            _, C_model = solve_tracer_diffusion_full(z, Dmix, kappa_C, time_eval, C0, t_start, max_dt=max_dt)
            injected_mass = np.sum(C0)*Lz/Nz
            errors = compute_prediction_errors(z, C_model, C_dns, zb, Lz, injected_mass)
            results[f'Nz={Nz},dt={max_dt:g}'] = prediction_error_summary(errors)
    return results


def plot_mesa_Dt_model(z_grid, Dt, z_1d, Dmix_1d, zb, L_cz, D_conv, q0):
    fig, (ax, ax_signed) = plt.subplots(2, 1, figsize=(diag.FIG_SINGLE, 3.8), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
    depth = diag.normalized_depth(z_grid, zb, L_cz)
    depth_1d = diag.normalized_depth(z_1d, zb, L_cz)
    Dt_normalized = Dt/D_conv
    Dmix_normalized = Dmix_1d/D_conv

    pointwise, = ax.plot(depth, Dt_normalized, color=diag.COLOR_SIMULATION,
                         linewidth=diag.LINEWIDTH_SECONDARY, label='Pointwise flux–gradient diagnostic')
    fitted, = ax.plot(depth_1d, Dmix_normalized, color=diag.COLOR_MODEL, linestyle='--',
                      linewidth=diag.LINEWIDTH_PRIMARY, label='Fitted 1D diffusivity')
    for axis in (ax, ax_signed):
        diag.shade_layers(axis, depth)
        axis.axvline(0, color=diag.COLOR_BOUNDARY, linestyle='--', linewidth=diag.LINEWIDTH_REFERENCE, label='Convective boundary')
        axis.axhline(0, color='black', linewidth=diag.LINEWIDTH_REFERENCE, zorder=0.5)

    ax.axvline(-q0, color=diag.COLOR_THRESHOLD, linestyle=':', linewidth=diag.LINEWIDTH_REFERENCE, label='Reference point')
    ax.axhline(1, color=diag.COLOR_THRESHOLD, linestyle=':', linewidth=diag.LINEWIDTH_REFERENCE)
    ax.text(0.98, 1, r'$D_{\rm conv}$', transform=ax.get_yaxis_transform(), ha='right', va='bottom')

    positive_Dt = Dt_normalized[np.isfinite(Dt_normalized) & (Dt_normalized > 0)]
    positive_Dmix = Dmix_normalized[np.isfinite(Dmix_normalized) & (Dmix_normalized > 0)]
    positive_scale = max(np.percentile(positive_Dt, 95) if positive_Dt.size else 0,
                         np.max(positive_Dmix) if positive_Dmix.size else 0, 1)
    ax.set_ylim(-0.05*positive_scale, 1.1*positive_scale)
    ax.set_ylabel(r'$D/D_{\rm conv}$')

    ax_signed.plot(depth, Dt_normalized, color=diag.COLOR_SIMULATION, linewidth=diag.LINEWIDTH_SECONDARY)
    ax_signed.set_ylim(-positive_scale, positive_scale)
    ax_signed.set_xlabel('Depth')
    ax_signed.set_ylabel(r'$D_{\rm point}/D_{\rm conv}$')

    diag.add_panel_labels((ax, ax_signed))
    fig.legend([pointwise, fitted], [pointwise.get_label(), fitted.get_label()], loc='lower center',
               bbox_to_anchor=(0.5, 0.995), ncol=1, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    diag.center_legend_above_axes(fig, (ax, ax_signed))
    plt.close(fig)
    return fig, ax


def plot_mesa_objective_surface(q0, lambda_ov, objective, best_q0, best_lambda):
    fig, ax = plt.subplots(figsize=(diag.FIG_SINGLE, 2.8))
    log_objective = np.log10(np.maximum(objective, np.finfo(float).tiny))
    contours = ax.contourf(q0, lambda_ov, log_objective, levels=20, cmap='viridis', antialiased=False)
    ax.plot(best_q0, best_lambda, marker='x', markersize=diag.MARKER_SIZE, color='white',
            markeredgewidth=1.0, linestyle='none')
    ax.annotate('Best fit', (best_q0, best_lambda), xytext=(5, 5), textcoords='offset points',
                ha='left', va='bottom')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$q_0$')
    ax.set_ylabel(r'$\lambda/L_{\rm cz}$')
    fig.colorbar(contours, ax=ax, label=r'$\log_{10}\mathcal{E}$')
    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def _finite_float_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def prediction_error_summary(errors):
    (rmse_full, rmse_rz, _, _, mass_error_rz, model_mass_drift, dns_mass_drift) = errors

    return {
        'mean_rmse_full': _finite_float_or_none(np.mean(rmse_full)),
        'final_rmse_full': _finite_float_or_none(rmse_full[-1]),
        'mean_rmse_rz': _finite_float_or_none(np.mean(rmse_rz)),
        'final_rmse_rz': _finite_float_or_none(rmse_rz[-1]),
        'final_rz_mass_error': _finite_float_or_none(mass_error_rz[-1]),
        'mean_stable_layer_rmse': _finite_float_or_none(np.mean(rmse_rz)),
        'final_stable_layer_rmse': _finite_float_or_none(rmse_rz[-1]),
        'final_stable_layer_mass_error_normalized_by_injected_mass': _finite_float_or_none(mass_error_rz[-1]),
        'model_total_mass_drift': _finite_float_or_none(model_mass_drift),
        'dns_total_mass_drift': _finite_float_or_none(dns_mass_drift),
    }


def mesa_fit_summary(fit_result, D_conv_factor):
    condition = fit_result['jacobian_condition']
    return {
        'D_conv_factor': float(D_conv_factor),
        'D_conv': float(fit_result['D_conv']),
        'D_boundary': float(fit_result['D_boundary']),
        'q0': float(fit_result['q0']),
        'lambda_over_Lcz': float(fit_result['lambda_ov']),
        'objective_rms': float(fit_result['objective_rms']),
        'success': bool(fit_result['success']),
        'message': fit_result['message'],
        'nfev': int(fit_result['nfev']),
        'at_bounds': bool(fit_result['at_bounds']),
        'poorly_localized': bool(fit_result['poorly_localized']),
        'jacobian_condition': (float(condition) if np.isfinite(condition) else None),
        'jacobian_scale': float(fit_result['jacobian_scale']),
    }


def plot_1d_tracer_prediction(time, z, C_model, C_dns, zb, L_cz, Lz, tracer_name):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(diag.FIG_SINGLE, 6.2))
    dz = z[1] - z[0]
    rz_weights = region_weights(z, dz, 0, zb)
    full_weights = region_weights(z, dz, 0, Lz)
    indices = np.unique([0, len(time)//2, len(time)-1])
    depth = diag.normalized_depth(z, zb, L_cz)
    alphas = np.linspace(0.45, 1, len(indices))

    for i, alpha in zip(indices, alphas):
        ax1.plot(depth, C_dns[i], color=diag.COLOR_SIMULATION,
                 linewidth=diag.LINEWIDTH_TIMESERIES, alpha=alpha)
        ax1.plot(depth, C_model[i], color=diag.COLOR_MODEL, linestyle='--',
                 linewidth=diag.LINEWIDTH_TIMESERIES, alpha=alpha)

    error_sq = (C_model-C_dns)**2
    rmse_full = np.sqrt(np.sum(error_sq*full_weights, axis=1)/np.sum(full_weights))
    rmse_rz = np.sqrt(np.sum(error_sq*rz_weights, axis=1)/np.sum(rz_weights))
    mass_model = np.sum(C_model*rz_weights, axis=1)
    mass_dns = np.sum(C_dns*rz_weights, axis=1)

    diag.shade_layers(ax1, depth)
    ax1.axvline(0, color=diag.COLOR_BOUNDARY, linestyle='--', linewidth=diag.LINEWIDTH_REFERENCE, label='Convective boundary')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel(r'$C$')
    diag.add_reference_legend(ax1)

    ax2.plot(time, rmse_full, color=diag.COLOR_C1, linewidth=diag.LINEWIDTH_TIMESERIES)
    ax2.plot(time, rmse_rz, color=diag.COLOR_C3, linestyle='--',
             linewidth=diag.LINEWIDTH_TIMESERIES)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'$\mathrm{RMSE}(C)$')

    ax3.plot(time, mass_dns, color=diag.COLOR_SIMULATION, linewidth=diag.LINEWIDTH_TIMESERIES)
    ax3.plot(time, mass_model, color=diag.COLOR_MODEL, linestyle='--',
             linewidth=diag.LINEWIDTH_TIMESERIES)
    ax3.set_xlabel('Time')
    ax3.set_ylabel(r'$M_{C,\rm RZ}$')

    legend_handles = [
        Line2D([0], [0], color=diag.COLOR_SIMULATION, linewidth=diag.LINEWIDTH_TIMESERIES,
               label='2D simulation'),
        Line2D([0], [0], color=diag.COLOR_MODEL, linestyle='--', linewidth=diag.LINEWIDTH_TIMESERIES,
               label='1D diffusion model'),
        *[Line2D([0], [0], color='0.2', alpha=alpha, linewidth=diag.LINEWIDTH_TIMESERIES,
                 label=rf'$t={time[i]:.1f}$')
          for i, alpha in zip(indices, alphas)],
        Line2D([0], [0], color=diag.COLOR_C1, linewidth=diag.LINEWIDTH_TIMESERIES, label='Full domain'),
        Line2D([0], [0], color=diag.COLOR_C3, linestyle='--',
               linewidth=diag.LINEWIDTH_TIMESERIES, label='Stable layer'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2,
               frameon=False)
    diag.add_panel_labels((ax1, ax2, ax3))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    diag.center_legend_above_axes(fig, (ax1, ax2, ax3))
    plt.close(fig)
    return fig, (ax1, ax2, ax3)


def print_prediction_errors(tracer_name, rmse_full, rmse_rz, mass_error_rz, model_mass_drift, dns_mass_drift):
    print()
    print(f'{tracer_name} prediction:')
    print(f'Mean full-domain RMSE: {np.mean(rmse_full):.6e}')
    print(f'Final full-domain RMSE: {rmse_full[-1]:.6e}')
    print(f'Mean stable-layer RMSE: {np.mean(rmse_rz):.6e}')
    print(f'Final stable-layer RMSE: {rmse_rz[-1]:.6e}')
    print(f'Final stable-layer tracer-mass error / injected mass: {mass_error_rz[-1]:.3%}')
    print(f'1D total tracer mass drift: {model_mass_drift:.3e}')
    print(f'Interpolated DNS total tracer mass drift: {dns_mass_drift:.3e}')


def main():
    args = parse_args()
    run_folder = diag.get_run_folder(args.run_id, args.mode)
    out_folder = f'analysis/{args.mode}s/figures'
    os.makedirs(out_folder, exist_ok=True)


    train_start_offset = 420.5
    train_end_offset = 560.0
    Nz_1d = 300

    _, normalized = load_run_metadata(f'{run_folder}/info.json')
    bg = normalized['background']

    release_time_C1 = normalized['timing']['release_time_C1']
    metadata_release_time_C2 = normalized['timing']['release_time_C2']
    metadata_release_time_C3 = normalized['timing']['release_time_C3']

    x_grid = np.asarray(bg['x'])
    z_grid = np.asarray(bg['z'])
    zb = bg['zb']
    kappa_C = bg['kappa_C']
    Lz = normalized['params']['Lz']
    L_cz = normalized['params']['L_cz']

    if not np.all(np.diff(z_grid) > 0):
        raise ValueError('z_grid must be strictly increasing')

    if not np.isclose(zb, Lz-L_cz):
        raise ValueError('zb is inconsistent with Lz-L_cz')

    # Only C1 field data are needed to infer Dt.
    mask_frac = min(1, 1.1*(1-release_time_C1/normalized['stop_sim_time']))
    z_cz_mid = zb + L_cz/2
    f = diag.load_tracer_moments(run_folder, 'C1_concentration', bg['nu'], x_grid, z_grid, z_cz_mid, frac=mask_frac, back=True, step=5)

    # C2 and C3 are used only as held-out validation tracers.
    profile_tasks_available = diag.get_snapshot_tasks(run_folder, 'profile_snapshots')
    profile_tasks = get_optional_tracer_tasks(profile_tasks_available)
    p = diag.load_profile_snapshots(run_folder, tasks=profile_tasks)
    release_time_C2 = get_optional_tracer_release(p, 'C2', metadata_release_time_C2)
    release_time_C3 = get_optional_tracer_release(p, 'C3', metadata_release_time_C3)

    # Use common available end time.
    t_end = min(f['time'][-1], p['time'][-1])

    # Preserve the previously validated mature-C1 training interval.
    t_train_start = release_time_C1 + train_start_offset
    t_train_end = release_time_C1 + train_end_offset

    optional_release_times = [time for time in (release_time_C2, release_time_C3) if time is not None]
    if optional_release_times and t_train_end >= min(optional_release_times):
        raise ValueError('C1 training overlaps another tracer release')

    if f['time'][0] > t_train_start:
        raise ValueError('Field data start after training window')

    train_mask_f = (f['time'] >= t_train_start) & (f['time'] < t_train_end)

    if np.sum(train_mask_f) < 2:
        raise ValueError('Need two field training snapshots')

    print(f'Dt training interval: {t_train_start:.1f} - {t_train_end:.1f}')
    print(f'Dt training snapshots: {np.sum(train_mask_f)}')

    # Infer Dt only from mature C1.
    F_C1 = f['tracer_flux'][train_mask_f]
    C1_havg = f['tracer_havg'][train_mask_f]
    _, _, _, Dt_train_C1 = diag.compute_turbulent_tracer_diffusivity(F_C1, C1_havg, z_grid, zb)

    # Flow-derived bulk-CZ coefficient. The homogeneous tracer does not
    # constrain this normalization.
    D_conv, w_rms_cz, lambda_E, ell_mix = estimate_Dconv(f, train_mask_f, x_grid, z_grid, zb, L_cz)

    train_mask_p = ((p['time'] >= t_train_start) & (p['time'] <= t_train_end))
    if np.sum(train_mask_p) < 2:
        raise ValueError('Need two C1 fit profiles')

    fit_time = p['time'][train_mask_p]
    fit_profiles = p['C1_havg'][train_mask_p]

    # Fit only this early C1 interval, then freeze the diffusivity.
    sensitivity_factors = (1.0, 0.5, 2.0)
    solver_dt = np.median(np.diff(fit_time))/2
    mesa_fits = {}
    initial_q0 = 0.05
    initial_lambda = 0.1
    for factor in sensitivity_factors:
        mesa_fits[factor] = fit_mesa_diffusivity(
            z_grid, fit_profiles, fit_time, zb, L_cz, Lz, factor*D_conv, kappa_C, Nz=Nz_1d,
            initial_q0=initial_q0, initial_lambda=initial_lambda, max_dt=solver_dt,
        )
        if factor == 1.0:
            initial_q0 = mesa_fits[factor]['q0']
            initial_lambda = mesa_fits[factor]['lambda_ov']

    mesa_fit = mesa_fits[1.0]
    z_1d = mesa_fit['z_1d']
    Dmix_C1 = mesa_fit['Dmix_1d']
    pointwise_Dmix = build_pointwise_Dmix_1d(Dt_train_C1, z_grid, z_1d, zb)
    comparison_models = {
        'fitted_exponential': Dmix_C1,
        'molecular_only': np.zeros_like(z_1d),
        'constant_Dconv': np.full_like(z_1d, D_conv),
        'nonnegative_pointwise': pointwise_Dmix,
    }

    print(f'CZ vertical w_rms: {w_rms_cz:.6e}')
    print(f'CZ dominant horizontal wavelength: {lambda_E:.6e}')
    print(f'CZ mixing length used: {ell_mix:.6e}')
    print(f'Flow-derived CZ turbulent diffusivity: {D_conv:.6e}')
    print(f'MESA q0/Lcz: {mesa_fit["q0"]:.6e}')
    print(f'MESA lambda/Lcz: {mesa_fit["lambda_ov"]:.6e}')
    print(f'MESA boundary diffusivity: {mesa_fit["D_boundary"]:.6e}')
    print(f'MESA fit normalized RMS: {mesa_fit["objective_rms"]:.6e}')

    fig, _ = plot_mesa_Dt_model(z_grid, Dt_train_C1, z_1d, Dmix_C1, zb, L_cz, D_conv, mesa_fit['q0'])
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_1d_Dt.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_1d_Dt.png', dpi=180)

    fig, _ = plot_1d_tracer_prediction(
        mesa_fit['time_eval'], z_1d, mesa_fit['C_model'], mesa_fit['C_dns'], zb, L_cz, Lz,
        'C1 training fit',
    )
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_mesa_fit.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_mesa_fit.png', dpi=180)

    q0_surface, lambda_surface, objective_surface = evaluate_mesa_objective_surface(
        mesa_fit, zb, L_cz, Lz, D_conv, kappa_C,
    )
    fig, _ = plot_mesa_objective_surface(
        q0_surface, lambda_surface, objective_surface, mesa_fit['q0'], mesa_fit['lambda_ov'],
    )
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_mesa_objective.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_mesa_objective.png', dpi=180)

    sensitivity_summaries = {str(factor): mesa_fit_summary(mesa_fits[factor], factor) for factor in sensitivity_factors}
    fit_metadata = {
        'model': 'mesa_exponential',
        'coordinate': {
            'definition': 'q = (zb-z)/L_cz; positive downward',
            'boundary_z': float(zb),
            'L_cz': float(L_cz),
        },
        'flow_normalization': {
            'w_rms_cz': float(w_rms_cz),
            'lambda_E': float(lambda_E),
            'ell_mix': float(ell_mix),
            'D_conv': float(D_conv),
            'formula': 'D_conv = w_rms_cz*ell_mix/3',
        },
        'fit_window': {
            'requested_start': float(t_train_start),
            'requested_end': float(t_train_end),
            'actual_start': float(fit_time[0]),
            'actual_end': float(fit_time[-1]),
            'profiles': int(len(fit_time)),
            'training_tracer': 'C1',
        },
        'primary_fit': mesa_fit_summary(mesa_fit, 1.0),
        'sensitivity': sensitivity_summaries,
        'sensitivity_range': {
            key: [
                float(min(summary[key] for summary in sensitivity_summaries.values())),
                float(max(summary[key] for summary in sensitivity_summaries.values())),
            ]
            for key in ('D_boundary', 'q0', 'lambda_over_Lcz')
        },
        'validation': {},
    }
    validation_metadata = {
        'protocol': {
            'training_tracer': 'C1',
            'fit_end': float(fit_time[-1]),
            'frozen_after_fit': True,
            'one_continuous_trajectory_per_tracer': True,
            'grid': {'type': 'independent cell-centered finite volume', 'Nz': int(Nz_1d)},
            'max_dt': float(solver_dt),
            'stable_layer_mass_normalization': 'injected tracer mass',
        },
        'models': {
            'fitted_exponential': 'C1-fitted MESA exponential profile',
            'molecular_only': 'molecular diffusivity only',
            'constant_Dconv': 'constant flow-derived turbulent diffusivity',
            'nonnegative_pointwise': 'old clipped pointwise flux-gradient profile',
        },
        'comparisons': {},
    }

    # Evolve C1 once from the first training profile; do not reset at holdout.
    idx_C1_start = int(np.flatnonzero(train_mask_p)[0])
    t_C1_start = float(p['time'][idx_C1_start])
    C1_0_dns = p['C1_havg'][idx_C1_start]
    C1_0 = np.interp(z_1d, z_grid, C1_0_dns)
    C1_all_mask = (p['time'] > t_C1_start) & (p['time'] <= t_end)
    time_C1_all = p['time'][C1_all_mask]
    C1_dns_all = np.array([np.interp(z_1d, z_grid, C) for C in p['C1_havg'][C1_all_mask]])
    C1_all = compare_diffusivity_models(
        z_1d, comparison_models, kappa_C, time_C1_all, C1_0, t_C1_start, C1_dns_all, zb, Lz,
        max_dt=solver_dt,
    )
    holdout = time_C1_all > t_train_end
    time_C1 = time_C1_all[holdout]
    C1_dns = C1_dns_all[holdout]
    injected_mass_C1 = np.sum(C1_0)*Lz/Nz_1d
    C1_comparisons = {}
    for name, result in C1_all.items():
        C_model = result['C_model'][holdout]
        C1_comparisons[name] = {
            'C_model': C_model,
            'errors': compute_prediction_errors(z_1d, C_model, C1_dns, zb, Lz, injected_mass_C1),
        }
    validation_metadata['comparisons']['C1_temporal_holdout'] = comparison_summary(C1_comparisons)
    C1_model = C1_comparisons['fitted_exponential']['C_model']
    C1_errors = C1_comparisons['fitted_exponential']['errors']
    fit_metadata['validation']['C1_temporal_holdout'] = prediction_error_summary(C1_errors)
    rmse_full_C1, rmse_rz_C1, _, _, mass_error_C1, model_mass_drift_C1, dns_mass_drift_C1 = C1_errors
    print_prediction_errors('C1 temporal holdout', rmse_full_C1, rmse_rz_C1, mass_error_C1, model_mass_drift_C1, dns_mass_drift_C1)
    fig, _ = plot_1d_tracer_prediction(time_C1, z_1d, C1_model, C1_dns, zb, L_cz, Lz, 'C1')
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_1d_prediction.pdf')
    fig.savefig(f'{out_folder}/{args.run_id}_p4_C1_1d_prediction.png', dpi=180)

    # C2 and C3 remain independent held-out tracers and never enter the fit.
    tracer_specs = (
        ('C2', release_time_C2, 'C2_fresh_step', f'{args.run_id}_p4_C1Dt_C2_1d_prediction'),
        ('C3', release_time_C3, 'C3_fresh_gaussian', f'{args.run_id}_p4_C1Dt_C3_1d_prediction'),
    )
    for tracer, release_time, metadata_key, figure_name in tracer_specs:
        if release_time is None:
            fit_metadata['validation'][metadata_key] = None
            validation_metadata['comparisons'][metadata_key] = None
            print(f'Skipping {tracer} validation: tracer profile is absent or never released')
            continue
        profiles = p[f'{tracer}_havg']
        released = np.any(profiles != 0, axis=1) & (p['time'] >= release_time)
        idx_start = int(np.flatnonzero(released)[0])
        t_start = float(p['time'][idx_start])
        C0 = np.interp(z_1d, z_grid, profiles[idx_start])
        test_mask = (p['time'] > t_start) & (p['time'] <= t_end)
        time_test = p['time'][test_mask]
        C_dns = np.array([np.interp(z_1d, z_grid, C) for C in profiles[test_mask]])
        comparisons = compare_diffusivity_models(
            z_1d, comparison_models, kappa_C, time_test, C0, t_start, C_dns, zb, Lz,
            max_dt=solver_dt,
        )
        validation_metadata['comparisons'][metadata_key] = comparison_summary(comparisons)
        primary = comparisons['fitted_exponential']
        errors = primary['errors']
        fit_metadata['validation'][metadata_key] = prediction_error_summary(errors)
        rmse_full, rmse_rz, _, _, mass_error, model_mass_drift, dns_mass_drift = errors
        print_prediction_errors(f'{tracer} using frozen C1-fitted diffusivity', rmse_full, rmse_rz, mass_error, model_mass_drift, dns_mass_drift)
        fig, _ = plot_1d_tracer_prediction(
            time_test, z_1d, primary['C_model'], C_dns, zb, L_cz, Lz, tracer,
        )
        fig.savefig(f'{out_folder}/{figure_name}.pdf')
        fig.savefig(f'{out_folder}/{figure_name}.png', dpi=180)

    convergence_resolutions = (Nz_1d//2, Nz_1d)
    convergence_timesteps = (2*solver_dt, solver_dt)
    validation_metadata['convergence'] = {
        'resolutions': list(convergence_resolutions),
        'timesteps': [float(dt) for dt in convergence_timesteps],
        'C1_temporal_holdout': numerical_convergence_check(
            z_grid, C1_0_dns, time_C1, p['C1_havg'][C1_all_mask][holdout], t_C1_start, zb, L_cz,
            Lz, D_conv, mesa_fit['q0'], mesa_fit['lambda_ov'], kappa_C,
            convergence_resolutions, convergence_timesteps,
        ),
    }

    fit_output = f'analysis/{args.mode}s/{args.run_id}_mesa_diffusivity_fit.json'
    validation_output = f'analysis/{args.mode}s/{args.run_id}_diffusivity_validation.json'
    write_json_atomic(fit_output, fit_metadata)
    write_json_atomic(validation_output, validation_metadata)
    print()
    print(f'Wrote diffusion validation figures to {out_folder}/{args.run_id}_p4_*.pdf')
    print(f'Wrote MESA diffusivity fit metadata to {fit_output}')
    print(f'Wrote independent validation results to {validation_output}')

if __name__ == '__main__':
    main()
