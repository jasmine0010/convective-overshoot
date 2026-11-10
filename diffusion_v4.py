import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

import diagnostics as diag
from metadata_io import load_run_metadata, write_json_atomic


def build_Dmix_1d(zb, L_cz, Lz, ell_mix, q0, lambda_ov, z_dns, w_rms_z, Nz=300):
    dz = Lz/Nz
    z_1d = (np.arange(Nz)+0.5)*dz
    q = (zb-z_1d)/L_cz
    z_ref = zb+q0*L_cz

    Dmix_1d = ell_mix*np.interp(z_1d, z_dns, w_rms_z)/3
    D_ref = ell_mix*np.interp(z_ref, z_dns, w_rms_z)/3
    exponential = q > -q0
    Dmix_1d[exponential] = D_ref*np.exp(-(q[exponential]+q0)/lambda_ov)

    return z_1d, Dmix_1d


def solve_tracer_diffusion_full(z_1d, Dmix_1d, kappa_C, time_eval, C0, t_start, max_dt):
    dz = z_1d[1] - z_1d[0]
    K = kappa_C + Dmix_1d
    K_face = 2*K[:-1]*K[1:]/(K[:-1]+K[1:])
    C = C0.copy()

    times = np.concatenate(([t_start], time_eval))
    C_all = np.empty((len(time_eval), len(z_1d)))

    solve_by_timestep = {}
    for n in range(len(times)-1):
        interval = times[n+1] - times[n]
        steps = int(np.ceil(interval/max_dt))
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
        C_all[n] = C

    return time_eval, C_all


def region_weights(z, dz, z_min, z_max):
    left = z - dz/2
    right = z + dz/2
    return np.maximum(0, np.minimum(right, z_max)-np.maximum(left, z_min))


def build_pointwise_Dmix_1d(Dt, z_grid, z_1d, zb):
    # Keep unresolved points distinct from resolved zero diffusivity.
    Dmix = np.interp(z_1d, z_grid, Dt, left=np.nan, right=np.nan)
    return np.maximum(Dmix, 0)


def finite_volume_grid(Lz, Nz):
    edges = np.linspace(0, Lz, Nz+1)
    return 0.5*(edges[:-1]+edges[1:]), edges


def cell_average(function, edges):
    nodes, weights = np.polynomial.legendre.leggauss(8)
    centers = 0.5*(edges[:-1]+edges[1:])
    half_widths = 0.5*np.diff(edges)
    points = centers[:, None]+half_widths[:, None]*nodes
    return 0.5*np.sum(function(points)*weights, axis=1)


def remap(z_source, profiles, edges):
    prepend = z_source[0] > edges[0]
    append = z_source[-1] < edges[-1]
    z = np.r_[edges[0] if prepend else [], z_source, edges[-1] if append else []]
    mapped = []

    for profile in profiles:
        y = np.r_[profile[0] if prepend else [], profile, profile[-1] if append else []]
        dx = np.diff(z)
        slopes = np.diff(y)/dx
        integral = np.r_[0, np.cumsum(0.5*(y[:-1]+y[1:])*dx)]
        index = np.clip(np.searchsorted(z, edges, side='right')-1, 0, len(z)-2)
        offset = edges-z[index]
        antiderivative = integral[index]+y[index]*offset+0.5*slopes[index]*offset**2
        mapped.append(np.diff(antiderivative)/np.diff(edges))

    return np.asarray(mapped)


def release_initial_condition(tracer, edges, metadata):
    params = metadata['params']
    background = metadata['background']
    release_time = float(metadata['timing'][f'release_time_{tracer}'])
    zb, L_cz = background['zb'], params['L_cz']

    if tracer == 'C3':
        mu, sigma = zb+0.15*L_cz, 0.15*L_cz
        amplitude = background.get('C3_amplitude', 1.0)
        profile = cell_average(lambda z: amplitude*np.exp(-(z-mu)**2/(2*sigma**2)), edges)
        return profile, release_time, 'prescribed_gaussian'

    tw = params['tw']
    profile = cell_average(lambda z: 0.5*(1+np.tanh((z-zb)/tw)), edges)
    return profile, release_time, 'prescribed_tanh'


def normalized_residual(model, dns, initial):
    scale = np.max(np.abs(initial))
    return (model-dns).ravel()/(scale*np.sqrt(model.size))


def fit_frozen_models(z, C0, release_time, train_times, train_dns, zb, L_cz, Lz, w_rms_cz, kappa_C, pointwise_D, max_dt, z_dns, w_rms_z):
    dz = z[1]-z[0]
    D_scale = max(w_rms_cz*L_cz/3, kappa_C)
    D_bounds = (1e-6*D_scale, 10*D_scale)
    ell_bounds = (3*D_bounds[0]/w_rms_cz, 3*D_bounds[1]/w_rms_cz)
    q0_bounds = (max(1e-4, 1e-2*dz/L_cz), 1.0)
    lambda_bounds = (max(1e-4, 0.5*dz/L_cz), max(1.0, zb/L_cz))

    def exponential_residual(log_parameters):
        ell_mix, q0, lambda_ov = np.exp(log_parameters)
        _, D = build_Dmix_1d(zb, L_cz, Lz, ell_mix, q0, lambda_ov, z_dns, w_rms_z, Nz=len(z))
        _, model = solve_tracer_diffusion_full(z, D, kappa_C, train_times, C0, release_time, max_dt=max_dt)
        return normalized_residual(model, train_dns, C0)

    lower = np.log([ell_bounds[0], q0_bounds[0], lambda_bounds[0]])
    upper = np.log([ell_bounds[1], q0_bounds[1], lambda_bounds[1]])
    initial = np.log([L_cz, 0.05, 0.1])
    fit = least_squares(exponential_residual, initial, bounds=(lower, upper), max_nfev=100)
    ell_mix, q0, lambda_ov = np.exp(fit.x)
    _, exponential_D = build_Dmix_1d(zb, L_cz, Lz, ell_mix, q0, lambda_ov, z_dns, w_rms_z, Nz=len(z))

    constant_bounds = (max(np.finfo(float).eps, 1e-6*D_scale), 10*D_scale)

    def constant_objective(log_D):
        D = np.full_like(z, np.exp(log_D))
        _, model = solve_tracer_diffusion_full(z, D, kappa_C, train_times, C0, release_time, max_dt=max_dt)
        return np.linalg.norm(normalized_residual(model, train_dns, C0))

    constant_fit = minimize_scalar(constant_objective, bounds=np.log(constant_bounds), method='bounded', options={'xatol': 1e-6})
    constant_D = float(np.exp(constant_fit.x))
    
    models = {
        'fitted_exponential': exponential_D,
        'constant_training_fit': np.full_like(z, constant_D),
        'flux-gradient': np.where(np.isfinite(pointwise_D), np.maximum(pointwise_D, 0), np.where(z >= zb, exponential_D, 0)),
        'molecular_only': np.zeros_like(z),
        'cz_mixing_only': np.where(z >= zb, exponential_D, 0.0)
    }

    bound_distance = np.minimum(fit.x-lower, upper-fit.x)
    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    sensitivity_ratio = float(singular_values[-1]/max(singular_values[0], np.finfo(float).eps))

    fit_info = {
        'D_cz': float(ell_mix*w_rms_cz/3),
        'ell_mix_fit': float(ell_mix),
        'q0': float(q0),
        'lambda_over_Lcz': float(lambda_ov),
        'D_boundary': float(ell_mix*np.interp(zb+q0*L_cz, z_dns, w_rms_z)*np.exp(-q0/lambda_ov)/3),
        'constant_D': constant_D,
        'exponential_fit_at_bounds': bool(np.any(bound_distance <= 1e-3*(upper-lower))),
        'weakly_constrained': bool(sensitivity_ratio < 1e-3),
        'exponential_fit_success': bool(fit.success),
        'constant_fit_success': bool(constant_fit.success)
    }
    return models, fit_info


def prediction_metrics(z, model, dns, C0, zb, Lz):
    dz = z[1]-z[0]
    stable_weights = region_weights(z, dz, 0, zb)
    full_weights = region_weights(z, dz, 0, Lz)
    injected_mass = np.sum(C0*full_weights)
    residual = model-dns
    full_rmse = np.sqrt(np.sum(residual**2*full_weights, axis=1)/np.sum(full_weights))
    initial_scale = max(float(np.ptp(C0)), float(np.max(np.abs(C0))), np.finfo(float).eps)
    model_mass = np.sum(model*full_weights, axis=1)
    dns_mass = np.sum(dns*full_weights, axis=1)

    return {
        'full_profile_rmse': full_rmse,
        'normalized_full_profile_rmse': full_rmse/initial_scale,
        'stable_layer_rmse': np.sqrt(np.sum(residual**2*stable_weights, axis=1)/np.sum(stable_weights)),
        'stable_layer_mass_error_over_injected_mass': (
            np.sum(residual*stable_weights, axis=1)/injected_mass
        ),
        'total_mass_drift': (model_mass-injected_mass)/injected_mass,
        'dns_total_mass_drift': (dns_mass-injected_mass)/injected_mass,
    }


def evaluate_models(z, models, kappa_C, times, C0, release_time, dns, zb, Lz, max_dt):
    evaluated = {}
    for name, D in models.items():
        if name == 'flux-gradient':  # Diffusivity diagnostic only, not a forward model.
            continue
        _, model = solve_tracer_diffusion_full(z, D, kappa_C, times, C0, release_time, max_dt=max_dt)
        evaluated[name] = {
            'profiles': model,
            'metrics': prediction_metrics(z, model, dns, C0, zb, Lz),
        }
    return evaluated


def metric_summary(metrics):
    return {
        'mean_full_profile_rmse': float(np.mean(metrics['full_profile_rmse'])),
        'mean_stable_layer_rmse': float(np.mean(metrics['stable_layer_rmse'])),
        'final_stable_layer_mass_error_over_injected_mass': float(metrics['stable_layer_mass_error_over_injected_mass'][-1]),
        'max_abs_total_mass_drift': float(np.max(np.abs(metrics['total_mass_drift']))),
    }


def convergence_check(z_dns, dns_profiles, times, tracer, metadata, zb, L_cz, Lz, ell_mix, q0, lambda_ov, kappa_C, resolutions, timesteps, w_rms_z):
    results = {}
    for Nz in resolutions:
        z, edges = finite_volume_grid(Lz, Nz)
        C0, t0, _ = release_initial_condition(tracer, edges, metadata)
        dns = remap(z_dns, dns_profiles, edges)
        _, D = build_Dmix_1d(zb, L_cz, Lz, ell_mix, q0, lambda_ov, z_dns, w_rms_z, Nz=Nz)
        for max_dt in timesteps:
            _, model = solve_tracer_diffusion_full(z, D, kappa_C, times, C0, t0, max_dt=max_dt)
            metrics = prediction_metrics(z, model, dns, C0, zb, Lz)
            results[f'Nz={Nz},dt={max_dt:g}'] = metric_summary(metrics)
    return results


def plot_comparison(z, zb, L_cz, validation):
    tracers = list(validation)
    fig, axes = plt.subplots(1, len(tracers), figsize=(diag.FIG_DOUBLE, 2.5), squeeze=False)
    depth = diag.normalized_depth(z, zb, L_cz)
    styles = {
        'fitted_exponential': (diag.COLOR_MODEL, '--', 'Exponential fit'),
        'constant_training_fit': ('#009E73', '-.', 'Constant fit'),
        'molecular_only': ('#CC79A7', '--', 'Molecular only'),
        'cz_mixing_only': ('0.15', ':', 'CZ only')
    }

    for ax, tracer in zip(axes[0], tracers):
        result = validation[tracer]
        ax.plot(depth, result['dns'][-1], color=diag.COLOR_SIMULATION, linewidth=diag.LINEWIDTH_PRIMARY,
                label='2D simulation')
        for name, (color, linestyle, label) in styles.items():
            ax.plot(depth, result['evaluated'][name]['profiles'][-1], color=color, linestyle=linestyle,
                    linewidth=diag.LINEWIDTH_SECONDARY, label=label)
        ax.axvline(0, color=diag.COLOR_BOUNDARY, linestyle='--', linewidth=diag.LINEWIDTH_REFERENCE)
        ax.set_title(tracer)
        ax.set_xlabel('Depth')
        ax.set_ylabel(r'$C$')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    diag.add_panel_labels(axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    diag.center_legend_above_axes(fig, axes[0], gap=0.15)
    plt.close(fig)
    return fig, axes[0]


def plot_C2_profiles(z, zb, L_cz, result, release_time):
    fig, axes = plt.subplots(2, 3, figsize=(diag.FIG_DOUBLE, 3.2), sharex='col', sharey='row', gridspec_kw={'height_ratios': [3, 1]})
    depth = diag.normalized_depth(z, zb, L_cz)
    for ax, residual_ax, age in zip(axes[0], axes[1], (50, 200, 400)):
        index = np.argmin(np.abs(result['times']-release_time-age))
        ax.plot(depth, result['dns'][index], color=diag.COLOR_SIMULATION, linewidth=diag.LINEWIDTH_PRIMARY, label='2D simulation')
        ax.plot(depth, result['evaluated']['fitted_exponential']['profiles'][index], color=diag.COLOR_MODEL, linestyle='--', label='Exponential fit')
        ax.axvline(0, color=diag.COLOR_BOUNDARY, linestyle='--', linewidth=diag.LINEWIDTH_REFERENCE)
        ax.set_title(f"{result['times'][index]-release_time:.0f} after release")
        residual_ax.plot(depth, result['evaluated']['fitted_exponential']['profiles'][index]-result['dns'][index], color=diag.COLOR_MODEL, linestyle='--', label='Exponential fit')
        residual_ax.axhline(0, color='0.6', linewidth=diag.LINEWIDTH_REFERENCE)
        residual_ax.axvline(0, color=diag.COLOR_BOUNDARY, linestyle='--', linewidth=diag.LINEWIDTH_REFERENCE)
        residual_ax.set_xlabel('Depth')
    axes[0, 0].set_ylim(top=1.1*axes[0, 0].get_ylim()[1])
    axes[0, 0].set_ylabel(r'$C$')
    axes[1, 0].set_ylabel('Residual')
    for ax in axes.flat:
        ax.tick_params(length=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    diag.add_panel_labels(axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.86), h_pad=0.3)
    diag.center_legend_above_axes(fig, axes[0], gap=0.25)
    plt.close(fig)
    return fig, axes


def plot_rmse(series, training_window, time_label='Simulation time'):
    fig, axes = plt.subplots(2, 1, figsize=(diag.FIG_DOUBLE, diag.FIG_MULTI_HEIGHT), sharex=True)
    colors = {'C1': diag.COLOR_C1, 'C2': diag.COLOR_C2, 'C3': diag.COLOR_C3}
    styles = {'C1': '-', 'C2': '--', 'C3': '-.'}
    for ax, metric, label in zip(axes, ('stable_layer_rmse', 'full_profile_rmse'), ('Stable-layer RMSE', 'Full-profile RMSE')):
        for tracer, result in series.items():
            ax.plot(result['times'], result['metrics'][metric], color=colors[tracer], linestyle=styles[tracer],
                    linewidth=diag.LINEWIDTH_TIMESERIES, label=tracer)
        if training_window is not None:
            ax.axvspan(*training_window, color='0.75', alpha=0.25, linewidth=0, label='C1 fit interval', zorder=0)
        ax.set_ylabel(label)
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
    axes[1].set_xlabel(time_label)
    diag.add_panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    diag.center_legend_above_axes(fig, axes, gap=0.05)
    plt.close(fig)
    return fig, axes


def plot_rmse_matched_age(series, release_times, training_window):
    elapsed = {tracer: {'times': result['times']-release_times[tracer], 'metrics': result['metrics']}
               for tracer, result in series.items()}
    start = max(result['times'][0] for result in elapsed.values())
    end = min(result['times'][-1] for result in elapsed.values())
    fit_window = tuple(time-release_times['C1'] for time in training_window)
    if fit_window[0] > end or fit_window[1] < start:
        fit_window = None
    fig, axes = plot_rmse(elapsed, fit_window, 'Time since release')
    axes[1].set_xlim(start, end)
    return fig, axes


def plot_residuals(z, zb, L_cz, validation):
    tracers = list(validation)
    residuals = [validation[tracer]['evaluated']['fitted_exponential']['profiles']-validation[tracer]['dns']
                 for tracer in tracers]
    limit = max(np.max(np.abs(residual)) for residual in residuals)
    fig, axes = plt.subplots(1, len(tracers), figsize=(diag.FIG_DOUBLE, 2.5), squeeze=False)
    depth = diag.normalized_depth(z, zb, L_cz)

    for ax, tracer, residual in zip(axes[0], tracers, residuals):
        image = ax.pcolormesh(depth, validation[tracer]['times'], residual, cmap='RdBu_r',
                              vmin=-limit, vmax=limit, shading='auto', rasterized=True)
        ax.axvline(0, color=diag.COLOR_BOUNDARY, linestyle='--', linewidth=diag.LINEWIDTH_REFERENCE)
        ax.set_title(tracer)
        ax.set_xlabel('Depth')
        ax.set_ylabel('Time')

    diag.add_panel_labels(axes[0])
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.18, top=0.83, wspace=0.35)
    colorbar_ax = fig.add_axes([0.89, 0.18, 0.018, 0.65])
    fig.colorbar(image, cax=colorbar_ax, label='1D - 2D concentration')
    plt.close(fig)
    return fig, axes[0]


def plot_model_diffusivities(z, zb, L_cz, models):
    depth = diag.normalized_depth(z, zb, L_cz)
    styles = {
        'fitted_exponential': (diag.COLOR_MODEL, '--', 'Exponential fit'),
        'constant_training_fit': ('#009E73', '-.', 'Constant fit'),
        'cz_mixing_only': ('0.15', ':', 'CZ only')
    }
    fig, ax = plt.subplots(figsize=(diag.FIG_SINGLE, 3.1))
    axes = [ax]
    for ax in axes:
        diag.shade_layers(ax, depth)
        ax.axhline(0, color='0.7', linewidth=diag.LINEWIDTH_REFERENCE, zorder=0.5)
        for name, (color, linestyle, label) in styles.items():
            ax.plot(depth, models[name], color=color, linestyle=linestyle,
                    linewidth=diag.LINEWIDTH_PRIMARY if name == 'fitted_exponential' else diag.LINEWIDTH_SECONDARY,
                    label=label)
        ax.axvline(0, color=diag.COLOR_BOUNDARY, linestyle='--', linewidth=diag.LINEWIDTH_REFERENCE,
                   label='Convective boundary')
        ax.set_xlabel('Depth')

    full_top = 1.1*max(np.max(models[name]) for name in styles)
    axes[0].set_xlim(np.min(depth), np.max(depth))
    axes[0].set_ylim(-0.05*full_top, full_top)
    axes[0].set_ylabel(r'$D_{\rm turb}$')
    diag.add_panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.75)
    diag.center_legend_above_axes(fig, axes, gap=0.05)
    plt.close(fig)
    return fig, axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=['test', 'run'], required=True)
    args = parser.parse_args()
    run_folder = diag.get_run_folder(args.run_id, args.mode)
    output_dir = f'analysis/{args.mode}s'
    figure_dir = f'{output_dir}/figures'
    os.makedirs(figure_dir, exist_ok=True)

    _, metadata = load_run_metadata(f'{run_folder}/info.json')
    params, background = metadata['params'], metadata['background']
    x_dns, z_dns = np.asarray(background['x']), np.asarray(background['z'])
    zb, L_cz, Lz = background['zb'], params['L_cz'], params['Lz']
    kappa_C = background['kappa_C']
    nz_1d = 300
    training_offsets = (400, 550)
    z, edges = finite_volume_grid(Lz, nz_1d)

    tracers = ('C1', 'C2', 'C3')
    profiles = diag.load_profile_snapshots(run_folder, tasks=[f'{tracer}_havg' for tracer in tracers])

    C1_0, C1_start, _ = release_initial_condition('C1', edges, metadata)
    train_start, train_end = (C1_start+offset for offset in training_offsets)
    train_mask = (profiles['time'] >= train_start) & (profiles['time'] <= train_end)
    train_times = profiles['time'][train_mask]
    train_dns = remap(z_dns, profiles['C1_havg'][train_mask], edges)

    field = diag.load_tracer_moments(run_folder, 'C1_concentration', background['nu'], x_dns, z_dns, zb+L_cz/2, step=5)
    field_train = (field['time'] >= train_start) & (field['time'] <= train_end)

    w_rms_z = np.sqrt(field['w_sq_havg'][field_train].mean(axis=0))
    w_rms_cz = diag.compute_u_rms_from_moments(field['w_sq_havg'], z_dns, (zb, zb+L_cz), field_train)
    lambda_E = diag.compute_lambda_E(field['w_mid'][field_train], x_dns)

    _, _, _, pointwise_dns = diag.compute_turbulent_tracer_diffusivity(field['tracer_flux'][field_train], field['tracer_havg'][field_train], z_dns, zb)
    pointwise_D = build_pointwise_Dmix_1d(pointwise_dns, z_dns, z, zb)
    max_dt = float(np.median(np.diff(profiles['time'])))

    models, fit = fit_frozen_models(z, C1_0, C1_start, train_times, train_dns, zb, L_cz, Lz, w_rms_cz, kappa_C, pointwise_D, max_dt, z_dns, w_rms_z)
    D_cz = fit['D_cz']
    ell_mix_fit = fit['ell_mix_fit']

    validation = {}
    release_times = {}
    for tracer in tracers:
        observed = profiles[f'{tracer}_havg']
        C0, start, _ = release_initial_condition(tracer, edges, metadata)
        release_times[tracer] = start
        mask = profiles['time'] > (train_end if tracer == 'C1' else start)
        times = profiles['time'][mask]
        dns = remap(z_dns, observed[mask], edges)
        evaluated = evaluate_models(z, models, kappa_C, times, C0, start, dns, zb, Lz, max_dt)
        validation[tracer] = {'times': times, 'dns': dns, 'evaluated': evaluated}

    rmse_series = {
        tracer: {'times': result['times'],
                 'metrics': result['evaluated']['fitted_exponential']['metrics']}
        for tracer, result in validation.items()
    }
    C1_since_release = profiles['time'] > C1_start
    C1_times = profiles['time'][C1_since_release]
    C1_dns = remap(z_dns, profiles['C1_havg'][C1_since_release], edges)
    _, C1_prediction = solve_tracer_diffusion_full(z, models['fitted_exponential'], kappa_C,
                                                   C1_times, C1_0, C1_start, max_dt)
    C1_metrics = prediction_metrics(z, C1_prediction, C1_dns, C1_0, zb, Lz)
    rmse_series['C1'] = {'times': C1_times, 'metrics': C1_metrics}

    convergence_resolutions = (nz_1d//2, nz_1d)
    convergence_timesteps = (2*max_dt, max_dt)
    C1_mask = profiles['time'] > train_end
    convergence = convergence_check(
        z_dns, profiles['C1_havg'][C1_mask], profiles['time'][C1_mask], 'C1', metadata,
        zb, L_cz, Lz, ell_mix_fit, fit['q0'], fit['lambda_over_Lcz'], kappa_C,
        convergence_resolutions, convergence_timesteps, w_rms_z,
    )

    figures = (
        ('v4_C2_profiles_over_time', plot_C2_profiles(z, zb, L_cz, validation['C2'], release_times['C2'])[0]),
        ('v4_diffusivity_comparison', plot_comparison(z, zb, L_cz, validation)[0]),
        ('v4_rmse', plot_rmse(rmse_series, (train_start, train_end))[0]),
        ('v4_rmse_matched_age', plot_rmse_matched_age(rmse_series, release_times, (train_start, train_end))[0]),
        ('v4_exponential_residuals', plot_residuals(z, zb, L_cz, validation)[0]),
        ('v4_model_diffusivities', plot_model_diffusivities(z, zb, L_cz, models)[0]),
    )
    for name, fig in figures:
        fig.savefig(f'{figure_dir}/{args.run_id}_{name}.pdf')
        fig.savefig(f'{figure_dir}/{args.run_id}_{name}.png', dpi=180)

    output = {
        'errors': {
            tracer: {name: metric_summary(model['metrics']) for name, model in result['evaluated'].items()}
            for tracer, result in validation.items()
        },
        'fit': {
            'C1_training_window': [float(train_start), float(train_end)],
            'D_cz': float(D_cz),
            'w_rms_cz': float(w_rms_cz),
            'ell_mix_fit': float(ell_mix_fit),
            'lambda_E': float(lambda_E),
            'ell_mix_fit_over_Lcz': float(ell_mix_fit/L_cz),
            'lambda_E_over_Lcz': float(lambda_E/L_cz),
            'q0': fit['q0'],
            'lambda_over_Lcz': fit['lambda_over_Lcz'],
            'D_boundary': fit['D_boundary'],
            'constant_D': fit['constant_D'],
            'exponential_fit_at_bounds': fit['exponential_fit_at_bounds'],
            'weakly_constrained': bool(fit['weakly_constrained']),
            'exponential_fit_success': fit['exponential_fit_success'],
            'constant_fit_success': fit['constant_fit_success'],
        },
        'convergence_C1': convergence,
    }
    output_path = f'{output_dir}/{args.run_id}_diffusivity_validation_v4.json'
    write_json_atomic(output_path, output)

    print(f'RMS-based CZ diffusivity: {D_cz:.6g}')
    print(f'Measured w_rms, CZ: {w_rms_cz:.6g}')
    print(f'Effective mixing length: {ell_mix_fit:.6g}')
    print(f'ell_mix_fit/L_cz: {ell_mix_fit/L_cz:.6g}')
    print(f'DNS lambda_E: {lambda_E:.6g}')
    print(f'lambda_E/L_cz: {lambda_E/L_cz:.6g}')
    print(f'Fitted q0: {fit['q0']:.6g}')
    print(f'lambda/L_cz: {fit['lambda_over_Lcz']:.6g}')
    print(f'Fit at bounds: {fit['exponential_fit_at_bounds']}')
    print(f'Weakly constrained: {fit['weakly_constrained']}')
    
    print(f'Wrote {output_path}')
    print(f'Wrote {figure_dir}/{args.run_id}_v4_*.pdf')


if __name__ == '__main__':
    main()
