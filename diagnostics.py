import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import curve_fit, minimize_scalar

from metadata_io import write_json_atomic


FIG_SINGLE = 3.4
FIG_DOUBLE = 7.0
FIG_HEIGHT = 2.6
FIG_MULTI_HEIGHT = 4.8
PANEL_LABEL_SIZE = 10
LINEWIDTH_PRIMARY = 1.6
LINEWIDTH_SECONDARY = 1.2
LINEWIDTH_TIMESERIES = 1.0
LINEWIDTH_REFERENCE = 0.8
MARKER_SIZE = 3.5
COLOR_SIMULATION = '#0072B2'
COLOR_MODEL = '#E69F00'
COLOR_C1 = '#0072B2'
COLOR_C2 = '#E69F00'
COLOR_C3 = '#009E73'
COLOR_BOUNDARY = '#D62728'
COLOR_THRESHOLD = '0.55'
COLOR_CONVECTIVE_LAYER = '#DDEBDD'
COLOR_AVERAGING_INTERVAL = '#C7D7EE'
AVERAGING_INTERVAL_ALPHA = 0.45

plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'font.size': 8.5,
    'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'legend.frameon': False,
    'lines.linewidth': LINEWIDTH_SECONDARY,
    'lines.markersize': MARKER_SIZE,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0,
})


def normalized_depth(z, zb, L_cz):
    return (zb-np.asarray(z))/L_cz


def add_panel_labels(axes):
    texts = []
    for index, ax in enumerate(np.asarray(axes, dtype=object).ravel()):
        texts.append(ax.text(0.02, 0.98, f'({chr(97+index)})', transform=ax.transAxes, ha='left', va='top',
                             fontsize=PANEL_LABEL_SIZE))
    return texts


def center_legend_above_axes(fig, axes, gap=0.08):
    positions = [ax.get_position() for ax in np.asarray(axes, dtype=object).ravel()]
    center = (min(pos.x0 for pos in positions)+max(pos.x1 for pos in positions))/2
    top = max(pos.y1 for pos in positions)+gap/fig.get_figheight()
    fig.legends[-1].set_bbox_to_anchor((center, top), transform=fig.transFigure)


def shade_layers(ax, depth, orientation='x'):
    depth_min = np.nanmin(depth)
    if orientation == 'x':
        convective = ax.axvspan(depth_min, 0, color=COLOR_CONVECTIVE_LAYER, alpha=0.35, zorder=0)
        ax.margins(x=0)
    else:
        convective = ax.axhspan(depth_min, 0, color=COLOR_CONVECTIVE_LAYER, alpha=0.35, zorder=0)
        ax.margins(y=0)
    return (convective,)


def add_reference_legend(ax, loc='upper right'):
    handles, labels = ax.get_legend_handles_labels()
    references = [(handle, label) for handle, label in zip(handles, labels)
                  if label in {'Convective boundary', '5% depth',
                               'Variance depth', 'Residual depth', 'Mixing depth',
                               'Molecular diffusivity'}]
    if not references:
        return None
    handles, labels = zip(*references)
    legend = ax.legend(handles, labels, loc=loc, frameon=False, fontsize=8.5,
                       handlelength=1.6, borderaxespad=0.3)
    ax.add_artist(legend)
    return legend


def get_data_legend_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    references = {'Convective boundary', '5% depth', 'Variance depth',
                  'Residual depth', 'Mixing depth', 'Molecular diffusivity'}
    selected = [(handle, label) for handle, label in zip(handles, labels) if label not in references]
    return tuple(zip(*selected)) if selected else ((), ())


def get_run_folder(run_id, mode):
    base = f'data/{mode}s'
    found = sorted(glob.glob(f'{base}/{run_id}_Ra*'))
    if not found:
        raise FileNotFoundError(f'No run folder: {base}/{run_id}_Ra*')
    if len(found) > 1:
        raise RuntimeError(f'Ambiguous run folder: {run_id!r}')
    return found[0]


def get_h5_files(folder, prefix):
    return sorted(glob.glob(f'{folder}/{prefix}_s*.h5'), key=lambda x: int(x.split('_s')[-1].split('.h5')[0]))


def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float) and not np.isfinite(obj):
        return None
    else:
        return obj


def get_snapshot_tasks(folder, prefix):
    files = get_h5_files(f'{folder}/{prefix}', prefix)
    if not files:
        raise FileNotFoundError(f'No {prefix} files: {folder}/{prefix}')

    with h5py.File(files[0], mode='r') as file:
        return set(file['tasks'].keys())


def _selected_snapshot_files(folder, tasks, prefix, frac, back, step):
    if not tasks:
        raise ValueError('No snapshot tasks')
    if not 0 < frac <= 1:
        raise ValueError('frac must be in (0, 1]')
    if not isinstance(step, int) or step < 1:
        raise ValueError('step must be a positive integer')

    files = get_h5_files(f'{folder}/{prefix}', prefix)
    if not files:
        raise FileNotFoundError(f'No {prefix} files: {folder}/{prefix}')

    n_files = max(1, int(frac*len(files)))
    return files[-n_files:] if back else files[:n_files]


def iter_snapshot_chunks(folder, tasks, prefix='field_snapshots', frac=1, back=True, step=1):
    """Yield selected snapshots one HDF5 file at a time in existing file order."""
    files = _selected_snapshot_files(folder, tasks, prefix, frac, back, step)
    for path in files:
        with h5py.File(path, mode='r') as file:
            missing = [task for task in tasks if task not in file['tasks']]
            if missing:
                raise KeyError(f'Missing HDF5 tasks in {path}: {missing}')

            chunk = {task: file['tasks'][task][::step] for task in tasks}
            chunk['time'] = np.asarray(file['tasks'][tasks[0]].dims[0]['sim_time'][::step])
        yield chunk


def _load_snapshots(folder, tasks, prefix, frac, back, step):
    data = {task: [] for task in tasks}
    time = []

    for chunk in iter_snapshot_chunks(folder, tasks, prefix, frac, back, step):
        for task in tasks:
            data[task].append(chunk[task])
        time.append(chunk['time'])

    for task in tasks:
        arr = np.concatenate(data[task])
        singleton_axes = tuple(axis for axis, size in enumerate(arr.shape) if axis != 0 and size == 1)
        if singleton_axes:
            arr = np.squeeze(arr, axis=singleton_axes)
        data[task] = arr
    data['time'] = np.concatenate(time)
    return data


def load_field_snapshots(folder, tasks, prefix='field_snapshots', frac=1, back=True, step=1):
    return _load_snapshots(folder, tasks, prefix, frac, back, step)


def load_snapshot_at_time(folder, tasks, target_time, prefix='field_snapshots'):
    files = get_h5_files(f'{folder}/{prefix}', prefix)
    if not files:
        raise FileNotFoundError(f'No {prefix} files: {folder}/{prefix}')

    nearest = None
    for path in files:
        with h5py.File(path, mode='r') as file:
            times = np.asarray(file['tasks'][tasks[0]].dims[0]['sim_time'])
            index = int(np.argmin(np.abs(times-target_time)))
            candidate = (abs(times[index]-target_time), path, index)
            nearest = candidate if nearest is None or candidate[0] < nearest[0] else nearest

    _, path, index = nearest
    with h5py.File(path, mode='r') as file:
        snapshot = {task: np.asarray(file['tasks'][task][index]) for task in tasks}
        snapshot['time'] = float(file['tasks'][tasks[0]].dims[0]['sim_time'][index])
    return snapshot


def load_profile_snapshots(folder, tasks, prefix='profile_snapshots', frac=1, back=True, step=1):
    return _load_snapshots(folder, tasks, prefix, frac, back, step)


def load_scalar_snapshots(folder, tasks, prefix='scalar_snapshots', frac=1, back=True, step=1):
    return _load_snapshots(folder, tasks, prefix, frac, back, step)


def _scalar_field_chunk(array):
    singleton_axes = tuple(axis for axis, size in enumerate(array.shape) if axis not in (0, 2, 3) and size == 1)
    return np.squeeze(array, axis=singleton_axes) if singleton_axes else array


def load_scalar_havg(folder, task, prefix='field_snapshots', frac=1, back=True, step=1):
    """Load a scalar field as per-snapshot horizontal-average profiles."""
    profiles, times = [], []
    for chunk in iter_snapshot_chunks(folder, [task], prefix, frac, back, step):
        scalar = _scalar_field_chunk(chunk[task])
        profiles.append(np.mean(scalar, axis=1))
        times.append(chunk['time'])
    return {'time': np.concatenate(times), 'havg': np.concatenate(profiles)}


_VELOCITY_MOMENT_KEYS = (
    'ux_havg', 'w_havg', 'w_sq_havg', 'w_var', 'speed_sq_havg',
    'KE_havg', 'KE_domavg', 'max_Re', 'Re_havg', 'KE_diss_rate', 'w_mid',
)


def _velocity_chunk_moments(velocity, nu, x_grid, z_grid, z_mid):
    ux = velocity[:, 0, :, :]
    w = velocity[:, 1, :, :]
    speed_sq = ux**2 + w**2
    KE_havg = 0.5*np.mean(speed_sq, axis=1)
    Re = np.sqrt(speed_sq)/nu

    dudx = np.gradient(ux, x_grid, axis=1)
    dwdx = np.gradient(w, x_grid, axis=1)
    dudz = np.gradient(ux, z_grid, axis=2)
    dwdz = np.gradient(w, z_grid, axis=2)
    epsilon = nu*(2*dudx**2 + 2*dwdz**2 + (dudz + dwdx)**2)
    mid_idx = np.argmin(np.abs(z_grid-z_mid))

    return {
        'ux_havg': np.mean(ux, axis=1),
        'w_havg': np.mean(w, axis=1),
        'w_sq_havg': np.mean(w**2, axis=1),
        'w_var': np.var(w, axis=1),
        'speed_sq_havg': np.mean(speed_sq, axis=1),
        'KE_havg': KE_havg,
        'KE_domavg': np.mean(KE_havg, axis=1),
        'max_Re': np.max(Re, axis=(1, 2)),
        'Re_havg': np.mean(Re, axis=1),
        'KE_diss_rate': np.mean(epsilon, axis=(1, 2)),
        'w_mid': w[:, :, mid_idx],
    }


def load_velocity_moments(folder, nu, x_grid, z_grid, z_mid, prefix='field_snapshots', frac=1, back=True, step=1):
    """Load the reduced velocity data needed by active field diagnostics."""
    moments = {key: [] for key in _VELOCITY_MOMENT_KEYS}
    times = []
    for chunk in iter_snapshot_chunks(folder, ['velocity'], prefix, frac, back, step):
        chunk_moments = _velocity_chunk_moments(chunk['velocity'], nu, x_grid, z_grid, z_mid)
        for key in moments:
            moments[key].append(chunk_moments[key])
        times.append(chunk['time'])
    return {'time': np.concatenate(times), **{key: np.concatenate(value) for key, value in moments.items()}}


def load_tracer_moments(folder, tracer_task, nu, x_grid, z_grid, z_mid, prefix='field_snapshots', frac=1, back=True, step=1):
    """Load reduced velocity and velocity-tracer moments in one field pass."""
    result = {key: [] for key in (*_VELOCITY_MOMENT_KEYS, 'tracer_havg', 'tracer_flux')}
    times = []
    for chunk in iter_snapshot_chunks(folder, ['velocity', tracer_task], prefix, frac, back, step):
        velocity = chunk['velocity']
        tracer = _scalar_field_chunk(chunk[tracer_task])
        velocity_moments = _velocity_chunk_moments(velocity, nu, x_grid, z_grid, z_mid)
        w = velocity[:, 1, :, :]
        tracer_havg = np.mean(tracer, axis=1)
        w_tracer_havg = np.mean(w*tracer, axis=1)

        for key in _VELOCITY_MOMENT_KEYS:
            result[key].append(velocity_moments[key])
        result['tracer_havg'].append(tracer_havg)
        result['tracer_flux'].append(w_tracer_havg - velocity_moments['w_havg']*tracer_havg)
        times.append(chunk['time'])
    return {'time': np.concatenate(times), **{key: np.concatenate(value) for key, value in result.items()}}


def compute_u_rms_from_moments(speed_sq_havg, z_grid, z_bounds, time_mask):
    """Compute layer RMS speed from streamed horizontal moments and a time mask."""
    speed_sq = np.mean(np.asarray(speed_sq_havg)[time_mask], axis=0)
    z_lower, z_upper = z_bounds
    speed_sq_integral = PchipInterpolator(z_grid, speed_sq).integrate(z_lower, z_upper)
    return np.sqrt(speed_sq_integral / (z_upper - z_lower))

def compute_field_arrays(f, nu, rho0=None):
    ux = f['velocity'][:,0,:,:]
    w = f['velocity'][:,1,:,:]

    if rho0 is None:
        KE = 0.5*(ux**2 + w**2)
        KE_flux = 0.5*w*(ux**2 + w**2)
    else:
        KE = 0.5*rho0*(ux**2 + w**2)
        KE_flux = 0.5*rho0*w*(ux**2 + w**2)

    KE_havg = KE.mean(axis=1)
    KE_domavg = KE.mean(axis=(1,2))

    Re = np.sqrt(ux**2 + w**2)/nu

    return {
        'ux': ux,
        'w': w,
        'KE': KE,
        'KE_havg': KE_havg,
        'KE_domavg': KE_domavg,
        'KE_flux': KE_flux,
        'Re': Re,
        'max_Re': np.max(Re, axis=(1,2))
    }


def compute_and_store_derived(run_folder, info, f, p, field_arrays, u_rms, w_rms, w_rms_interface, Nu, release_idx_C1, release_idx_C2, flow_mask_p, flow_mask_f):
    bg = info['background']
    x_grid, z_grid = np.array(bg['x']), np.array(bg['z'])
    zb = bg['zb']

    cz_mask = z_grid >= zb

    w_rms_cz = w_rms[flow_mask_p][:,cz_mask].mean()

    cz_mid = zb + info['params']['L_cz']/2
    idx = np.argmin(np.abs(z_grid - cz_mid))
    lambda_E = compute_lambda_E(field_arrays['w'][flow_mask_f][:,:,idx], x_grid)
    tau_conv = estimate_tau_conv(lambda_E, u_rms, info['params']['L_cz'])

    w_rms_interface_mean = w_rms_interface[flow_mask_p].mean()

    KE_mean = field_arrays['KE_domavg'][flow_mask_f].mean()

    KE_diss_rate = compute_KE_dissipation_rate_field(bg['nu'], f, x_grid, z_grid).mean(axis=(1,2))
    KE_diss_rate_mean = KE_diss_rate[flow_mask_f].mean()

    ratio, _, _ = compute_KE_ratio(field_arrays['KE'][flow_mask_f], z_grid, zb)
    ratio_mean = ratio.mean()

    Nu_mean = Nu[flow_mask_p].mean()
    if release_idx_C1 is None:
        max_drift_C1, mass_conserved_C1 = np.nan, None
    else:
        max_drift_C1, mass_conserved_C1 = check_mass_conservation(p['C1_total'], release_idx_C1)

    if release_idx_C2 is None:
        max_drift_C2, mass_conserved_C2 = np.nan, None
    else:
        max_drift_C2, mass_conserved_C2 = check_mass_conservation(p['C2_total'], release_idx_C2)

    info['derived'] = {
        'tau_conv': float(tau_conv),
        'w_rms_cz': float(w_rms_cz),
        'w_rms_interface_mean': float(w_rms_interface_mean),
        'Nu_mean': float(Nu_mean),
        'KE_ratio_mean': float(ratio_mean),
        'KE_mean': float(KE_mean),
        'KE_diss_rate_mean': float(KE_diss_rate_mean),
        'max_drift_C1': float(max_drift_C1),
        'mass_conserved_C1': None if mass_conserved_C1 is None else bool(mass_conserved_C1),
        'max_drift_C2': float(max_drift_C2),
        'mass_conserved_C2': None if mass_conserved_C2 is None else bool(mass_conserved_C2)
    }
    info = clean_nans(info)
    write_json_atomic(f'{run_folder}/info.json', info)
    return info['derived']


def compute_and_store_derived_from_moments(run_folder, info, p, moments, u_rms, w_rms, w_rms_interface, Nu, release_idx_C1, release_idx_C2, flow_mask_p, flow_mask_f):
    bg = info['background']
    x_grid, z_grid = np.asarray(bg['x']), np.asarray(bg['z'])
    zb = bg['zb']
    cz_mask = z_grid >= zb

    w_rms_cz = w_rms[flow_mask_p][:, cz_mask].mean()
    lambda_E = compute_lambda_E(moments['w_mid'][flow_mask_f], x_grid)
    tau_conv = estimate_tau_conv(lambda_E, u_rms, info['params']['L_cz'])
    w_rms_interface_mean = w_rms_interface[flow_mask_p].mean()
    KE_mean = moments['KE_domavg'][flow_mask_f].mean()
    KE_diss_rate_mean = moments['KE_diss_rate'][flow_mask_f].mean()
    ratio, _, _ = compute_KE_ratio_from_havg(moments['KE_havg'][flow_mask_f], z_grid, zb)
    ratio_mean = ratio.mean()
    Nu_mean = Nu[flow_mask_p].mean()

    if release_idx_C1 is None:
        max_drift_C1, mass_conserved_C1 = np.nan, None
    else:
        max_drift_C1, mass_conserved_C1 = check_mass_conservation(p['C1_total'], release_idx_C1)
    if release_idx_C2 is None:
        max_drift_C2, mass_conserved_C2 = np.nan, None
    else:
        max_drift_C2, mass_conserved_C2 = check_mass_conservation(p['C2_total'], release_idx_C2)

    info['derived'] = {
        'tau_conv': float(tau_conv),
        'w_rms_cz': float(w_rms_cz),
        'w_rms_interface_mean': float(w_rms_interface_mean),
        'Nu_mean': float(Nu_mean),
        'KE_ratio_mean': float(ratio_mean),
        'KE_mean': float(KE_mean),
        'KE_diss_rate_mean': float(KE_diss_rate_mean),
        'max_drift_C1': float(max_drift_C1),
        'mass_conserved_C1': None if mass_conserved_C1 is None else bool(mass_conserved_C1),
        'max_drift_C2': float(max_drift_C2),
        'mass_conserved_C2': None if mass_conserved_C2 is None else bool(mass_conserved_C2)
    }
    info = clean_nans(info)
    write_json_atomic(f'{run_folder}/info.json', info)
    return info['derived']


def compute_w_rms(w_sq_havg):
    w_sq = np.maximum(w_sq_havg, 0)
    return np.sqrt(w_sq)


def compute_u_rms(u, z_grid, z_bounds):
    ux = u[:,0,:,:]
    w = u[:,1,:,:]
    speed_sq = np.mean(ux**2 + w**2, axis=(0, 1))
    z_lower, z_upper = z_bounds
    speed_sq_integral = PchipInterpolator(z_grid, speed_sq).integrate(z_lower, z_upper)
    return np.sqrt(speed_sq_integral / (z_upper - z_lower))


def compute_w_rms_interface(w_rms, z_grid, zb):
    w_rms = np.asarray(w_rms)
    idx = np.argmax(z_grid > zb)
    return w_rms[:,idx]


def compute_w_variance(w):
    mu = np.mean(w, axis=1, keepdims=True)
    return np.sum((w - mu)**2, axis=1) / w.shape[1]


def compute_KE_ratio(KE, z_grid, zb):
    cz_mask = z_grid >= zb
    rz_mask = z_grid < zb
    KE_cz = KE[:,:,cz_mask].mean(axis=(1,2))
    KE_rz = KE[:,:,rz_mask].mean(axis=(1,2))
    ratio = np.full_like(KE_cz, np.nan, dtype=float)
    np.divide(KE_rz, KE_cz, out=ratio, where=KE_cz != 0)
    return ratio, KE_cz, KE_rz


def compute_KE_ratio_from_havg(KE_havg, z_grid, zb):
    cz_mask = z_grid >= zb
    rz_mask = z_grid < zb
    KE_cz = KE_havg[:, cz_mask].mean(axis=1)
    KE_rz = KE_havg[:, rz_mask].mean(axis=1)
    ratio = np.full_like(KE_cz, np.nan, dtype=float)
    np.divide(KE_rz, KE_cz, out=ratio, where=KE_cz != 0)
    return ratio, KE_cz, KE_rz


def compute_KE_dissipation_rate_field(nu, f, x_grid, z_grid, rho0=None):
    u = f['velocity'][:,0,:,:]
    w = f['velocity'][:,1,:,:]
    dudx = np.gradient(u, x_grid, axis=1)
    dwdx = np.gradient(w, x_grid, axis=1)
    dudz = np.gradient(u, z_grid, axis=2)
    dwdz = np.gradient(w, z_grid, axis=2)
    if rho0 is None:
        epsilon = nu*(2*dudx**2 + 2*dwdz**2 + (dudz + dwdx)**2)
    else:
        epsilon = rho0*nu*(2*dudx**2 + 2*dwdz**2 + (dudz + dwdx)**2)
    return epsilon


def get_boussinesq_cz_reference_gradient(background):
    return background['N2_cz'] / (background.get('g', 1.0) * background.get('alpha', 1.0))


def compute_Nu(p, dT0dz_cz_ref, z_grid, kappa, cz_mask):
    d_dz = np.gradient(p['theta_havg'], z_grid, axis=1)
    F_cond = -kappa*d_dz
    Nu_havg = (p['F_conv_havg'] + F_cond) / (-kappa * dT0dz_cz_ref)
    Nu = Nu_havg[:,cz_mask].mean(axis=1)
    return Nu, Nu_havg, F_cond


def compute_turbulent_thermal_diffusivity(p, z_grid):
    dtheta_dz = np.gradient(p['theta_havg'], z_grid, axis=1)
    return -p['F_conv_havg']/dtheta_dz


def compute_depth_from_var_5pct_frames(w_var_t, z_grid, zb):
    """Per-frame depth below zb where the vertical velocity variance falls to 5% of its maximum."""
    rz = z_grid < zb
    depths = np.full(len(w_var_t), np.nan)
    for i, variance in enumerate(w_var_t):
        threshold = 0.05*np.nanmax(variance)
        roots = PchipInterpolator(z_grid[rz], variance[rz]).solve(y=threshold, extrapolate=False)
        if len(roots):
            depths[i] = zb-np.max(roots)
    return depths


def compute_variance_symmetry_depth(variance_frames, z_grid, zb):
    depths = np.full(len(variance_frames), np.nan)
    stable = z_grid < zb

    for i, variance in enumerate(variance_frames):
        spline = CubicSpline(z_grid, variance)
        peak_index = np.argmax(variance)
        lower = max(0, peak_index-2)
        upper = min(len(z_grid)-1, peak_index+2)
        peak = minimize_scalar(lambda z: -spline(z), bounds=(z_grid[lower], z_grid[upper]), method='bounded').x

        reflected_z = 2*peak-z_grid
        mirrored = np.zeros_like(variance)
        valid = (reflected_z >= z_grid[0]) & (reflected_z <= z_grid[-1])
        mirrored[valid] = spline(reflected_z[valid])

        residual = np.abs(mirrored[stable]-variance[stable])
        residual_peak = np.argmax(residual)
        roots = PchipInterpolator(z_grid[stable], residual).solve(y=0.05*residual[residual_peak], extrapolate=False)
        roots = roots[roots < z_grid[stable][residual_peak]]
        if len(roots):
            depths[i] = zb-np.max(roots)

    return np.nanmean(depths), np.nanstd(depths, ddof=1)


def compute_depth_from_KE(L_cz, z, zb, KE_havg):
    def half_gaussian(z, A, delta):
        return A * np.exp(-0.5 * (z-zb)**2 / delta**2)

    rz = z < zb
    KE_profile = np.mean(KE_havg, axis=0)

    A0 = KE_profile[rz][-1]
    delta0 = 0.1*L_cz

    popt, _ = curve_fit(half_gaussian, z[rz], KE_profile[rz], p0=[A0, delta0], bounds=([0.0, 0.0], [np.inf, np.inf]))

    A, delta = popt
    z_width = zb - delta
    z_5 = zb - np.sqrt(2*np.log(20))*delta
    KE_fit = half_gaussian(z[rz], A, delta)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(z[rz], KE_fit, label='half gaussian')
    ax.plot(z[rz], KE_profile[rz], label='KE profile')
    ax.axvline(z_5, color='gray', linestyle=':')
    ax.text(0.05, 0.7, f'Gaussian width: {z_width:.3f}\n5% KE depth: {z_5:.3f}', transform=ax.transAxes)
    ax.set_ylabel(r'$KE_{\rm RZ}$')
    ax.set_xlabel('Depth')
    ax.legend()
    plt.tight_layout()
    plt.close(fig)

    return fig, z_width, z_5


def compute_depth_from_KE_frames(L_cz, z, zb, KE_havg):
    def half_gaussian(z, A, delta):
        return A*np.exp(-0.5*(z-zb)**2/delta**2)

    rz = z < zb

    d_width_t = np.full(len(KE_havg), np.nan)
    d_5_t = np.full(len(KE_havg), np.nan)

    for i, KE_profile in enumerate(KE_havg):
        A0 = KE_profile[rz][-1]
        delta0 = 0.1*L_cz

        popt, _ = curve_fit(half_gaussian, z[rz], KE_profile[rz], p0=[A0, delta0], bounds=([0.0, 0.0], [np.inf, np.inf]))

        A, delta = popt
        d_width_t[i] = delta

        threshold = 0.05*np.nanmax(KE_profile)
        roots = PchipInterpolator(z[rz], KE_profile[rz]).solve(y=threshold, extrapolate=False)
        if len(roots):
            d_5_t[i] = zb-np.max(roots)

    d_width_mean = np.nanmean(d_width_t)
    d_width_std = np.nanstd(d_width_t, ddof=1)

    d_5_mean = np.nanmean(d_5_t)
    d_5_std = np.nanstd(d_5_t, ddof=1)

    KE_profile = np.mean(KE_havg, axis=0)
    A0 = KE_profile[rz][-1]
    delta0 = 0.1*L_cz

    popt, _ = curve_fit(half_gaussian, z[rz], KE_profile[rz], p0=[A0, delta0], bounds=([0.0, 0.0], [np.inf, np.inf]))

    A, delta = popt
    threshold = 0.05*np.nanmax(KE_profile)
    roots = PchipInterpolator(z[rz], KE_profile[rz]).solve(y=threshold, extrapolate=False)
    d_5_avg_profile = zb-np.max(roots) if len(roots) else np.nan
    KE_fit = half_gaussian(z[rz], A, delta)

    depth = normalized_depth(z, zb, L_cz)
    KE_scale = np.nanmax(KE_profile)
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    shade_layers(ax, depth)
    ax.plot(depth[rz], KE_profile[rz]/KE_scale, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY,
            label='KE profile')
    ax.plot(depth[rz], KE_fit/A, color=COLOR_MODEL, linestyle='--', linewidth=LINEWIDTH_SECONDARY,
            label='Half-Gaussian fit')
    ax.axvline(d_5_avg_profile/L_cz, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='5% depth')
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax.set_ylabel(r'$\widetilde{KE}_{\rm RZ}$')
    ax.set_xlabel('Depth')
    add_reference_legend(ax)
    handles, labels = get_data_legend_handles(ax)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    center_legend_above_axes(fig, ax)
    plt.close(fig)

    return fig, d_width_t, d_width_mean, d_width_std, d_5_t, d_5_mean, d_5_std


def compute_depth_from_KE_flux_frames(KE_flux_havg, z_grid, zb, L_cz):
    rz = z_grid < zb
    z_rz = z_grid[rz]
    KE_flux_rz = KE_flux_havg[:,rz]

    # 5% depth for each frame
    depth_5_t = np.full(len(KE_flux_havg), np.nan)

    for i in range(len(KE_flux_havg)):
        profile_t = KE_flux_rz[i]
        idx_peak = np.argmax(profile_t)
        threshold = 0.05 * profile_t[idx_peak]

        pchip = PchipInterpolator(z_rz, profile_t)
        roots = pchip.solve(y=threshold, extrapolate=False)
        roots_deep = roots[roots < z_rz[idx_peak]]

        if len(roots_deep) > 0:
            depth_5_t[i] = zb - np.max(roots_deep)

    depth_5_mean = np.nanmean(depth_5_t)
    depth_5_std = np.nanstd(depth_5_t, ddof=1)

    # Time-averaged profile
    KE_flux = np.mean(KE_flux_havg, axis=0)
    profile = KE_flux[rz]

    idx_peak = np.argmax(profile)
    threshold = 0.05 * profile[idx_peak]

    pchip = PchipInterpolator(z_rz, profile)
    roots = pchip.solve(y=threshold, extrapolate=False)
    roots_deep = roots[roots < z_rz[idx_peak]]
    z_5 = np.max(roots_deep) if len(roots_deep) > 0 else np.nan

    idx_min = np.argmin(KE_flux)
    z_min = z_grid[idx_min]

    idx_max = np.argmax(profile)
    z_max = z_rz[idx_max]

    depth = normalized_depth(z_grid, zb, L_cz)
    depth_5 = (zb-z_5)/L_cz
    flux_scale = np.abs(profile[idx_peak])
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    shade_layers(ax, depth)
    ax.plot(depth, KE_flux/flux_scale, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax.axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
    ax.axvline(depth_5, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='5% depth')
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax.set_xlabel('Depth')
    ax.set_ylabel(r'$\widetilde{F}_{\rm KE}$')
    add_reference_legend(ax, loc='lower right')
    fig.tight_layout()
    plt.close(fig)

    return fig, depth_5_t, depth_5_mean, depth_5_std


def compute_depth_from_F_conv_frames(F_conv_havg, z_grid, zb, L_cz):
    rz = z_grid < zb
    z_rz = z_grid[rz]
    F_conv_rz = F_conv_havg[:,rz]

    # 5% depth for each frame
    depth_5_t = np.full(len(F_conv_havg), np.nan)

    for i in range(len(F_conv_havg)):
        profile_t = F_conv_rz[i]
        idx_peak = np.argmin(profile_t)
        threshold_t = 0.05 * profile_t[idx_peak]

        pchip = PchipInterpolator(z_rz, profile_t)
        roots = pchip.solve(y=threshold_t, extrapolate=False)
        roots_deep = roots[roots < z_rz[idx_peak]]

        if len(roots_deep) > 0:
            depth_5_t[i] = zb - np.max(roots_deep)

    depth_5_mean = np.nanmean(depth_5_t)
    depth_5_std = np.nanstd(depth_5_t, ddof=1)

    # Time-averaged profile
    F_conv_mean = np.mean(F_conv_havg, axis=0)
    F_conv_mean_rz = F_conv_mean[rz]

    idx_peak = np.argmin(F_conv_mean_rz)
    z_min_profile = z_rz[idx_peak]
    threshold = 0.05 * F_conv_mean_rz[idx_peak]

    pchip = PchipInterpolator(z_rz, F_conv_mean_rz)
    roots = pchip.solve(y=threshold, extrapolate=False)
    roots_deep = roots[roots < z_min_profile]
    z_5 = np.max(roots_deep) if len(roots_deep) > 0 else np.nan

    depth = normalized_depth(z_grid, zb, L_cz)
    depth_5 = (zb-z_5)/L_cz
    flux_scale = np.abs(F_conv_mean_rz[idx_peak])
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    shade_layers(ax, depth)
    ax.plot(depth, F_conv_mean/flux_scale, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax.axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
    ax.axvline(depth_5, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='5% depth')
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax.set_xlabel('Depth')
    ax.set_ylabel(r'$\widetilde{F}_{\rm conv}$')
    add_reference_legend(ax)
    fig.tight_layout()
    plt.close(fig)

    return fig, depth_5_t, depth_5_mean, depth_5_std


def compute_mixing_depth(tracer_flux, tracer_profiles, kappa_C, z_grid, zb, L_cz):
    flux_mean = np.mean(tracer_flux, axis=0)
    tracer_mean = np.mean(tracer_profiles, axis=0)
    gradient_mean = np.gradient(tracer_mean, z_grid)
    threshold = 0.05*np.nanmax(np.abs(gradient_mean))
    valid_gradient = np.abs(gradient_mean) > threshold
    diffusivity = np.full_like(z_grid, np.nan)
    diffusivity[valid_gradient] = -flux_mean[valid_gradient]/gradient_mean[valid_gradient]

    gradients = np.gradient(tracer_profiles, z_grid, axis=1)
    depths = np.full(len(tracer_flux), np.nan)
    stable = z_grid < zb

    for i, (flux, gradient) in enumerate(zip(tracer_flux, gradients)):
        frame_diffusivity = np.full_like(z_grid, np.nan)
        valid_gradient = np.abs(gradient) > 0.05*np.nanmax(np.abs(gradient))
        frame_diffusivity[valid_gradient] = -flux[valid_gradient]/gradient[valid_gradient]
        valid = stable & np.isfinite(frame_diffusivity)
        if np.count_nonzero(valid) >= 2:
            roots = PchipInterpolator(z_grid[valid], frame_diffusivity[valid]).solve(y=kappa_C, extrapolate=False)
            if len(roots):
                depths[i] = zb-np.max(roots)

    measured_depths = depths[np.isfinite(depths)]
    depth_mean = np.mean(measured_depths) if len(measured_depths) else np.nan
    depth_std = np.std(measured_depths, ddof=1) if len(measured_depths) > 1 else np.nan
    depth = normalized_depth(z_grid, zb, L_cz)

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.plot(depth[stable], diffusivity[stable]/kappa_C, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax.axhline(1, color=COLOR_MODEL, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Molecular diffusivity')
    if np.isfinite(depth_mean):
        ax.axvline(depth_mean/L_cz, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Mixing depth')
    ax.set_xlabel('Depth')
    ax.set_ylabel(r'$D_{\rm turb}/\kappa_C$')
    add_reference_legend(ax)
    fig.tight_layout()
    plt.close(fig)

    return fig, depth_mean, depth_std, diffusivity


def plot_depth_diagnostics(w_var, KE_havg, KE_flux_havg, F_conv_havg, Dt_tracer, kappa_C, z, zb, L_cz, depths):
    """Plot the depth-diagnostic profiles with the Table 1 depths marked.

    depths: dict of temporal-mean depths below zb (same units as z), keys
    'var', 'var_symmetry', 'KE', 'KE_flux', 'conv_flux', 'mixing'. Lines are
    drawn at these values rather than recomputed from the time-averaged profiles.
    """
    depth = normalized_depth(z, zb, L_cz)
    rz = z < zb
    z_rz = z[rz]
    fig = plt.figure(figsize=(FIG_DOUBLE, 4.9))
    grid = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.34)
    symmetry_grid = grid[0, 1].subgridspec(2, 1, height_ratios=[3, 1.35], hspace=0.08)
    ax_var = fig.add_subplot(grid[0, 0])
    ax_sym = fig.add_subplot(symmetry_grid[0])
    ax_residual = fig.add_subplot(symmetry_grid[1], sharex=ax_sym)
    ax_ke = fig.add_subplot(grid[0, 2])
    ax_ke_flux = fig.add_subplot(grid[1, 0])
    ax_conv_flux = fig.add_subplot(grid[1, 1])
    ax_mixing = fig.add_subplot(grid[1, 2])
    axes = (ax_var, ax_sym, ax_residual, ax_ke, ax_ke_flux, ax_conv_flux, ax_mixing)

    variance_scale = np.nanmax(w_var)
    ax_var.plot(depth, w_var/variance_scale, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    shade_layers(ax_var, depth)
    boundary_line = ax_var.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    depth_line = ax_var.axvline(depths['var']/L_cz, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Variance depth')
    ax_var.set_ylabel(r'$\sigma_w^2$')

    cs_peak = CubicSpline(z, w_var)
    i_peak = np.argmax(w_var)
    result = minimize_scalar(lambda zz: -cs_peak(zz), bounds=(z[max(0, i_peak-2)], z[min(len(z)-1, i_peak+2)]), method='bounded')
    z_reflected = 2*result.x-z
    w_var_mirror = np.zeros_like(w_var)
    valid_mirror = (z_reflected >= z[0]) & (z_reflected <= z[-1])
    w_var_mirror[valid_mirror] = cs_peak(z_reflected[valid_mirror])
    residual = w_var_mirror-w_var
    residual_depth = depths['var_symmetry']/L_cz
    for ax in (ax_sym, ax_residual):
        shade_layers(ax, depth)
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    # The residual depth is derived from the lower panel, so only mark it there.
    ax_residual.axvline(residual_depth, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Residual depth')
    ax_sym.plot(depth, w_var/variance_scale, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY, label='Original')
    ax_sym.plot(depth, w_var_mirror/variance_scale, color=COLOR_MODEL, linestyle='--', linewidth=LINEWIDTH_SECONDARY, label='Mirrored')
    ax_residual.plot(depth, residual, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax_residual.axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
    ax_sym.set_ylabel(r'$\sigma_w^2$')
    ax_residual.set_ylabel(r'$\Delta\sigma_w^2$', rotation=0, ha='left', va='top')
    ax_residual.yaxis.set_label_coords(0.02, 0.94)
    residual_range = np.nanmax(residual)-np.nanmin(residual)
    ax_residual.set_ylim(np.nanmin(residual)-0.12*residual_range, np.nanmax(residual)+0.8*residual_range)
    handles, labels = get_data_legend_handles(ax_sym)
    ax_sym.legend(handles, labels, loc='upper right', frameon=False, handlelength=1.6, borderaxespad=0.3)
    # Upper variance panel shares x with the residual below: hide its x tick marks and labels.
    ax_sym.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    KE_profile = np.mean(KE_havg, axis=0)
    shade_layers(ax_ke, depth)
    ax_ke.plot(depth, KE_profile/np.nanmax(KE_profile), color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax_ke.axvline(depths['KE']/L_cz, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='5% depth')
    ax_ke.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax_ke.set_ylabel(r'$\mathrm{KE}$')
    y_min, y_max = ax_ke.get_ylim()
    ax_ke.set_ylim(y_min, y_max+0.04*(y_max-y_min))

    def plot_flux(ax, values, peak_kind, ylabel, depth_value):
        profile = np.mean(values, axis=0)
        stable = profile[rz]
        idx_peak = np.argmax(stable) if peak_kind == 'max' else np.argmin(stable)
        shade_layers(ax, depth)
        ax.plot(depth, profile/np.abs(stable[idx_peak]), color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
        ax.axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
        ax.axvline(depth_value/L_cz, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='5% depth')
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
        ax.set_ylabel(ylabel)

    plot_flux(ax_ke_flux, KE_flux_havg, 'max', r'$F_{\mathrm{KE}}$', depths['KE_flux'])
    plot_flux(ax_conv_flux, F_conv_havg, 'min', r'$F_{\mathrm{conv}}$', depths['conv_flux'])
    y_min, y_max = ax_conv_flux.get_ylim()
    ax_conv_flux.set_ylim(y_min, y_max+0.08*(y_max-y_min))

    ax_mixing.plot(depth[rz], Dt_tracer[rz]/kappa_C, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax_mixing.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE)
    ax_mixing.axhline(1, color=COLOR_MODEL, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Molecular diffusivity')
    if np.isfinite(depths['mixing']):
        ax_mixing.axvline(depths['mixing']/L_cz, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE)
    ax_mixing.margins(x=0)  # no padding, so the whole-depth rounding below gives 0 ... 2
    ax_mixing.set_ylabel(r'$D_{\rm turb}/\kappa_C$')
    add_reference_legend(ax_mixing)
    y_min, y_max = ax_mixing.get_ylim()
    ax_mixing.set_ylim(y_min, y_max+0.08*(y_max-y_min))

    for ax in (ax_var, ax_residual, ax_ke, ax_ke_flux, ax_conv_flux, ax_mixing):
        ax.set_xlabel('Depth')
    # Chebyshev grid points stop just inside the domain, so round the limits outwards
    # to whole depths so the outermost ticks (e.g. -1 and 2) are drawn.
    for ax in axes:
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(np.floor(round(x_min, 6)), np.ceil(round(x_max, 6)))
    add_panel_labels((ax_var, ax_sym, ax_ke, ax_ke_flux, ax_conv_flux, ax_mixing))
    fig.legend([boundary_line, depth_line], ['Convective boundary', 'Diagnostic depth'], loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, handlelength=1.8, columnspacing=1.8)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.90, hspace=0.30, wspace=0.34)
    center_legend_above_axes(fig, axes, gap=0.05)
    plt.close(fig)
    return fig, axes

def compute_lambda_E(w, x_grid):
    n = w.shape[1]
    dx = x_grid[1] - x_grid[0]
    w = w - np.mean(w, axis=1, keepdims=True)
    w_fft = np.fft.rfft(w, axis=1)
    spectrum = np.abs(w_fft)**2 / n
    mean_spectrum = np.mean(spectrum, axis=0)
    k = 2*np.pi*np.fft.rfftfreq(n, d=dx)
    idx = 1 + np.argmax(mean_spectrum[1:])
    return 2*np.pi/k[idx]


def estimate_tau_conv(lambda_E, u_rms, L_cz):
    return np.pi*(lambda_E + 2*L_cz)/(4*u_rms)


def check_mass_conservation(C_total, release_idx, threshold=0.005):
    C_total = C_total.ravel()[release_idx:]
    drift_frac = np.abs(C_total - C_total[0]) / np.abs(C_total[0])
    max_drift = np.max(drift_frac)
    return max_drift, max_drift < threshold


def get_release_index(C_total):
    released = np.flatnonzero(np.asarray(C_total).ravel() != 0)
    return int(released[0]) if len(released) else None


def plot_u_components(time, u, z_grid, zb):
    z_rz = z_grid > zb
    ux = u[:,0,:,:][:,:,z_rz].mean(axis=(1, 2))
    w = u[:,1,:,:][:,:,z_rz].mean(axis=(1, 2))
    ux_profile = u[:,0,:,:].mean(axis=(0, 1))
    w_profile = u[:,1,:,:].mean(axis=(0, 1))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 5))
    ax1.plot(time, ux, linewidth=LINEWIDTH_TIMESERIES)
    ax2.plot(time, w, linewidth=LINEWIDTH_TIMESERIES)
    ax3.plot(z_grid, ux_profile, linewidth=LINEWIDTH_TIMESERIES)
    ax4.plot(z_grid, w_profile, linewidth=LINEWIDTH_TIMESERIES)
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\langle u_x\rangle_{\rm CZ}$')
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'$\langle w\rangle_{\rm CZ}$')
    ax3.set_xlabel('Depth')
    ax3.set_ylabel(r'$u_x$')
    ax4.set_xlabel('Depth')
    ax4.set_ylabel(r'$w$')
    plt.tight_layout()
    plt.close(fig)
    return fig, (ax1, ax2, ax3, ax4)


def plot_u_components_from_moments(time, ux_havg, w_havg, z_grid, zb, L_cz, average_mask):
    cz = z_grid > zb
    ux = ux_havg[:, cz].mean(axis=1)
    w = w_havg[:, cz].mean(axis=1)
    ux_profile = ux_havg.mean(axis=0)
    w_profile = w_havg.mean(axis=0)
    depth = normalized_depth(z_grid, zb, L_cz)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(FIG_DOUBLE, FIG_MULTI_HEIGHT))
    ax1.plot(time, ux, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    ax2.plot(time, w, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    interval = time[average_mask]
    interval_patch = ax1.axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    ax2.axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    for ax in (ax1, ax2):
        ax.set_xlabel('Time')
    ax3.plot(depth, ux_profile, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    ax4.plot(depth, w_profile, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    for ax in (ax3, ax4):
        shade_layers(ax, depth)
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
        ax.set_xlabel('Depth')
    ax1.set_ylabel(r'$\langle u_x\rangle_{\rm CZ}$')
    ax2.set_ylabel(r'$\langle w\rangle_{\rm CZ}$')
    ax3.set_ylabel(r'$u_x$')
    ax4.set_ylabel(r'$w$')
    add_panel_labels((ax1, ax2, ax3, ax4))
    fig.legend([interval_patch], ['Averaging interval'], loc='lower center',
               bbox_to_anchor=(0.99, 0.995), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    center_legend_above_axes(fig, (ax1, ax2, ax3, ax4), gap=0.05)
    plt.close(fig)
    return fig, (ax1, ax2, ax3, ax4)


def plot_heat_fluxes(z_grid, zb, L_cz, F_conv_havg, theta_havg, kappa, dT0dz):
    F_conv = F_conv_havg.mean(axis=0)
    theta_avg = theta_havg.mean(axis=0)
    dtheta_dz = np.gradient(theta_avg, z_grid)

    F_diff_pert = -kappa*dtheta_dz
    F_diff_full = -kappa*(dtheta_dz + dT0dz)

    F_tot_pert = F_conv + F_diff_pert
    F_tot_full = F_conv + F_diff_full
    depth = normalized_depth(z_grid, zb, L_cz)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_SINGLE, FIG_MULTI_HEIGHT), sharex=True)
    for ax, F_diff, F_tot, ylabel in zip(
        axes, (F_diff_pert, F_diff_full), (F_tot_pert, F_tot_full),
        (r'$F_{\rm pert}$', r'$F_{\rm full}$'),
    ):
        shade_layers(ax, depth)
        ax.plot(depth, F_conv, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY,
                label='Convective')
        ax.plot(depth, F_diff, color=COLOR_MODEL, linestyle='--', linewidth=LINEWIDTH_SECONDARY,
                label='Diffusive')
        ax.plot(depth, F_tot, color=COLOR_C3, linestyle=':', linewidth=LINEWIDTH_SECONDARY,
                label='Total')
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE,
                   label='Convective boundary')
        ax.set_ylabel(ylabel)
        ax.margins(y=0.12)
    axes[-1].set_xlabel('Depth')
    add_panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    center_legend_above_axes(fig, axes)
    plt.close(fig)
    return fig, tuple(axes)


def plot_scalar_time(s):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,5))
    ax1.semilogy(s['time'], s['w_sq'], linewidth=LINEWIDTH_TIMESERIES)
    ax2.semilogy(s['time'], s['KE'], linewidth=LINEWIDTH_TIMESERIES)
    ax1.set_ylabel(r'$w^2$')
    ax2.set_ylabel(r'$KE$')
    ax1.set_xlabel('Time')
    plt.tight_layout()
    plt.close(fig)
    return fig, (ax1, ax2)


def plot_bg_stratification(z_grid, dT0dz_grid, zb, L_cz, tw=None):
    depth = normalized_depth(z_grid, zb, L_cz)
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.margins(x=0)
    if tw is not None:
        ax.axvspan(-2*tw/L_cz, 2*tw/L_cz, color='#B8A3D6', alpha=0.3, zorder=0.2, label='Transition')
    ax.plot(depth, dT0dz_grid, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax.axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax.set_xlabel('Depth')
    ax.set_ylabel(r'$dT_0/dz$')
    ax.margins(y=0.18)
    x_min, x_max = ax.get_xlim()
    ticks = ax.get_xticks()
    ax.set_xticks(np.r_[x_min, ticks[(ticks > x_min) & (ticks < x_max)], x_max])
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    center_legend_above_axes(fig, ax, gap=0.05)
    plt.close(fig)
    return fig, ax


def plot_profiles(p, w_rms, z_grid, zb, L_cz):
    time = p['time']
    depth = normalized_depth(z_grid, zb, L_cz)
    fig, axes = plt.subplots(3, 1, figsize=(FIG_SINGLE, 6.4), sharex=True)
    images = (
        axes[0].pcolormesh(time, depth, w_rms.T, shading='auto', cmap='viridis'),
        axes[1].pcolormesh(time, depth, p['KE_havg'].T, shading='auto', cmap='viridis'),
        axes[2].pcolormesh(time, depth, p['KE_flux_havg'].T, shading='auto', cmap='RdBu_r'),
    )
    vmax = np.nanmax(np.abs(p['KE_flux_havg']))
    images[2].set_clim(-vmax, vmax)
    for image, ax, label in zip(images, axes, (r'$w_{\rm rms}$', r'$KE$', r'$F_{\rm KE}$')):
        fig.colorbar(image, ax=ax, label=label)
        ax.axhline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
        ax.set_ylabel('Depth')
        ax.invert_yaxis()
    axes[-1].set_xlabel('Time')
    add_reference_legend(axes[0])
    add_panel_labels(axes)
    fig.tight_layout()
    plt.close(fig)
    return fig, tuple(axes)


def plot_w_variance(w_var, z_grid, zb, L_cz, var_depth_mean, var_depth_std):
    depth = normalized_depth(z_grid, zb, L_cz)
    variance_scale = np.nanmax(w_var)
    depth_mean = var_depth_mean/L_cz
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    shade_layers(ax, depth)
    ax.plot(depth, w_var/variance_scale, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE,
               label='Convective boundary')
    ax.axvline(depth_mean, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Variance depth')
    ax.set_xlabel('Depth')
    ax.set_ylabel(r'$\widetilde{\sigma}_w^2$')
    add_reference_legend(ax)
    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def plot_w_variance_symmetry(Lz, L_cz, w_var, z, zb):
    cs_peak = CubicSpline(z, w_var)
    i_peak = np.argmax(w_var)

    i0 = max(0, i_peak-2)
    i1 = min(len(z)-1, i_peak+2)

    result = minimize_scalar(lambda zz: -cs_peak(zz), bounds=(z[i0], z[i1]), method='bounded')
    peak = result.x

    z_reflected = 2*peak - z
    w_var_mirror = np.zeros_like(w_var)

    valid = (z_reflected >= z[0]) & (z_reflected <= z[-1])
    w_var_mirror[valid] = cs_peak(z_reflected[valid])

    residual = w_var_mirror - w_var

    # 10% loc
    rz = z < zb
    z_rz = z[rz]
    residual_rz = np.abs(residual[rz])
    idx_peak = np.argmax(residual_rz)
    threshold = 0.05*residual_rz[idx_peak]

    pchip = PchipInterpolator(z_rz, residual_rz)
    roots = pchip.solve(y=threshold, extrapolate=False)
    roots_deep = roots[roots < z_rz[idx_peak]]
    z_residual_5 = np.max(roots_deep) if len(roots_deep) > 0 else np.nan
    depth_residual_5 = zb - z_residual_5 if np.isfinite(z_residual_5) else np.nan

    depth = normalized_depth(z, zb, L_cz)
    variance_scale = np.nanmax(w_var)
    depth_residual_normalized = depth_residual_5/L_cz
    fig, axes = plt.subplots(2, 1, figsize=(FIG_SINGLE, 4.2), height_ratios=[3, 1], sharex=True)
    for ax in axes:
        shade_layers(ax, depth)
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
        ax.axvline(depth_residual_normalized, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Residual depth')
    axes[0].plot(depth, w_var/variance_scale, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY,
                 label='Original')
    axes[0].plot(depth, w_var_mirror/variance_scale, color=COLOR_MODEL, linestyle='--',
                 linewidth=LINEWIDTH_SECONDARY, label='Mirrored')
    axes[1].plot(depth, residual, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    axes[1].axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
    axes[0].set_ylabel(r'$\widetilde{\sigma}_w^2$')
    axes[1].set_ylabel(r'$\Delta\sigma_w^2$')
    axes[1].set_xlabel('Depth')
    residual_min, residual_max = np.nanmin(residual), np.nanmax(residual)
    residual_range = residual_max-residual_min
    axes[1].set_ylim(residual_min-0.12*residual_range, residual_max+0.55*residual_range)
    add_panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    artists = dict(zip(labels, handles))
    legend_order = ('Original', 'Convective boundary', 'Mirrored', 'Residual depth')
    axes[0].legend([artists[label] for label in legend_order], legend_order, loc='lower right',
                   bbox_to_anchor=(1, 1), frameon=False, ncol=2, handlelength=1.6,
                   columnspacing=1.2, borderaxespad=0.15)
    fig.tight_layout(h_pad=0.4)
    fig.subplots_adjust(hspace=0.10)
    plt.close(fig)
    return fig, tuple(axes)


def plot_w_rms_interface(p, w_rms_interface, z_grid, zb, average_mask=None):
    time = p['time']
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.plot(time, w_rms_interface, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    if average_mask is not None and np.any(average_mask):
        interval = time[average_mask]
        interval_patch = ax.axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
        ax.legend([interval_patch], ['Averaging interval'], loc='lower right',
                  bbox_to_anchor=(1, 1), frameon=False, borderaxespad=0.15)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$w_{\rm rms}(z_b)$')
    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def plot_diagnostics(info, f, p, arr, cz, rz, w_rms_interface, flow_mask_f, flow_mask_p):
    idx = np.argmax(np.asarray(info['background']['z']) > np.asarray(info['background']['zb']))
    time_f = f['time']
    time_p = p['time']
    interval_f = time_f[flow_mask_f]
    interval_p = time_p[flow_mask_p]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(FIG_DOUBLE, FIG_MULTI_HEIGHT))
    ax1.plot(time_f, arr['KE_domavg'], color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    ax2.plot(time_f, cz, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES, label='Convective layer')
    ax2.plot(time_f, rz, color=COLOR_MODEL, linestyle='--', linewidth=LINEWIDTH_TIMESERIES,
             label='Stable layer')
    ax3.plot(time_p, p['F_conv_havg'][:, idx], color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    ax4.plot(time_p, w_rms_interface, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    interval_patch = ax1.axvspan(interval_f[0], interval_f[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    ax2.axvspan(interval_f[0], interval_f[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    for ax in (ax3, ax4):
        ax.axvspan(interval_p[0], interval_p[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel('Time')
        ax.margins(y=0.15)
    ax1.set_ylabel(r'$\langle KE\rangle$')
    ax2.set_ylabel(r'$\langle KE\rangle$')
    ax3.set_ylabel(r'$F_{\rm conv}(z_b)$')
    ax4.set_ylabel(r'$w_{\rm rms}(z_b)$')
    add_panel_labels((ax1, ax2, ax3, ax4))
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend([*handles, interval_patch], [*labels, 'Averaging interval'], loc='lower center',
               bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    center_legend_above_axes(fig, (ax1, ax2, ax3, ax4), gap=0.05)
    plt.close(fig)
    return fig



def plot_stationarity(time_f, Re_cz, time_p, flow_mask_f, flow_mask_p, w_sq_havg, z_grid, zb, L_cz):
    interval_f = time_f[flow_mask_f]
    interval_p = time_p[flow_mask_p]
    variance = PchipInterpolator(z_grid, w_sq_havg, axis=1)
    w_rms_cz = np.sqrt(variance.integrate(zb, zb+L_cz)/L_cz)
    w_rms_rz = np.sqrt(variance.integrate(0, zb)/zb)
    w_rms_all = np.sqrt(variance.integrate(0, zb+L_cz)/(zb+L_cz))

    fig, axes = plt.subplots(2, 1, figsize=(FIG_SINGLE, 4.6), sharex=True)
    axes[0].plot(time_f, Re_cz, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    axes[1].plot(time_p, w_rms_cz, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES, label='Convective layer')
    axes[1].plot(time_p, w_rms_rz, color=COLOR_MODEL, linewidth=LINEWIDTH_TIMESERIES, label='Stable layer')
    axes[1].plot(time_p, w_rms_all, color=COLOR_C3, linewidth=LINEWIDTH_TIMESERIES, label='Overall')

    interval_patch = axes[0].axvspan(
        interval_f[0], interval_f[-1], color=COLOR_AVERAGING_INTERVAL,
        alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    axes[1].axvspan(interval_p[0], interval_p[-1], color=COLOR_AVERAGING_INTERVAL,
                   alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    # Axes start exactly at t=0 and at zero (Re and w_rms are non-negative); no padding at the end time.
    axes[0].set_xlim(0, max(time_f[-1], time_p[-1]))
    axes[0].set_ylim(0, 1.08*np.nanmax(Re_cz))
    _, w_max = axes[1].get_ylim()
    axes[1].set_ylim(0, 1.08*w_max)

    # Label the two regimes just above panel (a), centred over each time range and sharing a baseline.
    regime_transform = axes[0].get_xaxis_transform()
    regime_texts = [
        axes[0].text(interval_f[0]/2, 1.03, 'Initial transient', transform=regime_transform,
                     ha='center', va='bottom', fontsize=8, clip_on=False),
        # Centred over the averaging interval.
        axes[0].text((interval_f[0]+interval_f[-1])/2, 1.03, 'Statistically\nstationary regime',
                     transform=regime_transform, ha='center', va='bottom', multialignment='center',
                     fontsize=8, linespacing=1.1, clip_on=False),
    ]

    axes[0].set_ylabel(r'$\mathrm{Re}_{\rm CZ}$')
    axes[1].set_ylabel(r'$w_{\rm rms}$')
    axes[-1].set_xlabel('Time')
    add_panel_labels(axes)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend([*handles, interval_patch], [*labels, 'Averaging interval'], loc='lower center',
               bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    # Place the legend above the regime labels rather than directly above the axes.
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    positions = [ax.get_position() for ax in axes]
    center = (min(pos.x0 for pos in positions)+max(pos.x1 for pos in positions))/2
    labels_top = max(text.get_window_extent(renderer).y1 for text in regime_texts)/fig.bbox.height
    fig.legends[-1].set_bbox_to_anchor((center, labels_top+0.05/fig.get_figheight()), transform=fig.transFigure)
    plt.close(fig)
    return fig, tuple(axes)

def plot_Re_KE(time, max_Re, KE_domavg, average_mask):
    interval = time[average_mask]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_SINGLE, 4.0), sharex=True)
    axes[0].plot(time, max_Re, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    axes[1].plot(time, KE_domavg, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    interval_patch = axes[0].axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    axes[1].axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    for ax in axes:
        ax.margins(y=0.15)
    axes[-1].set_xlabel('Time')
    axes[0].set_ylabel(r'$\mathrm{Re}_{\max}$')
    axes[1].set_ylabel(r'$\langle KE\rangle$')
    add_panel_labels(axes)
    fig.legend([interval_patch], ['Averaging interval'], loc='lower center',
               bbox_to_anchor=(0.99, 0.995), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    center_legend_above_axes(fig, axes, gap=0.05)
    plt.close(fig)
    return fig, tuple(axes)


def plot_Re_cz(time, Re, z_grid, zb, mask):
    cz_mask = z_grid >= zb
    Re_cz = Re[:, :, cz_mask]
    Re_cz_t = Re_cz.mean(axis=(1, 2))
    Re_z = Re[mask].mean(axis=(0, 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,5))
    ax1.plot(time, Re_cz_t, linewidth=LINEWIDTH_TIMESERIES)
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\mathrm{Re}_{\rm CZ}$')
    ax2.plot(z_grid, Re_z, linewidth=LINEWIDTH_TIMESERIES)
    ax2.axvline(zb, c='r', ls='--')
    ax2.set_xlabel('Depth')
    ax2.set_ylabel(r'$\langle\mathrm{Re}\rangle_t$')
    plt.tight_layout()
    plt.close(fig)

    return fig, (ax1, ax2)


def plot_Re_cz_from_moments(time, Re_havg, z_grid, zb, L_cz, mask):
    cz_mask = z_grid >= zb
    Re_cz_t = Re_havg[:, cz_mask].mean(axis=1)
    Re_z = Re_havg[mask].mean(axis=0)
    depth = normalized_depth(z_grid, zb, L_cz)
    interval = time[mask]

    fig, axes = plt.subplots(2, 1, figsize=(FIG_SINGLE, 4.2))
    axes[0].plot(time, Re_cz_t, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    interval_patch = axes[0].axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$\mathrm{Re}_{\rm CZ}$')
    re_range = np.nanmax(Re_cz_t)-np.nanmin(Re_cz_t)
    axes[0].set_ylim(axes[0].get_ylim()[0], np.nanmax(Re_cz_t)+0.25*re_range)
    shade_layers(axes[1], depth)
    axes[1].plot(depth, Re_z, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    axes[1].axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    axes[1].set_xlabel('Depth')
    axes[1].set_ylabel(r'$\langle\mathrm{Re}\rangle_t$')
    add_reference_legend(axes[1])
    add_panel_labels(axes)
    fig.legend([interval_patch], ['Averaging interval'], loc='lower center',
               bbox_to_anchor=(0.99, 0.995), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    center_legend_above_axes(fig, axes, gap=0.05)
    plt.close(fig)
    return fig, tuple(axes)


def plot_KE_ratio(time, ratio, average_mask):
    interval = time[average_mask]
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.plot(time, ratio, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    interval_patch = ax.axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$R_{\rm KE}$')
    ax.legend([interval_patch], ['Averaging interval'], loc='lower right',
              bbox_to_anchor=(1, 1), frameon=False, borderaxespad=0.15)
    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def plot_KE(time, cz, rz, average_mask):
    interval = time[average_mask]
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.plot(time, cz, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES,
            label='Convective layer')
    ax.plot(time, rz, color=COLOR_MODEL, linestyle='--', linewidth=LINEWIDTH_TIMESERIES,
            label='Stable layer')
    interval_patch = ax.axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle KE\rangle$')
    handles, labels = get_data_legend_handles(ax)
    fig.legend([*handles, interval_patch], [*labels, 'Averaging interval'], loc='lower center',
               bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    center_legend_above_axes(fig, ax, gap=0.05)
    plt.close(fig)
    return fig, ax


def plot_KE_dissipation_rate(time, KE_diss_rate, average_mask):
    interval = time[average_mask]
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.plot(time, KE_diss_rate, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    interval_patch = ax.axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\epsilon_{\rm KE}$')
    ax.legend([interval_patch], ['Averaging interval'], loc='lower right',
              bbox_to_anchor=(1, 1), frameon=False, borderaxespad=0.15)
    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def plot_Nu(time, Nu, average_mask=None):
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.plot(time, Nu, color=COLOR_SIMULATION, linewidth=LINEWIDTH_TIMESERIES)
    if average_mask is not None and np.any(average_mask):
        interval = time[average_mask]
        interval_patch = ax.axvspan(interval[0], interval[-1], color=COLOR_AVERAGING_INTERVAL, alpha=AVERAGING_INTERVAL_ALPHA, zorder=0)
        ax.legend([interval_patch], ['Averaging interval'], loc='lower right',
                  bbox_to_anchor=(1, 1), frameon=False, borderaxespad=0.15)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\mathrm{Nu}_{\rm CZ}$')
    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def plot_wT(s):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4,4))
    ax1.plot(s['time'], s['wT'])
    ax2.plot(s['time'], s['T'])
    ax1.set_ylabel(r'$\langle wT\rangle$')
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'$\langle T\rangle$')
    plt.tight_layout()
    plt.close(fig)
    return fig, (ax1, ax2)


def plot_field_snapshot(x_grid, z_grid, zb, time, velocity, theta, tracer=None, tracer_name='C1'):
    # Each panel: data, label, colormap, colour limits (None = symmetric about zero from the data).
    panels = [
        (np.asarray(velocity)[1], r'$w$', 'PiYG', None),
        (np.squeeze(theta), r'$\theta$', 'RdBu_r', (-1.0, 1.0)),
    ]
    if tracer is not None:
        panels.append((np.squeeze(tracer), '$' + tracer_name + '$', 'viridis', (0.0, 1.0)))

    # Periodic in x: append the x=0 column at x=Lx so the plotted field spans the full domain [0, Lx].
    x_grid = np.asarray(x_grid)
    Lx = x_grid[-1] + (x_grid[1]-x_grid[0])
    x_plot = np.append(x_grid, Lx)

    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_DOUBLE, 1.9*len(panels)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    boundary = None
    for ax, (data, label, cmap, clim) in zip(axes, panels):
        data_plot = np.concatenate([data, data[:1]], axis=0)
        if clim is None:
            limit = np.nanmax(np.abs(data))
            clim = (-limit, limit)
        vmin, vmax = clim
        # Values outside [vmin, vmax] saturate to the end colours; colour bars have no extend triangles.
        image = ax.pcolormesh(x_plot, z_grid, data_plot.T, shading='auto', cmap=cmap,
                              vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(image, ax=ax, label=label, pad=0.02)
        ax.set_xlim(0, Lx)
        ax.set_xticks(np.arange(0, np.floor(Lx)+1))
        line = ax.axhline(zb, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE,
                          label='Convective boundary')
        boundary = line if boundary is None else boundary
        ax.set_ylabel(r'$z$')
        ax.set_box_aspect(3/5)

    axes[-1].set_xlabel(r'$x$')
    add_panel_labels(axes)
    axes[0].text(0, 1.035, rf'$t={time:.1f}$', transform=axes[0].transAxes, ha='left', va='bottom')
    axes[0].plot([0.56, 0.64], [1.07, 1.07], transform=axes[0].transAxes, color=COLOR_BOUNDARY,
                 linestyle='--', linewidth=LINEWIDTH_REFERENCE, clip_on=False)
    axes[0].text(0.67, 1.07, 'Convective boundary', transform=axes[0].transAxes, ha='left', va='center')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.close(fig)
    return fig, tuple(axes)


def animate_fields(panels, x, z, time, save_path=None, fps=20, dpi=100):
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(6,3*n))
    if n == 1:
        axes = [axes]

    ims = []
    for ax, panel in zip(axes, panels):
        data = panel['data']
        im = ax.pcolormesh(x.ravel(), z.ravel(), data[0].T, shading='gouraud', cmap=panel.get('cmap', 'RdBu_r'))
        fig.colorbar(im, ax=ax, label=panel['label'])
        ax.set_xlabel('Horizontal position')
        ax.set_ylabel('Depth')

        tail = data[len(data) // 2:]
        if panel.get('clim', 'symmetric') == 'symmetric':
            vmax = np.abs(tail).max()
            im.set_clim(-vmax, vmax)
        else:
            im.set_clim(tail.min(), tail.max())
        ims.append(im)

    plt.tight_layout()

    def update(frame):
        for im, panel in zip(ims, panels):
            im.set_array(panel['data'][frame].T.ravel())
        fig.suptitle(f't = {time[frame]:.2f}', fontsize=10, y=1)
        return ims

    ani = FuncAnimation(fig, update, frames=len(time), interval=80, blit=True)
    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close(fig)

    return fig, ani


# ------------------------- TRACER DIAGNOSTICS ----------------------------- #


def plot_C_diff(C, x_grid, z_grid, zb, L_cz, tracer_name):
    C_diff = C[-1] - C[0]
    C_diff_havg = C_diff.mean(axis=0)
    depth = normalized_depth(z_grid.ravel(), zb, L_cz)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(FIG_SINGLE, FIG_MULTI_HEIGHT))
    im1 = ax1.pcolormesh(x_grid.ravel(), depth, C_diff.T, shading='gouraud', cmap='RdBu_r')
    vmax_cdiff = np.abs(C_diff).max()
    im1.set_clim(-vmax_cdiff, vmax_cdiff)
    fig.colorbar(im1, ax=ax1, label=r'$\Delta C$')
    ax2.plot(depth, C_diff_havg, color=COLOR_SIMULATION, linewidth=LINEWIDTH_PRIMARY)
    shade_layers(ax2, depth)
    ax2.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE)
    im3 = ax3.pcolormesh(x_grid.ravel(), depth, C[-1].T, shading='gouraud', cmap='viridis')
    im4 = ax4.pcolormesh(x_grid.ravel(), depth, C[0].T, shading='gouraud', cmap='viridis')
    fig.colorbar(im3, ax=ax3, label=r'$C$')
    fig.colorbar(im4, ax=ax4, label=r'$C$')
    for ax in (ax1, ax3, ax4):
        ax.axhline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
        ax.invert_yaxis()
        ax.set_xlabel('Horizontal position')
        ax.set_ylabel('Depth')
    ax2.set_xlabel('Depth')
    ax2.set_ylabel(r'$\Delta C$')
    add_reference_legend(ax2)
    add_panel_labels((ax1, ax2, ax3, ax4))
    fig.tight_layout()
    plt.close(fig)
    return fig, (ax1, ax2)


def plot_C(p, z, zb, L_cz, t, tracer_name):
    idx = np.argmin(np.abs(p['time'] - t))
    concentration = p[f'{tracer_name}_havg'][idx]
    dCdz = np.gradient(concentration, z)
    depth = normalized_depth(z, zb, L_cz)
    color = {'C1': COLOR_C1, 'C2': COLOR_C2, 'C3': COLOR_C3}[tracer_name]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_SINGLE, FIG_MULTI_HEIGHT), sharex=True)
    ax1.plot(depth, concentration, color=color, linewidth=LINEWIDTH_PRIMARY)
    ax2.plot(depth, dCdz, color=color, linewidth=LINEWIDTH_PRIMARY)
    for ax in (ax1, ax2):
        shade_layers(ax, depth)
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax1.text(0.98, 0.96, rf'$t={p["time"][idx]:.1f}$', transform=ax1.transAxes, ha='right', va='top')
    ax1.set_ylabel(r'$C$')
    ax2.set_xlabel('Depth')
    ax2.set_ylabel(r'$\partial C/\partial z$')
    add_reference_legend(ax1)
    add_panel_labels((ax1, ax2))
    fig.tight_layout()
    plt.close(fig)
    return fig, (ax1, ax2)


def tracer_plots(z_grid, C_avg, F_C_avg, dC_dz_avg, Dt_tracer, zb, L_cz, kappa_C):
    depth = normalized_depth(z_grid, zb, L_cz)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_DOUBLE, FIG_MULTI_HEIGHT), sharex=True)
    ax1, ax2, ax3, ax4 = axes.ravel()
    ax1.plot(depth, C_avg, color=COLOR_C1, linewidth=LINEWIDTH_PRIMARY)
    ax2.plot(depth, F_C_avg, color=COLOR_C1, linewidth=LINEWIDTH_PRIMARY)
    ax3.plot(depth, dC_dz_avg, color=COLOR_C1, linewidth=LINEWIDTH_PRIMARY)
    ax4.plot(depth, Dt_tracer/kappa_C, color=COLOR_C1, linewidth=LINEWIDTH_PRIMARY)
    ax4.axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
    ax4.axhline(1, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE)
    for ax in axes.ravel():
        shade_layers(ax, depth)
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
        ax.set_xlabel('Depth')
    ax1.set_ylabel(r'$C$')
    ax2.set_ylabel(r'$F_C$')
    ax3.set_ylabel(r'$\partial C/\partial z$')
    ax4.set_ylabel(r'$D_{\rm turb}/\kappa_C$')
    add_reference_legend(ax4)
    add_panel_labels(axes)
    fig.tight_layout()
    plt.close(fig)
    return fig, ((ax1, ax2), (ax3, ax4))


def plot_tracer_profiles_times(time, C1, w, z_grid, zb, L_cz, times, kappa_C, tracer_name):
    C1_havg = C1.mean(axis=1)
    C1_prime = C1 - C1_havg[:, None, :]
    F_C1 = (w*C1_prime).mean(axis=1)
    dC1dz = np.gradient(C1_havg, z_grid, axis=1)

    Dt_tracer_C1 = np.full_like(F_C1, np.nan)
    grad_threshold_1 = 0.05*np.nanmax(np.abs(dC1dz), axis=1, keepdims=True)
    valid_1 = np.abs(dC1dz) > grad_threshold_1
    Dt_tracer_C1[valid_1] = -F_C1[valid_1]/dC1dz[valid_1]

    depth = normalized_depth(z_grid, zb, L_cz)
    base = np.asarray(matplotlib.colors.to_rgb({'C1': COLOR_C1, 'C2': COLOR_C2, 'C3': COLOR_C3}[tracer_name]))
    shades = np.linspace(0.4, 1.0, len(times))
    colors = [tuple(1-(1-base)*shade) for shade in shades]
    linestyles = (':', '-.', '--', '-')
    fig, axes = plt.subplots(3, 1, figsize=(FIG_SINGLE, FIG_MULTI_HEIGHT), sharex=True)
    ax1, ax2, ax3 = axes
    for index, t in enumerate(times):
        i = np.argmin(np.abs(time - t))
        label = rf'$t={time[i]:.1f}$'
        style = linestyles[index % len(linestyles)]
        ax1.plot(depth, dC1dz[i], color=colors[index], linestyle=style, linewidth=LINEWIDTH_PRIMARY, label=label)
        ax2.plot(depth, F_C1[i], color=colors[index], linestyle=style, linewidth=LINEWIDTH_PRIMARY, label=label)
        ax3.plot(depth, Dt_tracer_C1[i]/kappa_C, color=colors[index], linestyle=style, linewidth=LINEWIDTH_PRIMARY, label=label)
    ax3.axhline(1, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE)
    for ax in axes:
        shade_layers(ax, depth)
        ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax1.set_ylabel(r'$\partial C/\partial z$')
    ax2.set_ylabel(r'$F_C$')
    ax3.set_ylabel(r'$D_{\rm turb}/\kappa_C$')
    ax3.set_xlabel('Depth')
    add_reference_legend(ax3)
    handles, labels = get_data_legend_handles(ax1)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    add_panel_labels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    center_legend_above_axes(fig, axes)
    plt.close(fig)
    return fig, (ax1, ax2, ax3)


# --------------- turbulent diffusivity calculations ------------ #

def compute_turbulent_flux(C1, w):
    C1_havg = C1.mean(axis=1)
    C1_prime = C1 - C1_havg[:, None, :]

    F_C1 = (w * C1_prime).mean(axis=1)
    return F_C1, C1_havg


def compute_turbulent_tracer_diffusivity(F_C1, C1_havg, z_grid, zb):
    F_C1_avg = F_C1.mean(axis=0)
    C1_avg = C1_havg.mean(axis=0)
    dC1dz_avg = np.gradient(C1_avg, z_grid)

    Dt_tracer_C1 = np.full_like(z_grid, np.nan)
    grad_threshold_1 = 0.05*np.nanmax(abs(dC1dz_avg))
    valid_1 = (abs(dC1dz_avg) > grad_threshold_1)
    Dt_tracer_C1[valid_1] = -F_C1_avg[valid_1]/dC1dz_avg[valid_1]

    return F_C1_avg, C1_avg, dC1dz_avg, Dt_tracer_C1


def compute_turbulent_tracer_diffusivity_mean_after(F_C1, C1_havg, z_grid, zb):
    dC1dz = np.gradient(C1_havg, z_grid, axis=1)
    Dt_C1 = np.full_like(F_C1, np.nan)

    threshold_1 = 0.05 * np.nanmax(np.abs(dC1dz))
    valid_1 = np.abs(dC1dz) > threshold_1

    Dt_C1[valid_1] = -F_C1[valid_1] / dC1dz[valid_1]

    Dt_C1_mean = np.nanmean(Dt_C1, axis=0)
    Dt_C1_std = np.nanstd(Dt_C1, axis=0)

    return Dt_C1_mean, Dt_C1_std, Dt_C1


def plot_turbulent_tracer_diffusivity_comparison(Dt_tracer_C1, Dt_C1_mean, Dt_C1_std, z_grid, zb, L_cz, kappa_C):
    depth = normalized_depth(z_grid, zb, L_cz)
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    ax.plot(depth, Dt_C1_mean/kappa_C, color=COLOR_C1, linewidth=LINEWIDTH_PRIMARY, label='Mean instantaneous estimate')
    ax.plot(depth, Dt_tracer_C1/kappa_C, color=COLOR_SIMULATION, linestyle='--', linewidth=LINEWIDTH_SECONDARY, label='Time-averaged flux-gradient diagnostic')
    ax.fill_between(depth, (Dt_C1_mean-Dt_C1_std)/kappa_C, (Dt_C1_mean+Dt_C1_std)/kappa_C, color=COLOR_C1, alpha=0.2)
    shade_layers(ax, depth)
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax.axhline(1, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Molecular diffusivity')
    ax.set_xlabel('Depth')
    ax.set_ylabel(r'$D_{\rm turb}/\kappa_C$')
    add_reference_legend(ax)
    handles, labels = get_data_legend_handles(ax)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    center_legend_above_axes(fig, ax)
    plt.close(fig)
    return fig, ax


def plot_Dt_time_windows(time, C1, w, z_grid, zb, L_cz, release_time, kappa_C, windows=None):
    if windows is None:
        windows = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    t_end = time[-1]
    depth = normalized_depth(z_grid, zb, L_cz)
    base = np.asarray(matplotlib.colors.to_rgb(COLOR_C1))
    shades = np.linspace(0.4, 1.0, len(windows))
    linestyles = (':', '-.', '--', '-')

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, FIG_HEIGHT))
    for index, (a, b) in enumerate(windows):
        t1 = release_time + a*(t_end-release_time)
        t2 = release_time + b*(t_end-release_time)
        mask = (time >= t1) & (time < t2)

        C1_window = C1[mask]
        w_window = w[mask]
        C1_havg = C1_window.mean(axis=1)
        C1_prime = C1_window - C1_havg[:, None, :]
        F_C1 = (w_window*C1_prime).mean(axis=1)

        F_C1_avg = F_C1.mean(axis=0)
        C1_avg = C1_havg.mean(axis=0)
        dC1dz_avg = np.gradient(C1_avg, z_grid)

        Dt = np.full_like(z_grid, np.nan)
        threshold = 0.05*np.nanmax(np.abs(dC1dz_avg))
        valid = np.abs(dC1dz_avg) > threshold
        Dt[valid] = -F_C1_avg[valid]/dC1dz_avg[valid]

        color = tuple(1-(1-base)*shades[index])
        label = rf'${t1:.1f}\leq t<{t2:.1f}$'
        ax.plot(depth, Dt/kappa_C, color=color, linestyle=linestyles[index % len(linestyles)], linewidth=LINEWIDTH_PRIMARY, label=label)

    shade_layers(ax, depth)
    ax.axvline(0, color=COLOR_BOUNDARY, linestyle='--', linewidth=LINEWIDTH_REFERENCE, label='Convective boundary')
    ax.axhline(0, color='black', linewidth=LINEWIDTH_REFERENCE, zorder=0.5)
    ax.axhline(1, color=COLOR_THRESHOLD, linestyle=':', linewidth=LINEWIDTH_REFERENCE, label='Molecular diffusivity')
    ax.set_xlabel('Depth')
    ax.set_ylabel(r'$D_{\rm turb}/\kappa_C$')
    add_reference_legend(ax)
    handles, labels = get_data_legend_handles(ax)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    center_legend_above_axes(fig, ax)
    plt.close(fig)
    return fig, ax
