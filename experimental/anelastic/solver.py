import argparse
import time
from pathlib import Path

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

from scipy.integrate import solve_ivp
from metadata_io import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]

'''
nondimensionalization
L -> L_cz
u -> U_ff
t -> L_cz/U_ff
'''


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)

    p.add_argument('--Lx', type=float, default=4.0)
    p.add_argument('--Lz', type=float, default=3.0)
    p.add_argument('--L-cz', type=float, default=1.0)
    p.add_argument('--Nx', type=int, default=512)
    p.add_argument('--Nz', type=int, default=256)
    p.add_argument('--dealias', type=float, default=1.5)

    p.add_argument('--Ra', type=float, default=1e6)
    p.add_argument('--Pr', type=float, default=1.0)
    p.add_argument('--S', type=float, default=4.0)
    p.add_argument('--tw', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=5/3)
    p.add_argument('--eps-s', type=float, default=1e-3)
    p.add_argument('--Di', type=float, default=0.1)

    p.add_argument('--max-timestep', type=float, default=0.1)
    p.add_argument('--stop-sim-time', type=float, required=True)
    p.add_argument('--n-cores', '--n', dest='n_cores', type=int, default=8)

    return p.parse_args()


def get_run_folder(args):
    base = PROJECT_ROOT / 'data' / f'{args.mode}s'
    folder = f'{args.run_id}_Ra{args.Ra:.0e}_S{args.S:g}_Lcz{args.L_cz:g}_{args.Nx}x{args.Nz}'
    return base / folder


def write_info(run_folder, args):
    L_rz = args.Lz - args.L_cz

    info = {
        'run_id': args.run_id,
        'params': {
            'L_cz': args.L_cz,
            'L_rz': L_rz,
            'Lx': args.Lx,
            'Lz': args.Lz,
            'Nx': args.Nx,
            'Nz': args.Nz,
            'dealias': args.dealias,
            'Ra': args.Ra,
            'Pr': args.Pr,
            'eps_s': args.eps_s,
            'Di': args.Di,
            'S': args.S,
            'tw': args.tw,
            'max_timestep': args.max_timestep
        },
        'n_cores': args.n_cores,
        'stop_sim_time': args.stop_sim_time,
        'outcome': None,
        'timing': None
    }

    run_folder.mkdir(parents=True, exist_ok=True)
    write_json_atomic(f'{run_folder}/info.json', info)
    return info


def main():
    args = parse_args()
    run_folder = get_run_folder(args)

    # Parameters
    Lx = args.Lx
    Lz = args.Lz
    L_cz = args.L_cz
    L_rz = args.Lz - args.L_cz
    zb = L_rz
    timestepper = d3.RK222
    gamma = args.gamma
    eps_s = args.eps_s
    Di = args.Di

    cp = 1
    cv = cp/gamma
    R = cp - cv

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], args.Nx, bounds=(0, Lx), dealias=args.dealias)
    zbasis = d3.Chebyshev(coords['z'], args.Nz, bounds=(0, Lz), dealias=args.dealias)

    x, z = dist.local_grids(xbasis, zbasis)
    x_global = xbasis.global_grid(dist, scale=1).ravel()
    z_global = zbasis.global_grid(dist, scale=1).ravel()

    # Evolved fields
    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    s = dist.Field(name='s', bases=(xbasis, zbasis))
    p = dist.Field(name='p', bases=(xbasis, zbasis))

    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis))
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis))
    tau_s1 = dist.Field(name='tau_s1', bases=(xbasis))
    tau_s2 = dist.Field(name='tau_s2', bases=(xbasis))
    tau_p = dist.Field(name='tau_p')

    # Background profiles
    T0 = dist.Field(name='T0', bases=(zbasis))
    rho0 = dist.Field(name='rho0', bases=(zbasis))
    p0 = dist.Field(name='p0', bases=(zbasis))
    ds0dz = dist.Field(name='ds0dz', bases=(zbasis))

    # Passive scalar
    C1 = dist.Field(name='C1', bases=(xbasis, zbasis))
    tau_C1_1 = dist.Field(name='tau_C1_1', bases=(xbasis))
    tau_C1_2 = dist.Field(name='tau_C1_2', bases=(xbasis))
    C2 = dist.Field(name='C2', bases=(xbasis, zbasis))
    tau_C2_1 = dist.Field(name='tau_C2_1', bases=(xbasis))
    tau_C2_2 = dist.Field(name='tau_C2_2', bases=(xbasis))

    # Initial conditions
    tw = args.tw * L_cz
    N2_cz = -1.0
    N2_rz = args.S * abs(N2_cz)
    N2 = 0.5 * (N2_cz - N2_rz) * (1 + np.tanh((z-zb)/tw)) + N2_rz

    ds0dz['g'] = N2
    u['g'] = 0.0
    p['g'] = 0.0
    s.fill_random('g', seed=42, distribution='normal', scale=1e-5)
    s['g'] = 1e-5 * np.cos(2*np.pi*x/Lx) * np.cos(np.pi*z/(2*Lz)) # temp
    s['g'] *= z*(Lz - z)

    C1['g'] = 0.5*(1 + np.tanh((z - zb) / tw))
    mu, sigma = zb - 0.15*L_cz, 0.15*L_cz
    C2['g'] = np.exp(-(z-mu)**2 / (2*sigma**2))

    # Initial fields - reference
    def func(z, y):
        T0 = y[0]
        ln_p0 = y[1]

        ds0dz = 0.5 * (N2_cz - N2_rz) * (1 + np.tanh((z-zb)/tw)) + N2_rz
        dTdz = eps_s*T0*ds0dz - Di
        dlnpdz = -gamma/(gamma-1) * Di/T0

        return [dTdz, dlnpdz]

    T_bottom = 1.0
    p_bottom = R * T_bottom
    sol = solve_ivp(func, [0, Lz], [T_bottom, np.log(p_bottom)], dense_output=True)

    T_prof, ln_p_prof = sol.sol(z.ravel())

    T0['g'] = T_prof
    p0['g'] = np.exp(ln_p_prof)
    rho0['g'] = p0['g']/(R*T0['g'])

    dlnrho0_dz = d3.Differentiate(rho0, coords['z']) / rho0


    # Substitutions
    nu = np.sqrt(args.Pr / args.Ra)
    chi = 1 / np.sqrt(args.Ra*args.Pr)

    #Pe = 2e3
    #kappa_C = 0.09494056048917088 * L_cz / Pe    # w_rms_cz for Ra=5e6 ~0.09494056048917088
    kappa_C = 0.001

    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    grad_u = d3.grad(u) + ez * lift(tau_u1)
    grad_s = d3.grad(s) + ez * lift(tau_s1)
    grad_C1 = d3.grad(C1) + ez * lift(tau_C1_1)
    grad_C2 = d3.grad(C2) + ez * lift(tau_C2_1)

    w = u @ ez

    # stress tensor
    I = dist.IdentityTensor(coords)
    varpi = p / rho0
    strain = 0.5*(grad_u + d3.TransposeComponents(grad_u))
    stress = 2*rho0*nu*(strain - d3.trace(grad_u)*I/3)

    T0_global = T0.gather_data(root=0, layout='g')
    rho0_global = rho0.gather_data(root=0, layout='g')
    p0_global = p0.gather_data(root=0, layout='g')
    ds0dz_global = ds0dz.gather_data(root=0, layout='g')

    # Save into file
    if dist.comm.rank == 0:
        info = write_info(run_folder, args)
        info['background'] = {
            'z': z_global.tolist(),
            'x': x_global.tolist(),
            'T0': np.squeeze(T0_global).tolist(),
            'rho0': np.squeeze(rho0_global).tolist(),
            'p0': np.squeeze(p0_global).tolist(),
            'ds0dz': np.squeeze(ds0dz_global).tolist(),
            'nu': nu,
            'chi': chi,
            'eps_s': eps_s,
            'Di': Di,
            'zb': zb,
            'N2_cz': N2_cz
        }
        write_json_atomic(f'{run_folder}/info.json', info)

    # Problem
    problem = d3.IVP([u, s, p, C1, C2, tau_u1, tau_u2, tau_s1, tau_s2, tau_p, tau_C1_1, tau_C1_2, tau_C2_1, tau_C2_2], namespace=locals())
    problem.add_equation('trace(grad_u) + w*dlnrho0_dz + tau_p = 0')
    problem.add_equation('dt(u) - 1/rho0 * div(stress) + grad(varpi) - s*ez + lift(tau_u2) = -u@grad(u)')
    problem.add_equation('dt(s) - 1/(rho0*T0) * div(rho0*T0*chi*grad_s) + lift(tau_s2) = -u@grad(s) - w*ds0dz')
    problem.add_equation('dt(C1) - 1/rho0 * div(rho0*kappa_C*grad_C1) + lift(tau_C1_2) = -u@grad(C1)')
    problem.add_equation('dt(C2) - 1/rho0 * div(rho0*kappa_C*grad_C2) + lift(tau_C2_2) = -u@grad(C2)')

    # Boundary conditions
    problem.add_equation('(u@ez)(z=0) = 0')
    problem.add_equation('(u@ez)(z=Lz) = 0')
    problem.add_equation('(ex@(ez@grad_u))(z=0) = 0')
    problem.add_equation('(ex@(ez@grad_u))(z=Lz) = 0')
    problem.add_equation('(ez@grad_s)(z=0) = 0')
    problem.add_equation('s(z=Lz) = 0')
    problem.add_equation('integ(varpi) = 0')
    problem.add_equation('(ez@grad_C1)(z=0) = 0')
    problem.add_equation('(ez@grad_C1)(z=Lz) = 0')
    problem.add_equation('(ez@grad_C2)(z=0) = 0')
    problem.add_equation('(ez@grad_C2)(z=Lz) = 0')

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = args.stop_sim_time

    # Analysis
    fields = solver.evaluator.add_file_handler(f'{run_folder}/field_snapshots', sim_dt=0.5, max_writes=50, mode='overwrite')
    fields.add_task(u, name='velocity')
    fields.add_task(s, name='entropy_perturbation')
    fields.add_task(C1, name='C1_concentration')
    fields.add_task(C2, name='C2_concentration')

    profiles = solver.evaluator.add_file_handler(f'{run_folder}/profile_snapshots', sim_dt=0.1, max_writes=100, mode='overwrite')
    profiles.add_task(d3.Average(w*eps_s*s*rho0, coords['x']), name='F_entropy_havg')
    profiles.add_task(d3.Average(w*rho0*T0*((gamma-1)/gamma * p/p0 + eps_s*s), coords['x']), name='F_enthalpy_havg')
    profiles.add_task(d3.Average(0.5*rho0*(u@u), coords['x']), name='KE_havg')
    profiles.add_task(d3.Average(0.5*rho0*w*(u@u), coords['x']), name='KE_flux_havg')
    profiles.add_task(d3.Average(s, coords['x']), name='s_havg')
    profiles.add_task(d3.Average(w**2, coords['x']), name='w_sq_havg')
    profiles.add_task(d3.Average(C1, coords['x']), name='C1_havg')
    profiles.add_task(d3.Integrate(C1*rho0, ('x', 'z')), name='C1_total')
    profiles.add_task(d3.Average(C2, coords['x']), name='C2_havg')
    profiles.add_task(d3.Integrate(C2*rho0, ('x', 'z')), name='C2_total')

    scalars = solver.evaluator.add_file_handler(f'{run_folder}/scalar_snapshots', sim_dt=0.25, max_writes=2000, mode='overwrite')
    scalars.add_task(d3.Integrate(0.5*rho0*(u@u), ('x', 'z')) / (Lx*Lz), name='KE')

    # CFL
    CFL = d3.CFL(solver, initial_dt=args.max_timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=args.max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)*L_cz/nu, name='Re')
    flow.add_property(0.5*rho0*(u@u), name='KE')

    # Main loop
    t_start = time.time()
    outcome = 'ok'
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                max_u = np.sqrt(np.max(u['g'][0]**2 + u['g'][1]**2))
                max_s = np.max(np.abs(s['g']))
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, max(u)=%f, max(s)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re, max_u, max_s))

                logger.info(f"T0 min/max = {np.min(T0['g'])}, {np.max(T0['g'])}")
                logger.info(f"rho0 min/max = {np.min(rho0['g'])}, {np.max(rho0['g'])}")
                logger.info(f"p0 min/max = {np.min(p0['g'])}, {np.max(p0['g'])}")

                if not np.isfinite(max_Re):
                    logger.error(f'Non-finite Re')
                    outcome = 'blew_up'
                    break
    except:
        logger.error('Exception raised, triggering end of main loop.')
        outcome = 'error'
        raise
    finally:
        solver.log_stats()
        t_total = (time.time() - t_start) / 3600
        core_hours = t_total * args.n_cores
        if dist.comm.rank == 0:
            info['timing'] = {'wall_hours': t_total, 'core_hours': core_hours}
            info['outcome'] = outcome
            write_json_atomic(f'{run_folder}/info.json', info)


if __name__ == '__main__':
    main()