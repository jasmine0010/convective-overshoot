import os
import argparse
import time
from math import erf

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

from metadata_io import write_json_atomic
from boussinesq_units import free_fall_scales, diffusion_coefficients, background_profiles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['test', 'run'], required=True)

    p.add_argument('--Lx', type=float, default=4.0)
    p.add_argument('--Lz', type=float, default=3.0)
    p.add_argument('--L-cz', type=float, default=1.0, help='Reference CZ length; Lx and Lz use the same reference units')
    p.add_argument('--Nx', type=int, default=512)
    p.add_argument('--Nz', type=int, default=256)
    p.add_argument('--dealias', type=float, default=1.5)

    p.add_argument('--Ra', type=float, default=1e6)
    p.add_argument('--Pr', type=float, default=1.0)
    p.add_argument('--S', type=float, default=4.0)
    p.add_argument('--tw', type=float, default=0.05, help='Transition width divided by reference L_cz')
    p.add_argument('--kappa-C', type=float, default=0.001, help='Dimensionless tracer diffusivity, 1/Pe_C')
    p.add_argument('--N2-cz', type=float, default=-1.0, help='Negative reference CZ buoyancy frequency squared')
    p.add_argument('--g', type=float, default=1.0, help='Reference gravity for dimensional conversion')
    p.add_argument('--alpha', type=float, default=1.0, help='Reference thermal expansion coefficient')

    p.add_argument('--max-timestep', type=float, default=0.1)
    p.add_argument('--release-time-C1', type=float, default=1600)
    p.add_argument('--release-time-C2', type=float, default=2200)
    p.add_argument('--release-time-C3', type=float, default=2200)
    p.add_argument('--stop-sim-time', type=float, required=True)
    p.add_argument('--n-cores', '--n', dest='n_cores', type=int, default=8)

    return p.parse_args()


def get_run_folder(args):
    base = f'data/{args.mode}s'
    folder = f'{args.run_id}_Ra{args.Ra:.0e}_S{args.S:g}_Lcz{args.L_cz:g}_{args.Nx}x{args.Nz}'
    return f'{base}/{folder}'


def write_info(run_folder, args):
    scales = free_fall_scales(args.L_cz, args.N2_cz, args.g, args.alpha)
    L_rz = args.Lz/args.L_cz - 1

    info = {
        'run_id': args.run_id,
        'params': {
            'L_cz': 1.0,
            'L_rz': L_rz,
            'Lx': args.Lx/args.L_cz,
            'Lz': args.Lz/args.L_cz,
            'Nx': args.Nx,
            'Nz': args.Nz,
            'dealias': args.dealias,
            'Ra': args.Ra,
            'Pr': args.Pr,
            'S': args.S,
            'tw': args.tw,
            'max_timestep': args.max_timestep
        },
        'units': {
            'convention': 'cz_free_fall_v1',
            'scales': scales,
            'reference': {'L_cz': args.L_cz, 'N2_cz': args.N2_cz, 'g': args.g, 'alpha': args.alpha},
            'description': 'All coordinates, fields, diffusivities and simulation times are in CZ free-fall units.',
            'time': 't/t_ff; tau_conv is a measured circulation time in these units.',
            'temperature': 'theta/DeltaT0; beta=N2/abs(N2_cz)=dT0/dz.',
            'diffusivity': 'D/(U_ff*L0); D_dim=D*U_ff*L0.',
            'kappa_C': '1/Pe_C',
            'reference_units': 'Physical conversion requires reference inputs in consistent physical units.'
        },
        'n_cores': args.n_cores,
        'stop_sim_time': args.stop_sim_time,
        'outcome': None,
        'timing': None
    }

    os.makedirs(run_folder, exist_ok=True)
    write_json_atomic(f'{run_folder}/info.json', info)
    return info


def main():
    args = parse_args()
    run_folder = get_run_folder(args)

    # Parameters
    scales = free_fall_scales(args.L_cz, args.N2_cz, args.g, args.alpha)
    Lx, Lz = args.Lx/scales['L0'], args.Lz/scales['L0']
    L_cz = 1.0
    L_rz = Lz - L_cz
    zb = L_rz
    nu, kappa, kappa_C = diffusion_coefficients(args.Ra, args.Pr, args.kappa_C)
    timestepper = d3.RK222

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], args.Nx, bounds=(0, Lx), dealias=args.dealias)
    zbasis = d3.Chebyshev(coords['z'], args.Nz, bounds=(0, Lz), dealias=args.dealias)

    z_global = zbasis.global_grid(dist, scale=1).ravel()
    x_global = xbasis.global_grid(dist, scale=1).ravel()

    # Evolved fields
    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    theta = dist.Field(name='theta', bases=(xbasis, zbasis))
    p = dist.Field(name='p', bases=(xbasis, zbasis))

    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis))
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis))
    tau_theta1 = dist.Field(name='tau_theta1', bases=(xbasis))
    tau_theta2 = dist.Field(name='tau_theta2', bases=(xbasis))
    tau_p = dist.Field(name='tau_p')

    # Background profiles
    dT0dz = dist.Field(name='dT0dz', bases=(zbasis))
    T0 = dist.Field(name='T0', bases=(zbasis))

    # Passive scalar
    C1 = dist.Field(name='C1', bases=(xbasis, zbasis))
    tau_C1_1 = dist.Field(name='tau_C1_1', bases=(xbasis))
    tau_C1_2 = dist.Field(name='tau_C1_2', bases=(xbasis))
    C2 = dist.Field(name='C2', bases=(xbasis, zbasis))
    tau_C2_1 = dist.Field(name='tau_C2_1', bases=(xbasis))
    tau_C2_2 = dist.Field(name='tau_C2_2', bases=(xbasis))
    C3 = dist.Field(name='C3', bases=(xbasis, zbasis))
    tau_C3_1 = dist.Field(name='tau_C3_1', bases=(xbasis))
    tau_C3_2 = dist.Field(name='tau_C3_2', bases=(xbasis))

    # Substitutions
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    grad_u = d3.grad(u) + ez * lift(tau_u1)
    grad_theta = d3.grad(theta) + ez * lift(tau_theta1)
    grad_C1 = d3.grad(C1) + ez * lift(tau_C1_1)
    grad_C2 = d3.grad(C2) + ez * lift(tau_C2_1)
    grad_C3 = d3.grad(C3) + ez * lift(tau_C3_1)

    w = u @ ez

    # Initial conditions
    tw = args.tw
    N2_cz = -1.0
    dT0dz['g'], T0['g'] = background_profiles(z, zb, args.S, tw)

    # Match C3's injected mass to C1/C2 without changing its Gaussian shape.
    mu, sigma = zb + 0.15*L_cz, 0.15*L_cz
    a, b = (Lz-zb)/tw, zb/tw
    C1_mass = 0.5*(Lz + tw*(np.logaddexp(a, -a)-np.logaddexp(b, -b)))
    C3_mass = sigma*np.sqrt(np.pi/2)*(erf((Lz-mu)/(np.sqrt(2)*sigma))-erf(-mu/(np.sqrt(2)*sigma)))
    C3_amplitude = C1_mass/C3_mass

    u['g'] = 0.0
    p['g'] = 0.0
    theta.fill_random('g', seed=42, distribution='normal', scale=1e-3)
    theta['g'] *= z*(Lz - z)

    C1['g'] = 0.0
    C2['g'] = 0.0
    C3['g'] = 0.0

    # Save into file
    if dist.comm.rank == 0:
        info = write_info(run_folder, args)

        dT0dz_global, T0_global = background_profiles(z_global, zb, args.S, tw)
        C1_global = 0.5*(1 + np.tanh((z_global - zb) / tw))

        info['background'] = {
            'z': z_global.tolist(),
            'x': x_global.tolist(),
            'dT0dz': dT0dz_global.tolist(),
            'T0': T0_global.tolist(),
            'nu': nu,
            'kappa': kappa,
            'kappa_C': kappa_C,
            'zb': zb,
            'C1': C1_global.tolist(),
            'C3_amplitude': float(C3_amplitude),
            'N2_cz': N2_cz,
            'g': 1.0,
            'alpha': 1.0,
            'N2_rz': float(args.S),
            'Pe_C': float(1/kappa_C)
        }
        write_json_atomic(f'{run_folder}/info.json', info)

    # Problem
    problem = d3.IVP([u, theta, p, C1, C2, C3, tau_u1, tau_u2, tau_theta1, tau_theta2, tau_p, tau_C1_1, tau_C1_2, tau_C2_1, tau_C2_2, tau_C3_1, tau_C3_2], namespace=locals())

    problem.add_equation('trace(grad_u) + tau_p = 0')
    problem.add_equation('dt(u) - nu*div(grad_u) + grad(p) - theta*ez + lift(tau_u2) = - u@grad(u)')
    problem.add_equation('dt(theta) - kappa*div(grad_theta) + lift(tau_theta2) = - u@grad(theta) - w*dT0dz')
    problem.add_equation('dt(C1) - kappa_C*div(grad_C1) + lift(tau_C1_2) = - u@grad(C1)')
    problem.add_equation('dt(C2) - kappa_C*div(grad_C2) + lift(tau_C2_2) = - u@grad(C2)')
    problem.add_equation('dt(C3) - kappa_C*div(grad_C3) + lift(tau_C3_2) = - u@grad(C3)')

    # Boundary conditions
    problem.add_equation('(u@ez)(z=0) = 0')
    problem.add_equation('(u@ez)(z=Lz) = 0')
    problem.add_equation('(ex@(ez@grad_u))(z=0) = 0')
    problem.add_equation('(ex@(ez@grad_u))(z=Lz) = 0')
    problem.add_equation('(ez@grad_theta)(z=0) = 0')
    problem.add_equation('theta(z=Lz) = 0')
    problem.add_equation('integ(p) = 0')
    problem.add_equation('(ez@grad_C1)(z=0) = 0')
    problem.add_equation('(ez@grad_C1)(z=Lz) = 0')
    problem.add_equation('(ez@grad_C2)(z=0) = 0')
    problem.add_equation('(ez@grad_C2)(z=Lz) = 0')
    problem.add_equation('(ez@grad_C3)(z=0) = 0')
    problem.add_equation('(ez@grad_C3)(z=Lz) = 0')

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = args.stop_sim_time

    # Analysis
    fields = solver.evaluator.add_file_handler(f'{run_folder}/field_snapshots', sim_dt=0.5, max_writes=50, mode='overwrite')
    fields.add_task(u, name='velocity')
    fields.add_task(theta, name='theta')
    fields.add_task(C1, name='C1_concentration')
    fields.add_task(C2, name='C2_concentration')
    fields.add_task(C3, name='C3_concentration')

    profiles = solver.evaluator.add_file_handler(f'{run_folder}/profile_snapshots', sim_dt=0.1, max_writes=50, mode='overwrite')
    profiles.add_task(d3.Average(w*theta, coords['x']), name='F_conv_havg')
    profiles.add_task(d3.Average(0.5*(u@u), coords['x']), name='KE_havg')
    profiles.add_task(d3.Average(0.5*w*(u@u), coords['x']), name='KE_flux_havg')
    profiles.add_task(d3.Average(theta, coords['x']), name='theta_havg')
    profiles.add_task(d3.Average(w**2, coords['x']), name='w_sq_havg')
    profiles.add_task(d3.Average(C1, coords['x']), name='C1_havg')
    profiles.add_task(d3.Integrate(C1, ('x', 'z')), name='C1_total')
    profiles.add_task(d3.Average(C2, coords['x']), name='C2_havg')
    profiles.add_task(d3.Integrate(C2, ('x', 'z')), name='C2_total')
    profiles.add_task(d3.Average(C3, coords['x']), name='C3_havg')
    profiles.add_task(d3.Integrate(C3, ('x', 'z')), name='C3_total')

    scalars = solver.evaluator.add_file_handler(f'{run_folder}/scalar_snapshots', sim_dt=0.25, max_writes=2000, mode='overwrite')
    scalars.add_task(d3.Integrate(0.5*(u@u), ('x', 'z')) / (Lx*Lz), name='KE')

    # CFL
    CFL = d3.CFL(solver, initial_dt=args.max_timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=args.max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)*L_cz/nu, name='Re')
    flow.add_property(0.5*(u@u), name='KE')

    # Main loop
    t_start = time.time()
    outcome = 'ok'

    release_time_C1 = args.release_time_C1
    release_time_C2 = args.release_time_C2
    release_time_C3 = args.release_time_C3
    C1_released = False
    C2_released = False
    C3_released = False

    try:
        logger.info('Starting main loop')
        while solver.proceed:
            if (not C1_released) and solver.sim_time >= release_time_C1:
                C1.change_scales(1)
                C1['g'] = 0.5*(1 + np.tanh((z - zb) / tw))
                C1_released = True
                logger.info('Released C1 at iteration=%i, sim_time=%e', solver.iteration, solver.sim_time)

            if (not C2_released) and solver.sim_time >= release_time_C2:
                C2.change_scales(1)
                C2['g'] = 0.5*(1 + np.tanh((z - zb) / tw))
                C2_released = True
                logger.info('Released C2 at iteration=%i, sim_time=%e', solver.iteration, solver.sim_time)

            if (not C3_released) and solver.sim_time >= release_time_C3:
                C3.change_scales(1)
                C3['g'] = C3_amplitude*np.exp(-(z-mu)**2 / (2*sigma**2))
                C3_released = True
                logger.info('Released C3 at iteration=%i, sim_time=%e', solver.iteration, solver.sim_time)

            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
                if not np.isfinite(max_Re):
                    logger.error('Non-finite Re')
                    outcome = 'blew_up'
                    break
    except:
        outcome = 'error'
        raise
    finally:
        solver.log_stats()
        t_total = (time.time() - t_start) / 3600
        core_hours = t_total * args.n_cores
        if dist.comm.rank == 0:
            info['timing'] = {
                'wall_hours': t_total,
                'core_hours': core_hours,
                'release_time_C1': release_time_C1,
                'release_time_C2': release_time_C2,
                'release_time_C3': release_time_C3
            }
            info['outcome'] = outcome
            write_json_atomic(f'{run_folder}/info.json', info)


if __name__ == '__main__':
    main()
