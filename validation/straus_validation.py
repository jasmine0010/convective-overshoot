import os
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metadata_io import write_json_atomic


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
    p.add_argument('--g', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=1.0)

    p.add_argument('--max-timestep', type=float, default=0.05)
    p.add_argument('--stop-sim-time', type=float, required=True)
    p.add_argument('--n-cores', '--n', dest='n_cores', type=int, default=8)

    return p.parse_args()


def get_run_folder(args):
    base = f'data/{args.mode}s'
    folder = f'{args.run_id}_Ra{args.Ra:.0f}_S{args.S:g}_Lcz{args.L_cz:g}_{args.Nx}x{args.Nz}'
    return f'{base}/{folder}'


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
            'S': args.S,
            'tw': args.tw,
            'max_timestep': args.max_timestep
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
    Lx = args.Lx
    Lz = args.Lz
    L_cz = args.L_cz
    L_rz = args.Lz - args.L_cz
    zb = L_rz
    g = args.g
    alpha = args.alpha
    timestepper = d3.RK222
    
    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], args.Nx, bounds=(0, Lx), dealias=args.dealias)
    zbasis = d3.Chebyshev(coords['z'], args.Nz, bounds=(0, Lz), dealias=args.dealias)

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
    
    # Substitutions
    nu = np.sqrt(args.Pr/args.Ra)
    kappa = np.sqrt(1/(args.Ra*args.Pr))
    
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    
    grad_u = d3.grad(u) + ez * lift(tau_u1)
    grad_theta = d3.grad(theta) + ez * lift(tau_theta1)
    
    w = u @ ez

    # Initial conditions
    dT0dz['g'] = 0.5 - z
    T0['g'] = 0.5*z*(1-z)
    
    u['g'] = 0.0
    p['g'] = 0.0
    theta.fill_random('g', seed=42, distribution='normal', scale=1e-7)
    theta['g'] *= z*(Lz - z)
    
    # Save into file
    if dist.comm.rank == 0:
        info = write_info(run_folder, args)
        z_global = zbasis.global_grid(dist, scale=1).ravel()
        x_global = xbasis.global_grid(dist, scale=1).ravel()
        
        dT0dz_global = 0.5 - z_global
        T0_global = 0.5*z_global*(1-z_global)
        
        info['background'] = {
            'z': z_global.tolist(),
            'x': x_global.tolist(),
            'dT0dz': dT0dz_global.tolist(),
            'T0': T0_global.tolist(),
            'nu': nu,
            'kappa': kappa,
            'zb': zb,
            'g': g,
            'alpha': alpha
        }
        write_json_atomic(f'{run_folder}/info.json', info)

    # Problem
    problem = d3.IVP([u, theta, p, tau_u1, tau_u2, tau_theta1, tau_theta2, tau_p], namespace=locals())
    problem.add_equation('trace(grad_u) + tau_p = 0')
    problem.add_equation('dt(u) - nu*div(grad_u) + grad(p) - g*alpha*theta*ez + lift(tau_u2) = - u@grad(u)')
    problem.add_equation('dt(theta) - kappa*div(grad_theta) + lift(tau_theta2) = - u@grad(theta) - w*dT0dz')
    
    # Boundary conditions
    problem.add_equation('(u@ez)(z=0) = 0')
    problem.add_equation('(u@ez)(z=Lz) = 0')
    problem.add_equation('(ex@(ez@grad_u))(z=0) = 0')
    problem.add_equation('(ex@(ez@grad_u))(z=Lz) = 0')
    problem.add_equation('theta(z=0) = 0')
    problem.add_equation('theta(z=Lz) = 0')
    problem.add_equation('integ(p) = 0')

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = args.stop_sim_time

    # Analysis
    fields = solver.evaluator.add_file_handler(f'{run_folder}/field_snapshots', sim_dt=0.5, max_writes=50, mode='overwrite')
    fields.add_task(u, name='velocity')
    fields.add_task(theta, name='temperature_perturbation')

    profiles = solver.evaluator.add_file_handler(f'{run_folder}/profile_snapshots', sim_dt=0.25, max_writes=500, mode='overwrite')
    profiles.add_task(d3.Average(w*theta, coords['x']), name='F_conv_havg')
    profiles.add_task(d3.Average(0.5*(u@u), coords['x']), name='KE_havg')
    profiles.add_task(d3.Average(0.5*w*(u@u), coords['x']), name='KE_flux_havg')
    profiles.add_task(d3.Average(theta, coords['x']), name='theta_havg')
    profiles.add_task(d3.Average(w**2, coords['x']), name='w_sq_havg')

    scalars = solver.evaluator.add_file_handler(f'{run_folder}/scalar_snapshots', sim_dt=0.25, max_writes=2000, mode='overwrite')
    scalars.add_task(d3.Integrate(0.5*(u@u), ('x', 'z')) / (Lx*Lz), name='KE')
    scalars.add_task(d3.Integrate(w**2, ('x', 'z')) / (Lx*Lz), name='w_sq')

    # CFL
    CFL = d3.CFL(solver, initial_dt=args.max_timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=args.max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)/nu, name='Re')
    flow.add_property(0.5*(u@u), name='KE')

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
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
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