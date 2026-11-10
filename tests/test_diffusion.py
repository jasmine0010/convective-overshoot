import numpy as np
import pytest


d3 = pytest.importorskip('dedalus.public')


def test_scalar_diffusion_mode_decays():
    Lx = 1.0
    Lz = 1.0
    Nx = 16
    Nz = 24
    kappa_C = 0.01
    stop_sim_time = 0.02
    timestep = 1e-3

    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(
        coords['x'], size=Nx, bounds=(0, Lx), dealias=3/2,
    )
    zbasis = d3.Chebyshev(
        coords['z'], size=Nz, bounds=(0, Lz), dealias=3/2,
    )

    _, z = dist.local_grids(xbasis, zbasis)
    ez = coords.unit_vector_fields(dist)[1]
    lift_basis = zbasis.derivative_basis(2)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    C = dist.Field(name='C', bases=(xbasis, zbasis))
    tau_C1 = dist.Field(name='tau_C1', bases=(xbasis,))
    tau_C2 = dist.Field(name='tau_C2', bases=(xbasis,))

    wave_number = np.pi / Lz
    C['g'] = np.cos(wave_number * z)

    problem = d3.IVP([C, tau_C1, tau_C2], namespace=locals())
    problem.add_equation(
        'dt(C) - kappa_C*div(grad(C)) '
        '+ lift(tau_C1, -1) '
        '+ lift(tau_C2, -2) = 0'
    )
    problem.add_equation('(ez@grad(C))(z=0) = 0')
    problem.add_equation('(ez@grad(C))(z=Lz) = 0')

    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time
    while solver.proceed:
        solver.step(timestep)

    C.change_scales(1)
    _, z_eval = dist.local_grids(xbasis, zbasis, scales=1)
    exact = (
        np.cos(wave_number * z_eval)
        * np.exp(-kappa_C * wave_number**2 * solver.sim_time)
    )
    relative_error = (
        np.max(np.abs(C['g'] - exact))
        / np.max(np.abs(exact))
    )

    assert relative_error < 0.05
