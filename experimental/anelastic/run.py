"""Launch the experimental anelastic solver from the project root.

Run with: python -m experimental.anelastic.run
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    Ra = 1e7
    S = 5
    tw = 0.05
    Lx, Lz = 5, 3
    L_cz = 1
    Nx, Nz = 256, 128
    n_cores = 8
    max_timestep = 0.05
    stop_sim_time = 600
    mode = 'run'

    run_id = 'anelastic_0'
    cmd = [
        'mpiexec', '-n', str(n_cores), sys.executable, '-m', 'experimental.anelastic.solver',
        '--run-id', run_id, '--mode', mode, '--Ra', str(Ra), '--S', str(S), '--tw', str(tw),
        '--Lx', str(Lx), '--Lz', str(Lz), '--L-cz', str(L_cz), '--Nx', str(Nx), '--Nz', str(Nz),
        '--max-timestep', str(max_timestep), '--stop-sim-time', str(stop_sim_time)
    ]
    print(f'--- starting {run_id} ---')
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == '__main__':
    main()
