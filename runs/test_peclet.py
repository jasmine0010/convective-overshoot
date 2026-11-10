'''
python3 runs/initial_run.py

python3 plot_run.py --run-id peclet --mode run --t-mask-frac 0.3 --fields

python3 plot_tracer.py --run-id peclet --mode run --t-mask-frac 0.3
'''

import subprocess


kappa_C = 0.0000782
Ra = 5e6
S_vals = [2, 5]
tw = 0.05
Lx, Lz = 5, 3
L_cz = 1
Nx, Nz = 512, 256
n_cores = 8
stop_sim_time = 1500
mode = 'run'


for idx, S in enumerate(S_vals):
    run_id = f'runs_{idx}'
    cmd = [
        'mpiexec', '-n', str(n_cores), 'python3', 'convection.py', '--run-id', str(run_id), '--mode', mode,
        '--Ra', str(Ra), '--S', str(S), '--tw', str(tw), '--kappa-C', str(kappa_C), '--Lx', str(Lx), '--Lz', str(Lz), '--L-cz', str(L_cz),
        '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
    ]
    print(f'--- starting {run_id} ---')
    subprocess.run(cmd, check=True)