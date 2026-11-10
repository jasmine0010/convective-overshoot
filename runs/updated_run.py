'''
python3 runs/updated_run.py
python3 plot_run.py --run-id run_res --mode run --fields
python3 plot_tracer.py --run-id run_res --mode run
python3 diffusion_v4.py --run-id run_res --mode run

python3 plot_run.py --run-id run --mode run --fields
python3 plot_tracer.py --run-id run --mode run
python3 diffusion_v4.py --run-id run --mode run
'''

import subprocess


Ra = 5e7
S = 5
tw = 0.05
kappa_C = 0.001
Lx, Lz = 5, 3
L_cz = 1
Nx, Nz = 512, 256
n_cores = 8
stop_sim_time = 2600
mode = 'run'

run_id = f'run_res'
cmd = [
    'mpiexec', '-n', str(n_cores), 'python3', 'convection.py', '--run-id', str(run_id), '--mode', mode,
    '--Ra', str(Ra), '--S', str(S), '--tw', str(tw), '--kappa-C', str(kappa_C), '--Lx', str(Lx), '--Lz', str(Lz),
    '--L-cz', str(L_cz), '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
]

print(f'--- starting {run_id} ---')
subprocess.run(cmd, check=True)
