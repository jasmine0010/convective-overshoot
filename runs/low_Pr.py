'''
python3 runs/low_Pr.py

python3 plot_run.py --run-id low_Pr --mode run --t-mask-frac 0.7 --fields
'''

import subprocess


Ra = 5e6
Pr = 0.5
S = 5
tw = 0.05
Lx, Lz = 5, 3
L_cz = 1
Nx, Nz = 512, 256
n_cores = 8
tau_conv = 20
stop_sim_time = 800
mode = 'run'


run_id = f'low_Pr'
cmd = [
    'mpiexec', '-n', str(n_cores), 'python3', 'convection.py', '--run-id', str(run_id), '--mode', mode,
    '--Ra', str(Ra), '--Pr', str(Pr), '--S', str(S), '--tw', str(tw), '--Lx', str(Lx), '--Lz', str(Lz), '--L-cz', str(L_cz),
    '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
]
print(f'--- starting {run_id} ---')
subprocess.run(cmd, check=True)