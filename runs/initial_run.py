'''
python3 runs/initial_run.py

for run in runs_0 runs_1; do
    python3 plot_run.py --run-id $run --mode run --fields
done

for run in runs_0 runs_1; do
    python3 plot_tracer.py --run-id $run --mode run
done
'''

'''
python3 runs/initial_run.py
python3 plot_run.py --run-id run --mode run --fields
python3 plot_tracer.py --run-id run --mode run
python3 diffusion_v3.py --run-id run --mode run
'''

import subprocess


Ra = 5e6
#S_vals = [3, 5]
S = 5
tw = 0.05
Lx, Lz = 5, 3
L_cz = 1
Nx, Nz = 256, 126
n_cores = 8
stop_sim_time = 2600
mode = 'run'

run_id = f'run'
cmd = [
    'mpiexec', '-n', str(n_cores), 'python3', 'convection.py', '--run-id', str(run_id), '--mode', mode,
    '--Ra', str(Ra), '--S', str(S), '--tw', str(tw), '--Lx', str(Lx), '--Lz', str(Lz), '--L-cz', str(L_cz),
    '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
]
print(f'--- starting {run_id} ---')
subprocess.run(cmd, check=True)

'''
for idx, S in enumerate(S_vals):
    run_id = f'runs_{idx}'
    cmd = [
        'mpiexec', '-n', str(n_cores), 'python3', 'convection.py', '--run-id', str(run_id), '--mode', mode,
        '--Ra', str(Ra), '--S', str(S), '--tw', str(tw), '--Lx', str(Lx), '--Lz', str(Lz), '--L-cz', str(L_cz),
        '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
    ]
    print(f'--- starting {run_id} ---')
    subprocess.run(cmd, check=True)
'''
