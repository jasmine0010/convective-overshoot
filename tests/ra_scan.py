'''
python3 tests/ra_scan.py

for run in Ra_scan_0 Ra_scan_1 Ra_scan_2 Ra_scan_3 Ra_scan_4; do
    python3 plot_run.py --run-id $run --mode test --fields
done
'''

import subprocess


Ra_vals = [1e6, 5e6, 1e7, 5e7, 1e8]
S = 5
tw = 0.05
Lx, Lz = 5, 3
L_cz = 1
Nx, Nz = 256, 128
n_cores = 8
stop_sim_time = 70
mode = 'test'


for idx, Ra in enumerate(Ra_vals):
    run_id = f'Ra_scan_{idx}'
    cmd = [
        'mpiexec', '-n', str(n_cores), 'python3', 'convection.py', '--run-id', str(run_id), '--mode', mode,
        '--Ra', str(Ra), '--S', str(S), '--tw', str(tw), '--Lx', str(Lx), '--Lz', str(Lz), '--L-cz', str(L_cz),
        '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
    ]
    print(f'--- starting {run_id} ---')
    subprocess.run(cmd, check=True)