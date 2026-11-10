'''
python3 tests/s_scan.py

for run in S_scan_0 S_scan_1 S_scan_2 S_scan_3; do
    python3 plot_run.py --run-id $run --mode test --fields
done
'''

import subprocess


Ra = 1e6
S_vals = [3, 5, 10, 20]
tw = 0.05
Lx, Lz = 5, 3
L_cz = 1
Nx, Nz = 256, 128
n_cores = 8
stop_sim_time = 70
mode = 'test'


for idx, S in enumerate(S_vals):
    run_id = f'S_scan_{idx}'
    cmd = [
        'mpiexec', '-n', str(n_cores), 'python3', 'convection.py', '--run-id', str(run_id), '--mode', mode,
        '--Ra', str(Ra), '--S', str(S), '--tw', str(tw), '--Lx', str(Lx), '--Lz', str(Lz), '--L-cz', str(L_cz),
        '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
    ]
    print(f'--- starting {run_id} ---')
    subprocess.run(cmd, check=True)