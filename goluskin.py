'''
python3 goluskin.py

for run in goluskin_0 goluskin_1 goluskin_2 goluskin_3 goluskin_4; do
    python3 plot_goluskin.py --run-id $run --mode test --t-mask-frac 0.7 --fields
done
'''

import subprocess
import numpy as np
import pandas as pd
import diagnostics as diag
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 8.5


Ra = 2e7
Pr_vals = [0.2, 0.5, 1, 2, 5]
Lx = 12
Lz = 1
Nx, Nz = 1024, 128
n_cores = 8
stop_sim_time = [600, 800, 1000, 1600, 2200]
mode = 'test'

published_wT = [0.0891, 0.122, 0.148, 0.175, 0.200]
published_T = [0.0380, 0.0375, 0.0380, 0.0384, 0.0377]
results = []


for i, (Pr, p_T, p_wT, sst) in enumerate(zip(Pr_vals, published_T, published_wT, stop_sim_time)):
    run_id = f'goluskin_{i}'
    cmd = [
        'mpiexec', '-n', str(n_cores), 'python3', 'validation/goluskin_validation.py', '--run-id', str(run_id),
        '--mode', mode, '--Ra', str(Ra), '--Pr', str(Pr), '--Lx', str(Lx), '--Lz', str(Lz), '--Nx', str(Nx),
        '--Nz', str(Nz), '--stop-sim-time', str(sst)
    ]
    print(f'--- starting {run_id} ---')
    subprocess.run(cmd, check=True)

    run_folder = diag.get_run_folder(run_id, mode)
    s = diag.load_scalar_snapshots(run_folder, tasks=['KE', 'w_sq', 'wT', 'T'])
    t_mask_s = s['time'] >= 0.7 * s['time'][-1]
    wT = np.mean(s['wT'][t_mask_s]) * np.sqrt(Ra*Pr)
    T = np.mean(s['T'][t_mask_s])

    results.append({'Ra': Ra, 'Pr': Pr, 'T': T, 'Goluskin_T': p_T, 'wT': wT, 'Goluskin_wT': p_wT})

df = pd.DataFrame(results)

df.to_csv(f'analysis/{mode}s/figures/goluskin_T_wT.csv', index=False)
print(f'Saved')



for col in ['Pr', 'T', 'Goluskin_T', 'wT', 'Goluskin_wT']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

ax1.plot(df['Pr'], df['T'], 'o-', label='simulation')
ax1.plot(df['Pr'], df['Goluskin_T'], 's--', label='published')
ax1.set_xlabel('Pr')
ax1.set_ylabel('T')
ax1.set_title('mean temperature')
ax1.legend()

ax2.plot(df['Pr'], df['wT'], 'o-', label='simulation')
ax2.plot(df['Pr'], df['Goluskin_wT'], 's--', label='published')
ax2.set_xlabel('Pr')
ax2.set_ylabel('wT')
ax2.set_title('convective heat flux')
ax2.legend()

fig.tight_layout()
fig.savefig(f'analysis/{mode}s/figures/goluskin_comparison.png', dpi=150)
print(f'Saved')
