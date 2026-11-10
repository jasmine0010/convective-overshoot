'''
python3 straus.py

for run in straus_0 straus_1 straus_2 straus_3 straus_4; do
    python3 plot_straus.py --run-id $run --mode test --t-mask-frac 0.7 --fields
done
'''

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import diagnostics as diag
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 8.5


a_vals = [1.3, 1.8, 2.2, 2.6, 3.026, 3.4, 3.8, 4.2, 4.7, 5.2, 5.7, 6.2]
Ra_vals = [15000, 20000, 25000, 30000, 35000, 40000, 45000]

Pr = 1
Lz = 1
L_cz = 0.5
Nx, Nz = 256, 128
n_cores = 8
stop_sim_time = 150
mode = 'test'

Ra_c_vals = []


fig, axes = plt.subplots(3, 4, figsize=(8, 6))
axes = axes.flatten()

for i, a in enumerate(a_vals):
    slopes = []
    for j, Ra in enumerate(Ra_vals):
        run_id = f'straus_{i*10+j}'
        Lx = 2*np.pi/a
        cmd = [
            'mpiexec', '-n', str(n_cores), 'python3', 'validation/straus_validation.py', '--run-id', str(run_id),
            '--mode', mode, '--Ra', str(Ra), '--Pr', str(Pr), '--Lx', str(Lx), '--Lz', str(Lz),
            '--L-cz', str(L_cz), '--Nx', str(Nx), '--Nz', str(Nz), '--stop-sim-time', str(stop_sim_time)
        ]
        print(f'--- starting {run_id} ---')
        subprocess.run(cmd, check=True)
    
        run_folder = diag.get_run_folder(run_id, mode)
        s = diag.load_scalar_snapshots(run_folder, tasks=['KE', 'w_sq'])
        time_s = np.asarray(s['time'])
        KE = np.asarray(s['KE'])
        mask_s = (time_s >= 0.5 * time_s[-1]) & (KE > 0)
        #slope = np.mean(np.gradient(np.log(KE[mask_s]), time_s[mask_s]))
        slope, _ = np.polyfit(time_s[mask_s], np.log(KE[mask_s]), 1)
        slopes.append(slope)

    ax = axes[i]
    ax.axhline(0, color='black', linestyle='--')
    ax.plot(Ra_vals, slopes, 'o-', markersize=3)
    ax.set_title(f'a = {a}')
    
    Ra_c = np.interp(0, slopes, Ra_vals)
    Ra_c_vals.append(Ra_c)

fig.supxlabel('Ra')
fig.supylabel('log KE slope')
fig.tight_layout()
fig.savefig(f'analysis/{mode}s/figures/straus_logKE_slopes.png', dpi=150)
plt.close(fig)


plt.figure(figsize=(4,4))
plt.semilogy(a_vals, Ra_c_vals, 'o-', markersize=3)
plt.scatter(3.026, 16992.3, marker='x')
plt.ylabel('Critical Rayleigh number')
plt.xlabel('Horizontal wavenumber')
plt.tight_layout()
plt.savefig(f'analysis/{mode}s/figures/straus_Rac_ac.png', dpi=150)
plt.close()
