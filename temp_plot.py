import pandas as pd
import matplotlib.pyplot as plt

mode = 'test'
csv_file = f'analysis/{mode}s/figures/goluskin_T_wT.csv'

df = pd.read_csv(csv_file)

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