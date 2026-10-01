#!/usr/bin/env python3
"""Plot gamma-gated excitation spectra with exact gate bounds in the legend."""
from plot_inputhist_histograms import DEFAULT_INPUT, DEFAULT_OUTDIR, plt, mpl, uproot

mpl.rcParams.update({'font.size': 14, 'axes.labelsize': 18,
                     'axes.spines.top': True, 'axes.spines.right': True})
gates = [(1130, 1130, 1115, 1155, '#41049d'),
         (1808, 1809, 1789, 1840, '#ef5267'),
         (2524, 2524, 2485, 2560, '#009e73')]
fig, ax = plt.subplots(figsize=(12, 7))
with uproot.open(DEFAULT_INPUT) as source:
    for key, label, low, high, color in gates:
        hist = source[f'hExcE_{key}_PGACL_Recoil']
        assert f'addDopp {low}-{high} keV' in hist.title
        counts, edges = hist.to_numpy()
        ax.stairs(counts, edges, fill=True, color=color, alpha=.10)
        ax.stairs(counts, edges, color=color, linewidth=1.8,
                  label=rf'{label} keV: ${low} < E_\gamma < {high}$ keV')
ax.set(xlim=(0, 15), ylim=(0, 47.2), xlabel=r'$E_x$ [MeV]', ylabel='Counts / 100 keV')
ax.tick_params(which='both', direction='in', top=True, right=True)
ax.minorticks_on()
ax.grid(which='major', color='.85', linewidth=.75, alpha=.75)
ax.grid(which='minor', color='.93', linewidth=.45, alpha=.55)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.legend(loc='upper left', fontsize=12, framealpha=.95, fancybox=False, edgecolor='.4')
stem = DEFAULT_OUTDIR / 'hExcE_PGACL_Recoil_gamma_gates_overlay'
for suffix in ('.pdf', '.png'):
    fig.savefig(stem.with_suffix(suffix))
plt.close(fig)
print(stem.with_suffix('.pdf'))
