#!/usr/bin/env python3
"""Analyze archived ETACHA runs and Shima Mg+C equilibrium estimates.

ETACHA is run externally; this script never substitutes invented transition rates.
Shima Table 1 moments (NIFS-DATA-010, printed p.22, Z=12) are interpolated
in velocity (sqrt(E/u)); Eq.3 on printed p.11 supplies the Gaussian shape.
Discrete fractions are normalized over q=0..12. This is a reconstruction from
the sparse printed table, not a digitization of the original fitted curves.
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output/charge_states"
SHIMA_E = np.array([.02, .04, .1, .2, .4, 1., 2., 4., 6.])
SHIMA_Q = np.array([2.55, 3.34, 4.44, 5.42, 6.86, 9.01, 10.10, 11.28, 11.62])
SHIMA_D = np.array([.897, 1.05, 1.05, 1.05, 1.05, .955, .856, .701, .598])
Q = np.arange(13)


def shima_fractions(energy, interpolation="velocity"):
    energy = np.asarray(energy)
    if np.any((energy < SHIMA_E[0]) | (energy > SHIMA_E[-1])):
        raise ValueError("Outside the Shima table; do not silently extrapolate")
    transform = np.sqrt if interpolation == "velocity" else lambda x: x
    x, xp = transform(energy), transform(SHIMA_E)
    mean = np.interp(x, xp, SHIMA_Q)
    width = np.interp(x, xp, SHIMA_D)
    f = np.exp(-.5 * ((Q - mean[..., None]) / width[..., None])**2)
    return f / f.sum(axis=-1, keepdims=True)


def read_etacha(directory):
    """Read percent populations, whose column index is electron count, not q."""
    first = next(directory.glob("*Eta0009.txt"))
    text = first.read_text()
    if "Final energy" not in text:
        raise ValueError(f"Unfinished ETACHA output: {first}")
    qin = int(re.search(r"incident charge=\s*(\d+)", text)[1])
    ein = float(re.search(r"incident energy=\s*([\d.]+)", text)[1])
    final_energy = float(re.search(r"Final energy\s*:\s*([\d.]+)", text)[1])

    def numeric_rows(path):
        rows = []
        for line in path.read_text().splitlines():
            try:
                vals = [float(v) for v in line.split()]
            except ValueError:
                continue
            if len(vals) == 12:
                rows.append(vals)
        return np.asarray(rows)

    a = numeric_rows(first)
    b = numeric_rows(next(directory.glob("*Eta1019.txt")))
    if len(a) == 0 or len(a) != len(b):
        raise ValueError(f"Missing or unmatched population rows: {directory}")
    np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-10)
    # Electron populations n=0..12 become charge populations q=12..0.
    f = np.concatenate([a[:, 1:11], b[:, 1:4]], axis=1)[:, ::-1] / 100
    if f.min() < -1e-7 or np.max(abs(f.sum(axis=1)-1)) > 3e-6:
        raise ValueError(f"Unphysical population normalization: {directory}")
    roundoff = float(np.max(abs(f.sum(axis=1)-1)))
    f = np.maximum(f, 0)
    f /= f.sum(axis=1, keepdims=True)
    unique = np.r_[True, np.diff(a[:, 0]) > 0]
    return dict(name=directory.name, qin=qin, ein=ein, t=a[unique, 0],
                f=f[unique], eout=a[unique, -1], roundoff=roundoff,
                final_energy=final_energy,
                energy_loss="No stopping power correction" not in text)


def at_thickness(run, thickness):
    if thickness < run['t'][0] or thickness > run['t'][-1]:
        raise ValueError("Requested thickness outside calculated grid")
    return np.array([np.interp(thickness, run['t'], run['f'][:, q]) for q in Q])


def summarize_events():
    rows = []
    for path in sorted((BASE / "output").glob("[0-9]*_exit_kinematics.csv")):
        df = pd.read_csv(path, usecols=["hasMg26Exit", "Mg26_E_MeV",
                                       "Mg26_theta_deg", "triton_S3_ringID"])
        for selection, mask in [
            ("all_exiting", df.hasMg26Exit.eq(1)),
            ("S3_coincident", df.hasMg26Exit.eq(1) & df.triton_S3_ringID.ge(0))]:
            energies = df.loc[mask, "Mg26_E_MeV"].to_numpy() / 26
            angles = df.loc[mask, "Mg26_theta_deg"].to_numpy()
            valid = np.isfinite(energies) & (energies >= .02) & (energies <= 6)
            # Work in chunks to avoid large N x 13 temporary arrays.
            sums = np.zeros(13)
            sums_linear = np.zeros(13)
            for start in range(0, valid.sum(), 50000):
                e = energies[valid][start:start + 50000]
                sums += shima_fractions(e).sum(axis=0)
                sums_linear += shima_fractions(e, "energy").sum(axis=0)
            fractions = sums / valid.sum()
            linear = sums_linear / valid.sum()
            # Forward planar path approximation only; no inference for side exits.
            forward = np.isfinite(angles) & (angles >= 0) & (angles < 10)
            tau = 49 / np.cos(np.deg2rad(angles[forward]))
            row = dict(state=path.name.replace("_exit_kinematics.csv", ""),
                       selection=selection, count=len(energies),
                       model_count=int(valid.sum()), excluded_count=int((~valid).sum()),
                       E_mean_MeVu=float(energies.mean()),
                       E_p05_MeVu=float(np.quantile(energies, .05)),
                       E_p95_MeVu=float(np.quantile(energies, .95)),
                       forward_under10deg_count=int(forward.sum()),
                       forward_tau_mean_ugcm2=float(tau.mean()),
                       mean_q=float(fractions @ Q),
                       width_q=float(np.sqrt(fractions @ Q**2-(fractions @ Q)**2)),
                       max_interpolation_difference_pp=float(100*np.max(abs(fractions-linear))))
            row.update({f"F{q}": float(fractions[q]) for q in Q})
            rows.append(row)
        print(path.name, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "shima_event_weighted_fractions.csv", index=False)
    pd.DataFrame(dict(E_MeVu=SHIMA_E, mean_q=SHIMA_Q, width_q=SHIMA_D)).to_csv(
        OUT / "shima_Mg_table1_transcription.csv", index=False)


def summarize_runs():
    runs = [read_etacha(p) for p in sorted((OUT / "etacha").glob("q*"))
            if any(p.glob("*Eta0009.txt"))]
    rows, evolution = [], []
    for run in runs:
        for t in [10, 25, 49, 100, 150, 200, 500]:
            if run['t'][0] <= t <= run['t'][-1]:
                f = at_thickness(run, t)
                row = dict(run=run['name'], q_in=run['qin'], E_in_MeVu=run['ein'],
                           energy_loss=run['energy_loss'], thickness_ugcm2=t,
                           E_population_row_MeVu=float(np.interp(t, run['t'], run['eout'])),
                           E_final_reported_MeVu=run['final_energy'],
                           mean_q=float(f @ Q), width_q=float(np.sqrt(f@Q**2-(f@Q)**2)))
                row.update({f"F{q}": float(f[q]) for q in Q})
                rows.append(row)
        frame = pd.DataFrame(run['f'], columns=[f'F{q}' for q in Q])
        frame.insert(0, 'thickness_ugcm2', run['t'])
        frame.insert(0, 'run', run['name'])
        frame['E_out_MeVu'] = run['eout']
        evolution.append(frame)
    pd.DataFrame(rows).to_csv(OUT / 'etacha_fraction_summary.csv', index=False)
    pd.concat(evolution).to_csv(OUT / 'etacha_charge_evolution.csv', index=False)
    (OUT / 'etacha_checks.json').write_text(json.dumps([
        {k:v for k,v in run.items() if k not in ['f','t','eout']}
        for run in runs], indent=2))
    return runs


def diagnostics(runs):
    lookup = {r['name']:r for r in runs}
    comparisons = []
    pairs = [(f'q{q}_E2.16_fixed', f'q{q}_E2.16_tight') for q in [4,8,12]]
    pairs += [('q8_E2.16_tight','q8_E2.16_refined'),
              ('q8_E2.16_tight','q8_E2.184_energy_loss')]
    for a,b in pairs:
        if a in lookup and b in lookup:
            for t in [49,200]:
                if t <= min(lookup[a]['t'][-1],lookup[b]['t'][-1]):
                    delta = at_thickness(lookup[a],t)-at_thickness(lookup[b],t)
                    comparisons.append(dict(run_a=a,run_b=b,thickness_ugcm2=t,
                        TV=float(.5*abs(delta).sum()),max_difference_pp=float(100*abs(delta).max())))
    pd.DataFrame(comparisons).to_csv(OUT/'sensitivity_checks.csv',index=False)


def plot(runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.dpi':180})
    lookup = {r['name']:r for r in runs}
    selected = [lookup.get(f'q{q}_E2.16_tight', lookup[f'q{q}_E2.16_fixed']) for q in [4,8,12]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    colors = ['#b05d18','#176b9b','#884b91']
    for r,c in zip(selected,colors):
        ax[0].plot(Q, 100*at_thickness(r,49), 'o-', color=c, label=f"ETACHA: incoming {r['qin']}+")
    f = shima_fractions(np.array([2.16]))[0]
    ax[0].plot(Q,100*f,'s--',color='#30363d',label='Shima equilibrium reconstruction')
    ax[0].set(xlim=(6.5,12.4),ylim=(0,55),xticks=range(7,13),xlabel='Exit charge q',
              ylabel='Particle fraction (%)', title='49 μg/cm² carbon; 2.16 MeV/u')
    ax[0].legend(fontsize=8.4)
    r=selected[1]
    for q,c in zip([8,9,10,11,12],['#959595','#6d8b35','#176b9b','#b05d18','#884b91']):
        ax[1].plot(r['t'],100*r['f'][:,q],label=f'{q}+',color=c)
    ax[1].axvline(49,color='#30363d',ls='--',lw=1)
    ax[1].set(xlim=(0,200),ylim=(0,70),xlabel='Carbon areal density (μg/cm²)',
              ylabel='Particle fraction (%)',title='Evolution from incoming Mg8+')
    ax[1].legend(ncol=3,fontsize=9)
    grid=np.linspace(1,200,500)
    refs=np.stack([np.stack([at_thickness(r,t) for t in grid]) for r in selected])
    pair=np.max([.5*np.abs(refs[i]-refs[j]).sum(axis=1) for i,j in [(0,1),(0,2),(1,2)]],axis=0)
    ax[2].plot(grid,100*pair,color='#176b9b',label='Largest difference among entrance states')
    ax[2].axhline(1,color='#30363d',ls=':',label='1% total-variation criterion')
    ax[2].axvline(49,color='#30363d',ls='--')
    ax[2].set(xlim=(0,200),ylim=(.1,100),yscale='log',xlabel='Carbon areal density (μg/cm²)',
              ylabel='Total-variation distance (%)',title='Memory of incoming charge')
    ax[2].legend(fontsize=8,loc='upper right')
    for a in ax:a.grid(alpha=.16)
    fig.suptitle('26Mg charge states: model predictions, not measured fractions',fontsize=16)
    fig.savefig(OUT/'mg26_charge_state_summary.png')
    fig.savefig(OUT/'mg26_charge_state_summary.svg')
    pd.DataFrame({'thickness_ugcm2':grid,'max_pairwise_TV':pair}).to_csv(OUT/'equilibrium_diagnostic.csv',index=False)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-events',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if not args.skip_events:summarize_events()
    runs=summarize_runs()
    diagnostics(runs)
    plot(runs)
