"""
plot_ch10_robustness.py
Figures for the robustness ladder. Reads the summary CSV written by
robustness_ladder.py and writes four figures, degradation curves, minimum
keep-out margin, violation counts per rung, and the safety-liveness split,
each as 300 dpi PNG plus vector PDF.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

CSV = 'rob_summary_20260723-030815-8f9a49.csv'
RUNGS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']
RUNG_X = np.arange(len(RUNGS))
R_KOZ = 5.0

# house style: restrained, print-ready
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# muted, colour-blind-safe trio
C = {'expert': '#4C4C4C', 'raw': '#C44E52', 'certified': '#4C72B0'}
LBL = {'expert': 'Expert', 'raw': 'Raw Policy', 'certified': 'Certified Policy'}
MK = {'expert': 'o', 'raw': 's', 'certified': '^'}


def load():
    rows = list(csv.DictReader(open(CSV)))
    d = {}
    for r in rows:
        d[(r['rung'], r['controller'])] = r
    def series(ctrl, key, cast=float):
        return np.array([cast(d[(R, ctrl)][key]) for R in RUNGS])
    return series


def style_axis(ax):
    ax.set_xticks(RUNG_X)
    ax.set_xticklabels(RUNGS)
    ax.set_xlabel('Severity Rung')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# =========================================================================
# FIGURE 1: degradation curves (three panels, three lines each)
# =========================================================================
def fig_degradation(series):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))

    # panel A: convergence fraction
    ax = axs[0]
    for c in ('expert', 'raw', 'certified'):
        y = series(c, 'converged', int) / 10.0   # percent of 1000
        ax.plot(RUNG_X, y, color=C[c], marker=MK[c], markersize=5,
                linewidth=1.8, label=LBL[c], markerfacecolor='white',
                markeredgewidth=1.4)
    ax.set_ylabel('Converged (\\%)')
    ax.set_title('Convergence To Hold Tolerance')
    ax.set_ylim(-5, 105)
    style_axis(ax)
    ax.legend(loc='upper right')

    # panel B: mean final position error
    ax = axs[1]
    for c in ('expert', 'raw', 'certified'):
        y = series(c, 'mean_final_pos_err')
        ax.plot(RUNG_X, y, color=C[c], marker=MK[c], markersize=5,
                linewidth=1.8, label=LBL[c], markerfacecolor='white',
                markeredgewidth=1.4)
    ax.set_ylabel('Mean Final Position Error (m)')
    ax.set_title('Arrival Accuracy')
    style_axis(ax)

    # panel C: mean fuel
    ax = axs[2]
    for c in ('expert', 'raw', 'certified'):
        y = series(c, 'mean_fuel')
        ax.plot(RUNG_X, y, color=C[c], marker=MK[c], markersize=5,
                linewidth=1.8, label=LBL[c], markerfacecolor='white',
                markeredgewidth=1.4)
    ax.set_ylabel('Mean Fuel (m/s)')
    ax.set_title('Fuel Expenditure')
    style_axis(ax)

    fig.suptitle('Fleet Degradation Across The Disturbance Ladder',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'fig_ch10_degradation.{ext}')
    plt.close(fig)
    print("saved fig_ch10_degradation")


# =========================================================================
# FIGURE 2: the filter earning its keep (minimum margin vs rung)
# =========================================================================
def fig_margin(series):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    raw = series('raw', 'min_margin')
    cert = series('certified', 'min_margin')

    ax.axhline(0.0, color='#B22222', linestyle='--', linewidth=1.2, zorder=1)
    ax.text(0.05, 0.06, 'Keep-Out Sphere Boundary', color='#B22222',
            fontsize=9, transform=ax.get_yaxis_transform(), va='bottom')

    ax.plot(RUNG_X, raw, color=C['raw'], marker=MK['raw'], markersize=7,
            linewidth=2.0, label=LBL['raw'], markerfacecolor='white',
            markeredgewidth=1.6, zorder=3)
    ax.plot(RUNG_X, cert, color=C['certified'], marker=MK['certified'],
            markersize=7, linewidth=2.0, label=LBL['certified'],
            markerfacecolor='white', markeredgewidth=1.6, zorder=3)

    # shade the breach region below zero
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -3.5, 0.0,
               color='#B22222', alpha=0.05, zorder=0)

    # annotate the L5 separation
    ax.annotate(f'{raw[-1]:+.2f} m', (RUNG_X[-1], raw[-1]),
                textcoords='offset points', xytext=(8, -2),
                color=C['raw'], fontsize=10, fontweight='bold')
    ax.annotate(f'{cert[-1]:+.3f} m', (RUNG_X[-1], cert[-1]),
                textcoords='offset points', xytext=(8, 4),
                color=C['certified'], fontsize=10, fontweight='bold')

    ax.set_ylabel('Minimum Keep-Out Margin (m)')
    ax.set_title('The Filter Earning Its Keep: Minimum Distance To The Sphere')
    style_axis(ax)
    ax.legend(loc='lower left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'fig_ch10_filter_margin.{ext}')
    plt.close(fig)
    print("saved fig_ch10_filter_margin")


# =========================================================================
# FIGURE 3: violation counts per rung (raw vs certified), grouped bars
# =========================================================================
def fig_violations(series):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    raw = series('raw', 'violations', int)
    cert = series('certified', 'violations', int)
    w = 0.38

    b1 = ax.bar(RUNG_X - w / 2, np.maximum(raw, 0.0), w, color=C['raw'],
                label=LBL['raw'], edgecolor='white', linewidth=0.6)
    b2 = ax.bar(RUNG_X + w / 2, np.maximum(cert, 0.0), w, color=C['certified'],
                label=LBL['certified'], edgecolor='white', linewidth=0.6)

    # value labels on the L5 bars (the only nonzero ones)
    for xi, val, off in [(RUNG_X[-1] - w / 2, raw[-1], -w / 2),
                         (RUNG_X[-1] + w / 2, cert[-1], w / 2)]:
        ax.annotate(f'{val}', (xi, val), textcoords='offset points',
                    xytext=(0, 3), ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Keep-Out Violations (Of 1000 Flights)')
    ax.set_title('Violations Per Rung: 131 Reduced To 1 At The Breaking Rung')
    style_axis(ax)
    ax.legend(loc='upper left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'fig_ch10_violations.{ext}')
    plt.close(fig)
    print("saved fig_ch10_violations")


# =========================================================================
# FIGURE 4: safety-liveness split (filter guards safety, not liveness)
# =========================================================================
def fig_safety_liveness(series):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))

    # left: liveness (convergence) - raw and certified overlap
    ax = axs[0]
    for c in ('raw', 'certified'):
        y = series(c, 'converged', int) / 10.0
        ax.plot(RUNG_X, y, color=C[c], marker=MK[c], markersize=6,
                linewidth=2.0, label=LBL[c], markerfacecolor='white',
                markeredgewidth=1.5)
    ax.set_ylabel('Converged (\\%)')
    ax.set_title('Liveness: The Filter Leaves Convergence Unchanged')
    ax.set_ylim(-5, 105)
    style_axis(ax)
    ax.legend(loc='upper right')

    # right: safety (margin) - raw and certified diverge at L4/L5
    ax = axs[1]
    ax.axhline(0.0, color='#B22222', linestyle='--', linewidth=1.1)
    for c in ('raw', 'certified'):
        y = series(c, 'min_margin')
        ax.plot(RUNG_X, y, color=C[c], marker=MK[c], markersize=6,
                linewidth=2.0, label=LBL[c], markerfacecolor='white',
                markeredgewidth=1.5)
    ax.set_ylabel('Minimum Keep-Out Margin (m)')
    ax.set_title('Safety: The Filter Holds The Sphere Where Raw Breaches')
    style_axis(ax)
    ax.legend(loc='lower left')

    fig.suptitle('Safety And Liveness Are Guarded Separately',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'fig_ch10_safety_liveness.{ext}')
    plt.close(fig)
    print("saved fig_ch10_safety_liveness")


if __name__ == '__main__':
    series = load()
    fig_degradation(series)
    fig_margin(series)
    fig_violations(series)
    fig_safety_liveness(series)
    print("\nAll four figures written (pdf + png).")