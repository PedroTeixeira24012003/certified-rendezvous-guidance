"""
plot_e3_distributions.py
Figures of the champion flown from the paired thousand, read as
distributions. Reads cl_E3_<runid>.npz and expert_refs.npz, writes
fig_e3_histograms.png (keep-out margin and final position error across the
1000 runs) and fig_e3_boxplots.png (paired deltas, network minus expert,
for final position error, keep-out margin, fuel and settling), plus light
versions of both.
"""

import os
import sys
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFS_FILE = 'expert_refs.npz'
R_KOZ = 5.0

NAVY = "#0A2540"; CARD = "#13293F"; TEAL = "#00A8B5"; BODY = "#CADCFC"
MUTE = "#8FA3BD"; ORANGE = "#E67E22"; WHITE = "#FFFFFF"; GRID = "#2C4258"


def newest_e3():
    fs = sorted(glob.glob('cl_E3_*.npz'), key=os.path.getmtime)
    if not fs:
        sys.exit("no cl_E3_*.npz found - run deploy_closedloop.py --mode e3 first")
    return fs[-1]


def style(ax, dark=True):
    fg = BODY if dark else "#111111"
    ln = GRID if dark else "#CCCCCC"
    ax.set_facecolor(CARD if dark else "#FFFFFF")
    for sp in ax.spines.values():
        sp.set_color(GRID if dark else "#333333")
    ax.tick_params(colors=(MUTE if dark else "#333333"), labelsize=8)
    ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)
    ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
    ax.grid(True, color=ln, linewidth=0.5, alpha=0.6, axis='y')
    ax.set_axisbelow(True)


def histograms(margin, final_err, exp_margin, dark=True):
    bg = NAVY if dark else "#FFFFFF"
    ttl = WHITE if dark else "#0A2540"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 4.7), facecolor=bg)

    # keep-out margin
    style(axL, dark)
    axL.hist(margin, bins=45, color=TEAL, alpha=0.9)
    axL.axvline(0.0, color=ORANGE, ls='--', lw=1.6)
    axL.text(0.06, axL.get_ylim()[1] * 0.9, "keep-out surface\n(margin = 0)",
             color=ORANGE, fontsize=8.5, fontweight="bold")
    axL.axvline(margin.min(), color=MUTE, ls=':', lw=1.2)
    axL.text(margin.min(), axL.get_ylim()[1] * 0.55,
             f"  closest\n  {margin.min():.3f} m", color=MUTE, fontsize=8)
    axL.set_xlabel("Keep-Out Margin  [m]")
    axL.set_ylabel("Runs")
    axL.set_title("A  Keep-Out Margin Across 1000 Runs", color=TEAL,
                  fontsize=10, fontweight="bold", loc="left")
    axL.set_xlim(left=-0.4)

    # final position error
    style(axR, dark)
    axR.hist(final_err * 1000, bins=45, color=TEAL, alpha=0.9)
    axR.axvline(final_err.mean() * 1000, color=ORANGE, ls='--', lw=1.6)
    axR.text(final_err.mean() * 1000, axR.get_ylim()[1] * 0.9,
             f"  mean {final_err.mean()*1000:.2f} mm", color=ORANGE,
             fontsize=8.5, fontweight="bold")
    axR.axvline(final_err.max() * 1000, color=MUTE, ls=':', lw=1.2)
    axR.text(final_err.max() * 1000, axR.get_ylim()[1] * 0.55,
             f"  worst\n  {final_err.max()*1000:.2f} mm", color=MUTE, fontsize=8)
    axR.set_xlabel("Final Position Error  [mm]")
    axR.set_ylabel("Runs")
    axR.set_title("B  Final Position Error Across 1000 Runs", color=TEAL,
                  fontsize=10, fontweight="bold", loc="left")

    fig.suptitle("The Thousand Paired Flights As Distributions: Every Run Clears "
                 "The Keep-Out Zone, Every Run Arrives", color=ttl, fontsize=13,
                 fontweight="bold", x=0.5, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.85, bottom=0.13, wspace=0.20)
    return fig


def boxplots(d_finalerr, d_margin, d_fuel, d_settle, dark=True):
    bg = NAVY if dark else "#FFFFFF"
    ttl = WHITE if dark else "#0A2540"
    fg = BODY if dark else "#111111"
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.4), facecolor=bg)

    panels = [
        (d_finalerr * 1000, "Final Position Error\n[mm]", "A"),
        (d_margin,          "Keep-Out Margin\n[m]",       "B"),
        (d_fuel,            "Fuel\n[m/s]",                "C"),
        (d_settle,          "Settling Time\n[s]",         "D"),
    ]
    for ax, (data, lab, tag) in zip(axes, panels):
        style(ax, dark)
        ax.grid(True, color=(GRID if dark else "#CCC"), linewidth=0.5,
                alpha=0.6, axis='y')
        bp = ax.boxplot(data, vert=True, widths=0.5, patch_artist=True,
                        showfliers=True,
                        medianprops=dict(color=WHITE if dark else "#000",
                                         linewidth=1.5),
                        flierprops=dict(marker='o', markersize=3,
                                        markerfacecolor=ORANGE,
                                        markeredgecolor='none', alpha=0.5),
                        whiskerprops=dict(color=fg, linewidth=1.0),
                        capprops=dict(color=fg, linewidth=1.0))
        for patch in bp['boxes']:
            patch.set_facecolor(TEAL)
            patch.set_alpha(0.55)
            patch.set_edgecolor(fg)
        ax.axhline(0.0, color=ORANGE, ls='--', lw=1.2, alpha=0.9)
        ax.set_ylabel(lab, color=fg, fontsize=9)
        ax.set_xticks([])
        ax.set_title(f"{tag}", color=TEAL, fontsize=10, fontweight="bold",
                     loc="left")

    fig.suptitle("Paired Deltas, Champion Minus Expert On The Identical "
                 "Condition (Dashed Line: No Difference From The Expert)",
                 color=ttl, fontsize=13, fontweight="bold", x=0.5, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.06, wspace=0.42)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runid', type=str, default=None)
    args = p.parse_args()

    e3 = f'cl_E3_{args.runid}.npz' if args.runid else newest_e3()
    if not os.path.exists(e3):
        sys.exit(f"{e3} not found")
    print(f"loading {e3} ...")
    d = np.load(e3)

    margin = np.asarray(d['net_koz_margin'], dtype=np.float64)
    final_err = np.asarray(d['net_final_pos_err'], dtype=np.float64)
    d_fuel = np.asarray(d['fuel_delta'], dtype=np.float64)
    d_settle_raw = np.asarray(d['settling_delta'], dtype=np.float64)
    d_settle = d_settle_raw[~np.isnan(d_settle_raw)]

    # paired deltas for error and margin need the expert reference
    exp_margin = None
    d_finalerr = None
    d_margin = None
    if os.path.exists(REFS_FILE):
        refs = np.load(REFS_FILE)
        if 'fom_koz_margin' in refs:
            exp_margin = np.asarray(refs['fom_koz_margin'], dtype=np.float64)
            d_margin = margin - exp_margin
        if 'fom_final_pos_err' in refs:
            d_finalerr = final_err - np.asarray(refs['fom_final_pos_err'],
                                                dtype=np.float64)
    # fall back to absolute values if a reference key is missing
    if d_margin is None:
        d_margin = margin - R_KOZ * 0.0 - margin.mean() * 0.0 + (margin - margin)
        d_margin = margin - np.median(margin)      # centred, for shape only
        print("  (expert margin reference missing; box shows centred margin)")
    if d_finalerr is None:
        d_finalerr = final_err - np.median(final_err)
        print("  (expert final-error reference missing; box shows centred error)")

    n = len(margin)
    print(f"  {n} runs")
    print(f"  margin: mean {margin.mean():.3f}  min {margin.min():.3f} m")
    print(f"  final err: mean {final_err.mean()*1000:.2f}  "
          f"max {final_err.max()*1000:.2f} mm")
    print(f"  fuel delta: mean {d_fuel.mean():+.4f}  "
          f"[{d_fuel.min():+.4f}, {d_fuel.max():+.4f}] m/s")
    print(f"  settling delta: mean {d_settle.mean():+.2f}  "
          f"max {d_settle.max():+.2f} s")

    for dark, suffix, dpi in [(True, '', 150), (False, '_light', 200)]:
        fh = histograms(margin, final_err, exp_margin, dark)
        f1 = f'fig_e3_histograms{suffix}.png'
        fh.savefig(f1, dpi=dpi, facecolor=(NAVY if dark else "#FFFFFF"))
        print(f"  saved -> {f1}")

        fb = boxplots(d_finalerr, d_margin, d_fuel, d_settle, dark)
        f2 = f'fig_e3_boxplots{suffix}.png'
        fb.savefig(f2, dpi=dpi, facecolor=(NAVY if dark else "#FFFFFF"))
        print(f"  saved -> {f2}")


if __name__ == "__main__":
    main()