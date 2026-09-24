"""
plot_dataset_distributions.py
Six-panel figure of the dataset's feature and label distributions, command
magnitude by phase, distance to hold, per-axis commands, orbital phase
sampling, position and velocity spreads. Saves a dark and a light version.
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATASET = "th_bc_dataset_behind.npz"
OUT = "th_dataset_distributions.png"

U_MAX = 0.082
HOLD = np.array([-8.0, 0.0, 0.0])
PHASE_SPLIT = 15.0          # metres to hold: transit above, holding below

# figure palette
NAVY = "#0A2540"
CARD = "#13293F"
TEAL = "#00A8B5"
BODY = "#CADCFC"
MUTE = "#8FA3BD"
ORANGE = "#E67E22"
WHITE = "#FFFFFF"
GRID = "#2C4258"


def style_axis(ax, title=None):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(GRID)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=7.5, length=3)
    ax.xaxis.label.set_color(BODY)
    ax.yaxis.label.set_color(BODY)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEAL, fontsize=10, fontweight="bold",
                     pad=9, loc="left")


def main():
    if not os.path.exists(DATASET):
        sys.exit(f"{DATASET} not found. Run this from the thesis-code directory.")

    print(f"loading {DATASET} ...")
    d = np.load(DATASET, mmap_mode="r")
    X = np.asarray(d["X"], dtype=np.float64)     # (N, 10)
    Y = np.asarray(d["Y"], dtype=np.float64)     # (N, 3)
    N = X.shape[0]
    print(f"  {N:,} demonstrations, {X.shape[1]} features")

    pos = X[:, 0:3]
    vel = X[:, 3:6]
    sin_t, cos_t = X[:, 6], X[:, 7]

    # split the demonstrations into transit and holding by distance to hold
    dist = np.linalg.norm(pos - HOLD, axis=1)
    unorm = np.linalg.norm(Y, axis=1)
    transit = dist > PHASE_SPLIT
    holding = ~transit

    n_tr, n_ho = int(transit.sum()), int(holding.sum())
    print(f"  transit {n_tr:,} ({100*n_tr/N:.1f}%)  "
          f"holding {n_ho:,} ({100*n_ho/N:.1f}%)")
    print(f"  mean |u| transit {unorm[transit].mean():.4f}  "
          f"holding {unorm[holding].mean():.4f} m/s^2")

    fig = plt.figure(figsize=(13.2, 8.0), facecolor=NAVY)
    gs = GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.28,
                  left=0.055, right=0.982, top=0.855, bottom=0.075)

    # panel A, command magnitude by phase
    ax = fig.add_subplot(gs[0, 0]); style_axis(ax, "A  Command Magnitude By Phase")
    bins = np.linspace(0, max(unorm.max(), U_MAX * 1.75), 90)
    ax.hist(unorm[holding], bins=bins, color=TEAL, alpha=0.85,
            label=f"Holding  {100*n_ho/N:.1f}%", log=True)
    ax.hist(unorm[transit], bins=bins, color=ORANGE, alpha=0.8,
            label=f"Transit  {100*n_tr/N:.1f}%", log=True)
    ax.axvline(unorm[holding].mean(), color=TEAL, ls="--", lw=1.2)
    ax.axvline(unorm[transit].mean(), color=ORANGE, ls="--", lw=1.2)
    ax.set_xlabel(r"$\|\mathbf{u}\|$  [m/s$^2$]")
    ax.set_ylabel("Demonstrations (log)")
    lg = ax.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, framealpha=1)
    for t in lg.get_texts():
        t.set_color(BODY)
    ax.text(0.97, 0.60,
            f"means differ\n{unorm[transit].mean()/unorm[holding].mean():.0f}\u00d7",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color=MUTE, style="italic")

    # panel B, distance to the hold point
    ax = fig.add_subplot(gs[0, 1]); style_axis(ax, "B  Distance To The Hold Point")
    ax.hist(dist, bins=110, color=TEAL, alpha=0.9)
    ax.axvline(PHASE_SPLIT, color=ORANGE, ls="--", lw=1.4)
    ax.text(PHASE_SPLIT + 2.5, ax.get_ylim()[1] * 0.86,
            f"phase split\n{PHASE_SPLIT:.0f} m", color=ORANGE, fontsize=7.5,
            fontweight="bold")
    ax.set_xlabel("Distance To Hold Point  [m]")
    ax.set_ylabel("Demonstrations")

    # panel C, per-axis commands against the thrust box
    ax = fig.add_subplot(gs[0, 2]); style_axis(ax, "C  Commanded Acceleration Per Axis")
    cb = np.linspace(-U_MAX * 1.04, U_MAX * 1.04, 120)
    for i, (lab, col) in enumerate(zip(("$u_x$", "$u_y$", "$u_z$"),
                                       (ORANGE, TEAL, BODY))):
        ax.hist(Y[:, i], bins=cb, histtype="step", lw=1.5, color=col,
                label=lab, log=True)
    for s in (-1, 1):
        ax.axvline(s * U_MAX, color=WHITE, ls=":", lw=1.1, alpha=0.75)
    ax.text(U_MAX, ax.get_ylim()[1] * 0.30, "  $u_{max}$", color=WHITE,
            fontsize=7.5, ha="left")
    ax.set_xlabel("Command  [m/s$^2$]")
    ax.set_ylabel("Demonstrations (log)")
    lg = ax.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, framealpha=1,
                   loc="upper left")
    for t in lg.get_texts():
        t.set_color(BODY)

    # panel D, sin/cos unit circle of the sampled true anomaly
    ax = fig.add_subplot(gs[1, 0]); style_axis(ax, "D  Orbital Phase Sampling")
    step = max(1, N // 60000)                    # thin for a legible scatter
    ax.scatter(cos_t[::step], sin_t[::step], s=0.6, color=TEAL, alpha=0.25,
               edgecolors="none")
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=ORANGE, lw=1.2, ls="--", alpha=0.9)
    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel(r"$\sin\theta$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.text(0.5, -1.16, f"std {sin_t.std():.3f} / {cos_t.std():.3f}"
                        r"  $\approx 1/\sqrt{2}$",
            ha="center", fontsize=7.5, color=MUTE, style="italic")

    # panel E, relative position spread
    ax = fig.add_subplot(gs[1, 1]); style_axis(ax, "E  Relative Position")
    for i, (lab, col) in enumerate(zip(("$x$  (V-bar)", "$y$  (R-bar)",
                                        "$z$  (H-bar)"),
                                       (ORANGE, TEAL, BODY))):
        ax.hist(pos[:, i], bins=110, histtype="step", lw=1.5, color=col,
                label=lab, log=True)
    ax.axvline(HOLD[0], color=WHITE, ls=":", lw=1.1, alpha=0.75)
    ax.set_xlabel("Position  [m]")
    ax.set_ylabel("Demonstrations (log)")
    lg = ax.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, framealpha=1)
    for t in lg.get_texts():
        t.set_color(BODY)

    # panel F, relative velocity spread
    ax = fig.add_subplot(gs[1, 2]); style_axis(ax, "F  Relative Velocity")
    for i, (lab, col) in enumerate(zip((r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"),
                                       (ORANGE, TEAL, BODY))):
        ax.hist(vel[:, i], bins=110, histtype="step", lw=1.5, color=col,
                label=lab, log=True)
    ax.set_xlabel("Velocity  [m/s]")
    ax.set_ylabel("Demonstrations (log)")
    lg = ax.legend(fontsize=7.5, facecolor=CARD, edgecolor=GRID, framealpha=1)
    for t in lg.get_texts():
        t.set_color(BODY)

    fig.suptitle("Feature And Label Distributions Across The Behavioural-Cloning Dataset",
                 color=WHITE, fontsize=15, fontweight="bold", x=0.055, ha="left",
                 y=0.975)
    fig.text(0.055, 0.922,
             f"{N:,} demonstrations from 10,000 accepted trajectories  \u00b7  "
             f"transit {100*n_tr/N:.1f}% / holding {100*n_ho/N:.1f}%  \u00b7  "
             f"mean $\\|u\\|$ {unorm[transit].mean():.4f} vs "
             f"{unorm[holding].mean():.4f} m/s$^2$",
             color=MUTE, fontsize=9.5, style="italic", ha="left")

    fig.savefig(OUT, dpi=150, facecolor=NAVY)
    print(f"\nsaved -> {OUT}")

    # light-background version
    out_light = OUT.replace(".png", "_light.png")
    for ax in fig.axes:
        ax.set_facecolor("#FFFFFF")
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.tick_params(colors="#333333")
        ax.xaxis.label.set_color("#111111")
        ax.yaxis.label.set_color("#111111")
        ax.grid(True, color="#CCCCCC", linewidth=0.5)
        if ax.get_title():
            ax.set_title(ax.get_title(), color="#0A2540", fontsize=10,
                         fontweight="bold", loc="left", pad=7)
        lg = ax.get_legend()
        if lg is not None:
            lg.get_frame().set_facecolor("#FFFFFF")
            lg.get_frame().set_edgecolor("#999999")
            for t in lg.get_texts():
                t.set_color("#111111")
    fig.suptitle("Feature And Label Distributions Across The Behavioural-Cloning Dataset",
                 color="#0A2540", fontsize=15, fontweight="bold", x=0.055,
                 ha="left", y=0.975)
    fig.savefig(out_light, dpi=200, facecolor="#FFFFFF")
    print(f"saved -> {out_light}   (light version for the thesis PDF)")


if __name__ == "__main__":
    main()