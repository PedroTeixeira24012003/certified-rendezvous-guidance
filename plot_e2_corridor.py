"""
plot_e2_corridor.py
Two figures of the champion flown from the worst-case corner against the
expert's own path from the identical start. Reads cl_E2_<runid>.npz and
expert_refs.npz, writes fig_e2_corridor.png (approach planes, 3D view,
terminal zoom, with the keep-out sphere and hold point) and
fig_e2_deviation.png (distance between the two paths over time with a
per-axis breakdown), plus light versions of both.
"""

import os
import sys
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REFS_FILE = 'expert_refs.npz'
DT = 0.5
R_KOZ = 5.0
HOLD = np.array([-8.0, 0.0, 0.0])

# figure palette
NAVY = "#0A2540"; CARD = "#13293F"; TEAL = "#00A8B5"; BODY = "#CADCFC"
MUTE = "#8FA3BD"; ORANGE = "#E67E22"; WHITE = "#FFFFFF"; GRID = "#2C4258"
EXPERT_C = TEAL
NET_C = ORANGE


def newest_e2():
    fs = sorted(glob.glob('cl_E2_*.npz'), key=os.path.getmtime)
    if not fs:
        sys.exit("no cl_E2_*.npz found - run deploy_closedloop.py --mode e2 first")
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
    ax.grid(True, color=ln, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def draw_koz_circle(ax, i0, i1):
    th = np.linspace(0, 2 * np.pi, 200)
    c = np.zeros((200, 3))
    c[:, i0] = R_KOZ * np.cos(th)
    c[:, i1] = R_KOZ * np.sin(th)
    ax.plot(c[:, i0], c[:, i1], color=MUTE, lw=1.2, ls='--', alpha=0.8,
            label='Keep-Out')


def corridor_figure(Xn, Xe, dark=True):
    bg = NAVY if dark else "#FFFFFF"
    fig = plt.figure(figsize=(13.0, 8.2), facecolor=bg)
    ttl = WHITE if dark else "#0A2540"

    # panel 1, V-bar (x) vs R-bar (y)
    ax1 = fig.add_subplot(2, 2, 1)
    style(ax1, dark)
    draw_koz_circle(ax1, 0, 1)
    ax1.plot(Xe[:, 0], Xe[:, 1], color=EXPERT_C, lw=1.8, label='Expert')
    ax1.plot(Xn[:, 0], Xn[:, 1], color=NET_C, lw=1.2, ls='--', label='Network')
    ax1.scatter([HOLD[0]], [HOLD[1]], color=WHITE if dark else "#000",
                s=40, marker='*', zorder=5, label='Hold Point')
    ax1.set_xlabel("V-bar  $x$  [m]"); ax1.set_ylabel("R-bar  $y$  [m]")
    ax1.set_title("A  Approach Plane (V-bar / R-bar)", color=TEAL,
                  fontsize=10, fontweight="bold", loc="left")
    ax1.axis('equal')
    lg = ax1.legend(fontsize=8, facecolor=CARD if dark else "#FFF",
                    edgecolor=GRID, framealpha=1, loc='best')
    for t in lg.get_texts():
        t.set_color(BODY if dark else "#111")

    # panel 2, V-bar (x) vs H-bar (z)
    ax2 = fig.add_subplot(2, 2, 2)
    style(ax2, dark)
    draw_koz_circle(ax2, 0, 2)
    ax2.plot(Xe[:, 0], Xe[:, 2], color=EXPERT_C, lw=1.8, label='Expert')
    ax2.plot(Xn[:, 0], Xn[:, 2], color=NET_C, lw=1.2, ls='--', label='Network')
    ax2.scatter([HOLD[0]], [HOLD[2]], color=WHITE if dark else "#000",
                s=40, marker='*', zorder=5)
    ax2.set_xlabel("V-bar  $x$  [m]"); ax2.set_ylabel("H-bar  $z$  [m]")
    ax2.set_title("B  Approach Plane (V-bar / H-bar)", color=TEAL,
                  fontsize=10, fontweight="bold", loc="left")
    ax2.axis('equal')

    # panel 3, 3D
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_facecolor(CARD if dark else "#FFFFFF")
    ax3.plot(Xe[:, 0], Xe[:, 1], Xe[:, 2], color=EXPERT_C, lw=1.8, label='Expert')
    ax3.plot(Xn[:, 0], Xn[:, 1], Xn[:, 2], color=NET_C, lw=1.2, ls='--',
             label='Network')
    u = np.linspace(0, 2 * np.pi, 24); v = np.linspace(0, np.pi, 16)
    xs = R_KOZ * np.outer(np.cos(u), np.sin(v))
    ys = R_KOZ * np.outer(np.sin(u), np.sin(v))
    zs = R_KOZ * np.outer(np.ones_like(u), np.cos(v))
    ax3.plot_surface(xs, ys, zs, color=MUTE, alpha=0.15, linewidth=0)
    ax3.scatter([HOLD[0]], [HOLD[1]], [HOLD[2]],
                color=WHITE if dark else "#000", s=40, marker='*')
    ax3.set_xlabel("$x$ [m]", color=BODY if dark else "#111", fontsize=8)
    ax3.set_ylabel("$y$ [m]", color=BODY if dark else "#111", fontsize=8)
    ax3.set_zlabel("$z$ [m]", color=BODY if dark else "#111", fontsize=8)
    ax3.tick_params(colors=MUTE if dark else "#333", labelsize=7)
    ax3.set_title("C  Three-Dimensional View", color=TEAL, fontsize=10,
                  fontweight="bold", loc="left")
    try:
        ax3.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # panel 4, terminal zoom, where the paths separate by centimetres
    ax4 = fig.add_subplot(2, 2, 4)
    style(ax4, dark)
    tail = slice(int(0.6 * len(Xe)), None)
    ax4.plot(Xe[tail, 0], Xe[tail, 1], color=EXPERT_C, lw=1.8, label='Expert')
    ax4.plot(Xn[tail, 0], Xn[tail, 1], color=NET_C, lw=1.4, ls='--',
             label='Network')
    ax4.scatter([HOLD[0]], [HOLD[1]], color=WHITE if dark else "#000",
                s=50, marker='*', zorder=5)
    ax4.set_xlabel("V-bar  $x$  [m]"); ax4.set_ylabel("R-bar  $y$  [m]")
    ax4.set_title("D  Terminal Approach, Magnified", color=TEAL, fontsize=10,
                  fontweight="bold", loc="left")

    fig.suptitle("The Champion Flown From The Worst-Case Corner, Against The "
                 "Expert's Own Path", color=ttl, fontsize=14, fontweight="bold",
                 x=0.5, y=0.975)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.07,
                        hspace=0.30, wspace=0.22)
    return fig


def deviation_figure(Xn, Xe, dark=True):
    bg = NAVY if dark else "#FFFFFF"
    ttl = WHITE if dark else "#0A2540"
    n = min(len(Xn), len(Xe))
    t = np.arange(n) * DT
    dev = np.linalg.norm(Xn[:n, :3] - Xe[:n, :3], axis=1)
    dax = Xn[:n, :3] - Xe[:n, :3]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 4.6), facecolor=bg)

    style(axL, dark)
    axL.plot(t, dev * 100, color=NET_C, lw=1.8)
    axL.axhline(dev.max() * 100, color=MUTE, ls=':', lw=1.0)
    axL.text(t[-1] * 0.02, dev.max() * 100 * 1.02,
             f"peak {dev.max()*100:.1f} cm", color=MUTE, fontsize=8.5)
    axL.set_xlabel("Time  [s]")
    axL.set_ylabel("Distance Between Paths  [cm]")
    axL.set_title("A  Total Deviation Over The Flight", color=TEAL, fontsize=10,
                  fontweight="bold", loc="left")

    style(axR, dark)
    for i, (lab, c) in enumerate(zip(("$x$ (V-bar)", "$y$ (R-bar)", "$z$ (H-bar)"),
                                     (ORANGE, TEAL, BODY if dark else "#333"))):
        axR.plot(t, dax[:, i] * 100, color=c, lw=1.4, label=lab)
    axR.axhline(0, color=MUTE, lw=0.8)
    axR.set_xlabel("Time  [s]")
    axR.set_ylabel("Signed Deviation Per Axis  [cm]")
    axR.set_title("B  Deviation By Axis", color=TEAL, fontsize=10,
                  fontweight="bold", loc="left")
    lg = axR.legend(fontsize=8, facecolor=CARD if dark else "#FFF",
                    edgecolor=GRID, framealpha=1)
    for tt in lg.get_texts():
        tt.set_color(BODY if dark else "#111")

    fig.suptitle("Deviation From The Expert Path Does Not Grow: The Signature Of "
                 "Settled, Not Compounding, Error", color=ttl, fontsize=13,
                 fontweight="bold", x=0.5, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.13, wspace=0.22)
    return fig, dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runid', type=str, default=None)
    args = p.parse_args()

    e2 = f'cl_E2_{args.runid}.npz' if args.runid else newest_e2()
    if not os.path.exists(e2):
        sys.exit(f"{e2} not found")
    if not os.path.exists(REFS_FILE):
        sys.exit(f"{REFS_FILE} not found")

    print(f"loading {e2} and {REFS_FILE} ...")
    d = np.load(e2)
    refs = np.load(REFS_FILE)
    Xn = np.asarray(d['X'], dtype=np.float64)     # network worst-case states
    Xe = np.asarray(refs['wc_X'], dtype=np.float64)  # expert worst-case states
    print(f"  network path {Xn.shape}, expert path {Xe.shape}")

    for dark, suffix, dpi in [(True, '', 150), (False, '_light', 200)]:
        fc = corridor_figure(Xn, Xe, dark)
        f1 = f'fig_e2_corridor{suffix}.png'
        fc.savefig(f1, dpi=dpi, facecolor=(NAVY if dark else "#FFFFFF"))
        print(f"  saved -> {f1}")

        fd, dev = deviation_figure(Xn, Xe, dark)
        f2 = f'fig_e2_deviation{suffix}.png'
        fd.savefig(f2, dpi=dpi, facecolor=(NAVY if dark else "#FFFFFF"))
        print(f"  saved -> {f2}")

    n = min(len(Xn), len(Xe))
    dev = np.linalg.norm(Xn[:n, :3] - Xe[:n, :3], axis=1)
    print(f"\n  deviation: peak {dev.max()*100:.2f} cm  "
          f"mean {dev.mean()*100:.2f} cm  final {dev[-1]*100:.2f} cm")
    print(f"  peak occurs at t = {np.argmax(dev)*DT:.1f} s "
          f"(step {np.argmax(dev)} of {n})")


if __name__ == "__main__":
    main()