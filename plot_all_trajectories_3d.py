"""
plot_all_trajectories_3d.py
One three-panel figure per controller (seven lambda_ta policies plus the
expert), flights re-flown on the deterministic crc32 tapes at the requested
rung. Left, the full 3D approach fan with the initial-condition cylinder.
Middle, a tight 3D close-up of the keep-out-to-hold corridor. Right, a 2D top
view of the x-y plane. Trajectories are coloured by outcome, green converged,
red not. Writes traj3d_<rung>_<label>.png at 300 dpi.
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import th_mpc_behind_1000run as EXP
import deploy_closedloop as DCL
import bc_data
import robustness_ladder as RL

MANIFEST = [
    ("ta=0.01",  "checkpoints/20260813-141246-4t-cc7328.pt"),
    ("ta=0.03",  "checkpoints/20260813-152458-4t-6fc9e3.pt"),
    ("ta=0.1",   "checkpoints/20260813-154058-4t-68990a.pt"),
    ("ta=0.3",   "checkpoints/20260813-161306-4t-cea077.pt"),
    ("ta=1.0",   "checkpoints/20260813-163542-4t-d2afae.pt"),
    ("ta=5.0",   "checkpoints/20260813-171313-4t-b1fc55.pt"),
    ("ta=8.0",   "checkpoints/20260813-183326-4t-828492.pt"),
    ("expert",   "expert"),
]

C_CONV = "#00c400"
C_FAIL = "#ff1010"
C_KOZ = "#e02020"          # red keep-out sphere
C_TOL = "#1ea6ff"          # light-blue tolerance sphere
C_CYL = "#7cc4ff"          # light-blue initial-condition cylinder
C_SAT = "#0b1f66"          # dark-blue satellite

SAT_HALF = 0.75
PANEL_LEN = 3.0
PANEL_W = 1.0
PANEL_T = 0.06


# ------------------------------------------------------------------ flight
def fly_capture(controller, model, solver, x0, theta0, a, e, x_mean, x_std, tape):
    state = np.concatenate([x0, [theta0]])
    X = [state.copy()]
    env_fn = tape.env_fn(a, e)
    p_val = np.array([a, e])
    divergent = False
    for k in range(EXP.SIM_STEPS):
        R = np.linalg.norm(state[:3])
        meas = state.copy()
        meas[:6] = state[:6] + tape.nav_noise(R)
        if controller == "expert":
            for j in range(EXP.N_P + 1):
                solver.set(j, "p", p_val)
            solver.set(0, "lbx", meas)
            solver.set(0, "ubx", meas)
            solver.solve()
            u = solver.get(0, "u")
        else:
            u = DCL.policy_command(model, meas, a, e, x_mean, x_std)
        u_app = tape.thrust_apply(u)
        state = RL.rk4_step_disturbed(state, u_app, EXP.DT, a, e, env_fn)
        X.append(state.copy())
        if not np.all(np.isfinite(state)):
            divergent = True
            break
        if np.linalg.norm(state[:3] - EXP.HOLD) > DCL.DIVERGE_DIST:
            divergent = True
            break
    X = np.array(X)
    fpe = float(np.linalg.norm(X[-1, :3] - EXP.HOLD))
    fvel = float(np.linalg.norm(X[-1, 3:6]))
    converged = (not divergent) and fpe < EXP.TOL_POS and fvel < EXP.TOL_VEL
    return X, converged


# ------------------------------------------------------------------ 3D depth line
def add_depth_line(ax, pts, base_rgb, cam_dir, a_near, a_far, lw, clip_box=None):
    if clip_box is not None:
        lo, hi = clip_box
        m = np.all((pts[:, :3] >= lo) & (pts[:, :3] <= hi), axis=1)
        if m.sum() < 2:
            return
        pts = pts[m]
    p = pts[:, :3]
    if len(p) < 2:
        return
    segs = np.stack([p[:-1], p[1:]], axis=1)
    mids = 0.5 * (p[:-1] + p[1:])
    depth = mids @ cam_dir
    d0, d1 = depth.min(), depth.max()
    t = (depth - d0) / (d1 - d0 + 1e-9)
    alphas = a_far + t * (a_near - a_far)
    colors = np.zeros((len(segs), 4))
    colors[:, 0], colors[:, 1], colors[:, 2] = base_rgb
    colors[:, 3] = alphas
    ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=lw))


# ------------------------------------------------------------------ scene bits
def sphere_surface(ax, c, r, color, alpha, n=32, zo=0):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = c[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = c[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0,
                    antialiased=True, shade=True, zorder=zo)


def sphere_wire(ax, c, r, color, lw, alpha, n=16, zo=20):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = c[0] + r * np.outer(np.cos(u), np.sin(v))
    ys = c[1] + r * np.outer(np.sin(u), np.sin(v))
    zs = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color=color, linewidth=lw, alpha=alpha,
                      rcount=n, ccount=n, zorder=zo)


def box(ax, c, hx, hy, hz, color, alpha=1.0):
    cx, cy, cz = c
    v = np.array([[cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
                  [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
                  [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
                  [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz]])
    f = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
         [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
         [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]
    pc = Poly3DCollection(f, facecolor=color, edgecolor="k", linewidths=0.3,
                          alpha=alpha)
    pc.set_zsort("max")
    ax.add_collection3d(pc)


def cylinder_x(ax, x0, x1, radius, center_yz, color, alpha, n=28):
    th = np.linspace(0, 2 * np.pi, n)
    xs = np.linspace(x0, x1, 2)
    thg, xg = np.meshgrid(th, xs)
    yg = center_yz[0] + radius * np.cos(thg)
    zg = center_yz[1] + radius * np.sin(thg)
    ax.plot_surface(xg, yg, zg, color=color, alpha=alpha, linewidth=0,
                    antialiased=True, shade=True, zorder=0)


def draw_scene_base(ax, ic, koz_n=32, draw_cyl=True):
    """Everything EXCEPT the tolerance sphere (that goes on top, last)."""
    sphere_surface(ax, (0, 0, 0), EXP.R_KOZ, C_KOZ, 0.16, n=koz_n, zo=0)
    box(ax, (0, 0, 0), SAT_HALF, SAT_HALF, SAT_HALF, C_SAT, 1.0)
    box(ax, (0,  SAT_HALF + PANEL_LEN / 2, 0),
        PANEL_W / 2, PANEL_LEN / 2, PANEL_T / 2, "black", 1.0)
    box(ax, (0, -(SAT_HALF + PANEL_LEN / 2), 0),
        PANEL_W / 2, PANEL_LEN / 2, PANEL_T / 2, "black", 1.0)
    if draw_cyl:
        cylinder_x(ax, ic["x0"], ic["x1"], ic["r"], ic["yz"], C_CYL, 0.18)


def draw_tolerance_on_top(ax, tol_n=22):
    """Draw the hold dot + tolerance sphere LAST so lines never bury it:
    light fill for volume + bright wireframe for the outline, high zorder."""
    sphere_surface(ax, EXP.HOLD, EXP.TOL_POS, C_TOL, 0.22, n=tol_n, zo=18)
    sphere_wire(ax, EXP.HOLD, EXP.TOL_POS, C_TOL, lw=1.1, alpha=0.95,
                n=tol_n, zo=21)
    ax.scatter(*EXP.HOLD, color="k", marker="o", s=18, zorder=22)


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def set_equal(ax, lo, hi, pad=1.05):
    ctr = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max() * pad
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    ax.set_box_aspect((1, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="L3", choices=list(RL.LADDER.keys()))
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-58.0)
    ap.add_argument("--zoom", type=float, default=11.0,
                    help="(reserved) close-up is fixed to x[-11,5], y,z[-7,7]")
    args = ap.parse_args()
    rung = args.rung
    cfg = RL.LADDER[rung]

    refs = np.load(DCL.REFS_FILE)
    x_mean, x_std = bc_data.load_norm_stats()
    n = int(refs["n_mc"])
    mc_x0 = refs["mc_x0"]; mc_th = refs["mc_theta0"]
    mc_a = refs["mc_a"]; mc_e = refs["mc_e"]

    x0s = mc_x0[:, 0]
    yz = mc_x0[:, 1:3]
    ic_yz = (float(yz[:, 0].mean()), float(yz[:, 1].mean()))
    ic = dict(x0=float(x0s.min()), x1=float(x0s.max()),
              yz=ic_yz,
              r=max(float(np.percentile(
                  np.linalg.norm(yz - np.array(ic_yz), axis=1), 98)), 1.0))
    print(f"  Initial cylinder: x in [{ic['x0']:.1f}, {ic['x1']:.1f}] m, "
          f"radius {ic['r']:.2f} m")

    entries = [(l, c) for l, c in MANIFEST
               if args.only is None or l in args.only]

    el = np.radians(args.elev); az = np.radians(args.azim)
    cam_dir = np.array([np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az), np.sin(el)])
    cam_dir /= np.linalg.norm(cam_dir)
    rgb_conv, rgb_fail = hexrgb(C_CONV), hexrgb(C_FAIL)

    # tight close-up corridor (x behind hold -> front of KOZ; y,z compressed)
    cl_lo = np.array([-11.0, -7.0, -7.0])
    cl_hi = np.array([5.0, 7.0, 7.0])

    solver = None
    for label, ckpt in entries:
        controller = "expert" if ckpt == "expert" else "raw"
        model = None
        if controller == "expert":
            if solver is None:
                solver = EXP.build_solver()
        else:
            model, _, _ = DCL.load_checkpoint(ckpt)

        print(f"\n  {label}: re-flying {n} at {rung} "
              f"({'~45 min' if controller == 'expert' else '~1 min'})")
        t0 = time.time()
        trajs = []
        n_conv = 0
        for i in range(n):
            tape = RL.DisturbanceTape(rung, i, cfg)
            X, conv = fly_capture(controller, model, solver, mc_x0[i],
                                  float(mc_th[i]), float(mc_a[i]),
                                  float(mc_e[i]), x_mean, x_std, tape)
            trajs.append((X[::args.stride], conv))
            n_conv += int(conv)
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"    {i+1}/{n}  {rate:.1f} fl/s  "
                      f"eta {(n-i-1)/rate/60:.1f} min  conv {n_conv}")
        pct = 100.0 * n_conv / n

        fig = plt.figure(figsize=(21, 7.5))
        axL = fig.add_subplot(131, projection="3d")
        axM = fig.add_subplot(132, projection="3d")
        axR = fig.add_subplot(133)                       # 2D top view
        for ax in (axL, axM):
            ax.view_init(elev=args.elev, azim=args.azim)

        # ---- LEFT: full approach (3D) ----
        draw_scene_base(axL, ic, koz_n=24, draw_cyl=True)
        allP = []
        for X, conv in trajs:
            if not conv:
                add_depth_line(axL, X, rgb_fail, cam_dir, 0.30, 0.04, 0.5)
                allP.append(X[:, :3])
        for X, conv in trajs:
            if conv:
                add_depth_line(axL, X, rgb_conv, cam_dir, 0.55, 0.10, 0.6)
                allP.append(X[:, :3])
        draw_tolerance_on_top(axL, tol_n=14)
        # start-point dots on the initial-condition cylinder (same colour code)
        s_conv = np.array([X[0, :3] for X, c in trajs if c])
        s_fail = np.array([X[0, :3] for X, c in trajs if not c])
        if len(s_fail):
            axL.scatter(s_fail[:, 0], s_fail[:, 1], s_fail[:, 2], s=4,
                        c=C_FAIL, alpha=0.8, depthshade=True, linewidths=0)
        if len(s_conv):
            axL.scatter(s_conv[:, 0], s_conv[:, 1], s_conv[:, 2], s=4,
                        c=C_CONV, alpha=0.9, depthshade=True, linewidths=0)
        P = np.vstack(allP + [np.array([[ic["x0"], 0, 0], list(EXP.HOLD),
                                        [EXP.R_KOZ, EXP.R_KOZ, EXP.R_KOZ]])])
        set_equal(axL, P.min(axis=0), P.max(axis=0))
        axL.set_title("Full approach", fontsize=12)
        axL.set_xlabel("X, V-bar [m]"); axL.set_ylabel("Y [m]")
        axL.set_zlabel("Z [m]")

        # ---- MIDDLE: hold close-up (3D), tight corridor ----
        # frame just the KOZ-to-hold corridor: x from just behind the hold
        # (-11) to the front face of the keep-out sphere (+5); y,z compressed.
        draw_scene_base(axM, ic, koz_n=40, draw_cyl=False)
        for X, conv in trajs:
            if not conv:
                add_depth_line(axM, X, rgb_fail, cam_dir, 0.30, 0.05, 0.5,
                               clip_box=(cl_lo, cl_hi))
        for X, conv in trajs:
            if conv:
                add_depth_line(axM, X, rgb_conv, cam_dir, 0.7, 0.15, 0.7,
                               clip_box=(cl_lo, cl_hi))
        draw_tolerance_on_top(axM, tol_n=26)
        # explicit (non-equal) limits so the corridor fills the panel
        axM.set_xlim(cl_lo[0], cl_hi[0])
        axM.set_ylim(cl_lo[1], cl_hi[1])
        axM.set_zlim(cl_lo[2], cl_hi[2])
        axM.set_box_aspect((cl_hi[0] - cl_lo[0],
                            cl_hi[1] - cl_lo[1],
                            cl_hi[2] - cl_lo[2]))
        axM.set_title("Hold and keep-out close-up", fontsize=12)
        axM.set_xlabel("X, V-bar [m]"); axM.set_ylabel("Y [m]")
        axM.set_zlabel("Z [m]")

        # ---- RIGHT: 2D top view (x-y), looking down z ----
        # keep-out and tolerance as circles; satellite as a small square
        axR.add_patch(Circle((0, 0), EXP.R_KOZ, facecolor=C_KOZ, alpha=0.16,
                             edgecolor=C_KOZ, lw=1.2, zorder=0))
        axR.add_patch(plt.Rectangle((-SAT_HALF, -SAT_HALF), 2 * SAT_HALF,
                                    2 * SAT_HALF, facecolor=C_SAT, zorder=3))
        # solar panels along +/- y (top view: rectangles, width PANEL_W in x)
        axR.add_patch(plt.Rectangle((-PANEL_W / 2, SAT_HALF),
                                    PANEL_W, PANEL_LEN,
                                    facecolor="black", zorder=3))
        axR.add_patch(plt.Rectangle((-PANEL_W / 2, -(SAT_HALF + PANEL_LEN)),
                                    PANEL_W, PANEL_LEN,
                                    facecolor="black", zorder=3))
        for X, conv in trajs:
            col = C_CONV if conv else C_FAIL
            a_ = 0.22 if conv else 0.12
            axR.plot(X[:, 0], X[:, 1], color=col, lw=0.4, alpha=a_,
                     zorder=1 if not conv else 2)
        # small endpoint dot at the final point of each trajectory (same colour)
        end_conv = np.array([X[-1, :2] for X, c in trajs if c])
        end_fail = np.array([X[-1, :2] for X, c in trajs if not c])
        if len(end_fail):
            axR.scatter(end_fail[:, 0], end_fail[:, 1], s=3, c=C_FAIL,
                        alpha=0.9, zorder=4, linewidths=0)
        if len(end_conv):
            axR.scatter(end_conv[:, 0], end_conv[:, 1], s=3, c=C_CONV,
                        alpha=0.95, zorder=5, linewidths=0)
        # tolerance region: very light, transparent fill + strong black contour
        axR.add_patch(Circle(EXP.HOLD[:2], EXP.TOL_POS, facecolor="#eaf6ff",
                             alpha=0.35, edgecolor="black", lw=2.2, zorder=6))
        axR.plot(EXP.HOLD[0], EXP.HOLD[1], "ko", ms=4, zorder=7)
        axR.set_aspect("equal")
        # same tight corridor as the 3D close-up: x -11..+5, y -7..+7
        axR.set_xlim(-11.0, 5.0)
        axR.set_ylim(-7.0, 7.0)
        axR.set_xlabel("X, V-bar [m]"); axR.set_ylabel("Y [m]")
        axR.set_title("Top view (X-Y plane)", fontsize=12)
        axR.grid(alpha=0.3)

        # ---- figure legend (no suptitle) ----
        fig.legend(handles=[
            Line2D([0], [0], color=C_CONV, lw=2,
                   label=f"Converged ({n_conv}/{n}, {pct:.1f}%)"),
            Line2D([0], [0], color=C_FAIL, lw=2, label="Not converged"),
            Line2D([0], [0], color=C_KOZ, lw=6, alpha=0.4,
                   label="Keep-out zone (5 m)"),
            Line2D([0], [0], color=C_TOL, lw=6, alpha=0.5,
                   label="Tolerance (0.5 m)"),
            Line2D([0], [0], color=C_CYL, lw=6, alpha=0.4,
                   label="Initial cylinder"),
        ], loc="lower center", ncol=5, fontsize=11, frameon=True,
            bbox_to_anchor=(0.5, -0.02))

        fig.tight_layout(rect=(0, 0.04, 1, 1))
        out = f"traj3d_{rung}_{label.replace('=', '')}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved -> {out}   conv {n_conv}/{n}   "
              f"[{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()