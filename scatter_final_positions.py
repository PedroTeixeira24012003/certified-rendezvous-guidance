"""
scatter_final_positions.py
Per-flight, per-axis final offset from the hold point. Re-flies a checkpoint
on the requested rung and plots the miss-cloud against the TOL_POS ball for
each axis pair, converged green, missed red, with per-axis mean and std. It
reuses the ladder's exact flight path (DisturbanceTape, paired reference ICs,
disturbed RK4) so the numbers match the ladder run; it only additionally
records the final offset as a vector instead of collapsing to a norm.
Writes final_scatter_<runid>_<rung>.png per checkpoint.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import robustness_ladder as RL
import deploy_closedloop as DCL
import th_mpc_behind_1000run as EXP


def fly_and_record(ckpt, rung, n):
    """Re-fly the checkpoint on `rung` for n conditions; return per-flight final
    offset vectors (n,3) and a converged mask (n,)."""
    model, cfg, _ = DCL.load_checkpoint(ckpt)
    import bc_data
    x_mean, x_std = bc_data.load_norm_stats()
    refs = np.load(DCL.REFS_FILE)           # same paired thousand the ladder uses
    lcfg = RL.LADDER[rung]
    n = min(n, int(refs['n_mc']))
    mc_x0, mc_th = refs['mc_x0'], refs['mc_theta0']
    mc_a, mc_e = refs['mc_a'], refs['mc_e']

    offs = np.zeros((n, 3)); conv = np.zeros(n, bool); fverr = np.zeros(n)
    for i in range(n):
        tape = RL.DisturbanceTape(rung, i, lcfg)
        # replicate fly_network_dist but keep the final TRUE state vector
        state = np.concatenate([mc_x0[i], [float(mc_th[i])]])
        Xh = [state.copy()]; Uh = []
        import models
        is_hist = isinstance(model, models.HistoryPolicy)
        if is_hist:
            fl, al = [], []; kw = model.k
        div = False
        for _ in range(EXP.SIM_STEPS):
            Rr = np.linalg.norm(state[:3])
            sm = state.copy(); sm[:6] = state[:6] + tape.nav_noise(Rr)
            if is_hist:
                u = DCL.policy_command_history(model, sm, float(mc_a[i]),
                                               float(mc_e[i]), x_mean, x_std, fl, al, kw)
            else:
                u = DCL.policy_command(model, sm, float(mc_a[i]), float(mc_e[i]),
                                       x_mean, x_std)
            ua = tape.thrust_apply(u); Uh.append(ua.copy())
            state = RL.rk4_step_disturbed(state, ua, EXP.DT, float(mc_a[i]),
                                          float(mc_e[i]), tape.env_fn(float(mc_a[i]), float(mc_e[i])))
            Xh.append(state.copy())
            if not np.all(np.isfinite(state)) or \
               np.linalg.norm(state[:3] - EXP.HOLD) > DCL.DIVERGE_DIST:
                div = True; break
        Xf = np.array(Xh)[-1, :3]
        off = Xf - EXP.HOLD
        offs[i] = off
        fverr[i] = np.linalg.norm(off)
        fvel = np.linalg.norm(np.array(Xh)[-1, 3:6]) if not div else 9e9
        conv[i] = (not div) and fverr[i] < EXP.TOL_POS and fvel < EXP.TOL_VEL
    return offs, conv, fverr, cfg.get('tag', ckpt)


def plot_clouds(offs, conv, tag, rung, tol, fname):
    """3 panels: x-y, x-z, y-z. Converged green, missed red, tol circle drawn."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0, 1, 'x', 'y'), (0, 2, 'x', 'z'), (1, 2, 'y', 'z')]
    for ax, (a, b, la, lb) in zip(axes, pairs):
        ax.scatter(offs[~conv, a], offs[~conv, b], s=6, c='crimson',
                   alpha=0.4, label='missed')
        ax.scatter(offs[conv, a], offs[conv, b], s=6, c='seagreen',
                   alpha=0.5, label='converged')
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(tol * np.cos(th), tol * np.sin(th), 'k--', lw=1,
                label=f'TOL {tol} m')
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel(f'{la} offset [m]'); ax.set_ylabel(f'{lb} offset [m]')
        ax.set_aspect('equal', 'box'); ax.set_title(f'{la}-{lb}')
        # per-axis stats in the corner
        ax.text(0.02, 0.98, f'{la}: mu{offs[:,a].mean():+.2f} sd{offs[:,a].std():.2f}\n'
                            f'{lb}: mu{offs[:,b].mean():+.2f} sd{offs[:,b].std():.2f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    axes[0].legend(loc='lower right', fontsize=8)
    conv_rate = int(conv.sum())
    fig.suptitle(f'{tag}  |  {rung}  |  {conv_rate}/{len(conv)} converged  '
                 f'(TOL={tol} m)')
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"  saved {fname}")
    # per-axis summary to stdout
    print(f"  PER-AXIS final offset (all {len(conv)} flights):")
    for j, nm in enumerate('xyz'):
        print(f"    {nm}:  mean {offs[:,j].mean():+.3f} m   std {offs[:,j].std():.3f} m   "
              f"|mean| {abs(offs[:,j].mean()):.3f}")
    print(f"  interpretation key: large |mean| on an axis = uncorrected SHIFT on that axis; "
          f"large std = SCATTER/over-correction on that axis.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--rung', default='L3')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--compare', default=None, help='second ckpt to plot alongside')
    args = ap.parse_args()
    tol = EXP.TOL_POS

    for ck in [args.ckpt] + ([args.compare] if args.compare else []):
        print(f"\nflying {ck} on {args.rung} ...")
        offs, conv, fverr, tag = fly_and_record(ck, args.rung, args.n)
        runid = ck.split('/')[-1].replace('.pt', '')
        fname = f"final_scatter_{runid}_{args.rung}.png"
        plot_clouds(offs, conv, tag, args.rung, tol, fname)


if __name__ == "__main__":
    main()