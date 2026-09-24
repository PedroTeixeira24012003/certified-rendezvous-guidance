"""
deploy_closedloop.orig.py
Closed-loop evaluator. Puts the trained network in the expert's place and
flies it, at every control instant the state becomes the ten features,
normalized as in training, and the network's physical command drives the
same plant the expert flew. Results are compared paired against the cached
expert references. With --filter the CBF-CLF-QP safety filter certifies
every command before it is applied; without it the run is unchanged.

Nothing is re-implemented, the plant and constants come from
th_mpc_behind_1000run, the feature builder from generate_dataset_behind,
the normalization from bc_data, the model builders from train_policy and
train_history, the filter from cbf_filter and the references from
expert_refs.npz.

Modes, E2 flies the worst-case corner and measures deviation from the
expert's own path, E3 flies the 1000 cached conditions and computes paired
deltas. A run is divergent if the distance to the hold point exceeds 200 m
at any step or any state is non-finite. Outputs cl_E2_<runid>.npz,
cl_E3_<runid>.npz and one summary row per evaluation in closedloop_log.csv.
"""

import os
import sys
import csv
import glob
import json
import time
import argparse
import numpy as np
import torch

import th_mpc_behind_1000run as EXP          # plant, constants, v_cap
import generate_dataset_behind as GEN        # make_features (training-identical)
import bc_data                               # normalization (training-identical)
import train_policy                          # build_model (training-identical)
import train_history                         # build_history_model
import models                                # HistoryPolicy type check
import cbf_filter                            # CBF-CLF-QP safety filter

REFS_FILE = 'expert_refs.npz'
LOG_FILE = 'closedloop_log.csv'
DIVERGE_DIST = 200.0                          # m, distance to hold


# ============================================================================
# Policy wrapper, state -> command, through the training-identical pipeline
# ============================================================================
def policy_command(model, state7, a, e, x_mean, x_std):
    """One network evaluation: exact feature build, exact normalization,
    physical command out (the tanh head already confines it to the thrust box)."""
    feats = GEN.make_features(state7, a, e)               # (10,)
    xn = bc_data.normalize_x(feats, x_mean, x_std)        # training transform
    with torch.no_grad():
        x_t = torch.from_numpy(xn.astype(np.float32)).unsqueeze(0)
        u = model(x_t, scaled=False).squeeze(0).numpy()   # m/s^2, in the box
    return u.astype(np.float64)


# ============================================================================
# Windowed flight support for the history policy. The flight window follows
# the training loader's conventions exactly, rows are the last k steps of
# (normalized features, scaled action) oldest first, pre-flight steps are
# padded by repeating the initial features with zero actions, the current
# step's action slot is zero, and past actions are the network's own previous
# commands scaled and clamped as the training targets were.
# build_flight_window is a pure function of the running histories.
# ============================================================================
def build_flight_window(feats_list, acts_scaled_list, s, k):
    """Window ending at step s. feats_list holds normalized features for steps
    0..s (current included). acts_scaled_list holds scaled commands for steps
    0..s-1 (the current command does not exist yet). Returns (k, 13) float32."""
    rows = np.zeros((k, 13), dtype=np.float32)
    for j_out, j in enumerate(range(s - k + 1, s + 1)):
        if j < 0:
            rows[j_out, :10] = feats_list[0]     # pad: repeat initial features
            # action stays zero (pad convention)
        else:
            rows[j_out, :10] = feats_list[j]
            if j < s:                            # past step: its command exists
                rows[j_out, 10:] = acts_scaled_list[j]
            # j == s: current step, action slot stays zero
    return rows


def policy_command_history(model, state7, a, e, x_mean, x_std,
                           feats_list, acts_scaled_list, k):
    """One history-network evaluation. Appends the current normalized features
    to feats_list, builds the window, runs the two-input forward, records the
    scaled command into acts_scaled_list, and returns the physical command.

    When a safety filter is active the RAW network command is what goes into
    the window; the filter is applied afterwards, on the returned command, by
    the caller. This keeps the history distribution identical to training."""
    feats = GEN.make_features(state7, a, e)
    fn = bc_data.normalize_x(feats, x_mean, x_std).astype(np.float32)
    feats_list.append(fn)
    s = len(feats_list) - 1

    x_hist = build_flight_window(feats_list, acts_scaled_list, s, k)
    with torch.no_grad():
        xh_t = torch.from_numpy(x_hist).unsqueeze(0)          # (1, k, 13)
        xn_t = torch.from_numpy(fn).unsqueeze(0)              # (1, 10)
        u = model(xh_t, xn_t, scaled=False).squeeze(0).numpy()  # m/s^2

    # record the command the network just issued, scaled exactly as training
    # scaled its targets, for the next step's window
    a_scaled = np.clip(u / bc_data.U_MAX,
                       -bc_data.CLAMP, bc_data.CLAMP).astype(np.float32)
    acts_scaled_list.append(a_scaled)
    return u.astype(np.float64)


# ============================================================================
# Fly the network closed-loop from one initial condition, mirroring the
# expert's run_closed_loop with the network in place of the solver, plus the
# fixed divergence rule. When filt is given, every network command is
# certified by the safety filter before it is applied.
# ============================================================================
def fly_network(model, x0_rel, theta0, a, e, x_mean, x_std,
                sim_steps=None, filt=None):
    if sim_steps is None:
        sim_steps = EXP.SIM_STEPS
    state = np.concatenate([x0_rel, [theta0]])
    X_hist = [state.copy()]
    U_hist = []
    arrival_step = None
    divergent = False

    # per-run filter instrumentation (only meaningful when filt is not None)
    n_intervene = 0
    n_fallback = 0
    corr_sum = 0.0
    corr_max = 0.0
    solve_us_sum = 0.0
    solve_us_max = 0.0

    # history models keep a rolling window of their own flight
    is_history = isinstance(model, models.HistoryPolicy)
    if is_history:
        feats_list, acts_scaled_list = [], []
        k_win = model.k

    for k in range(sim_steps):
        if is_history:
            u = policy_command_history(model, state, a, e, x_mean, x_std,
                                       feats_list, acts_scaled_list, k_win)
        else:
            u = policy_command(model, state, a, e, x_mean, x_std)

        # certify the network's command before applying it. For a history
        # model the RAW command was already recorded into the window above,
        # so filtering only the applied command keeps the history
        # distribution identical to training.
        if filt is not None:
            fout = filt.filter(state, u, a, e)
            u = fout['u']
            if fout['intervened']:
                n_intervene += 1
                corr_sum += fout['correction']
                corr_max = max(corr_max, fout['correction'])
            if fout['fallback'] != 0:
                n_fallback += 1
            solve_us_sum += fout['solve_us']
            solve_us_max = max(solve_us_max, fout['solve_us'])

        U_hist.append(u.copy())
        state = EXP.rk4_step(state, u, EXP.DT, a, e)
        X_hist.append(state.copy())

        # fixed divergence rule: distance to hold > 200 m or non-finite state
        if not np.all(np.isfinite(state)):
            divergent = True
            break
        if np.linalg.norm(state[:3] - EXP.HOLD) > DIVERGE_DIST:
            divergent = True
            break

        # arrival check, same tolerances as the expert, flight continues
        pos_err = np.linalg.norm(state[:3] - EXP.HOLD)
        vel = np.linalg.norm(state[3:6])
        if arrival_step is None and pos_err < EXP.TOL_POS and vel < EXP.TOL_VEL:
            arrival_step = k + 1

    X = np.array(X_hist)
    U = np.array(U_hist)

    # figures of merit, identical definitions to the expert's
    final_pos_err = np.linalg.norm(X[-1, :3] - EXP.HOLD)
    final_vel = np.linalg.norm(X[-1, 3:6])
    dist_target = np.linalg.norm(X[:, :3], axis=1)
    min_dist_target = float(np.min(dist_target))
    koz_ok = bool(min_dist_target >= EXP.R_KOZ - 1e-6)
    koz_margin = min_dist_target - EXP.R_KOZ
    speed = np.linalg.norm(X[:, 3:6], axis=1)
    peak_speed = float(np.max(speed))
    caps = np.array([EXP.v_cap_np(d) for d in dist_target])
    vcap_viol = float(np.max(np.maximum(speed - caps, 0.0)))
    fuel = float(np.sum(np.linalg.norm(U, axis=1)) * EXP.DT)
    effort = float(np.sum(np.sum(U ** 2, axis=1)) * EXP.DT)
    settling_time = arrival_step * EXP.DT if arrival_step is not None else np.nan
    converged = bool((not divergent)
                     and final_pos_err < EXP.TOL_POS
                     and final_vel < EXP.TOL_VEL)

    nsteps = len(U)
    return {
        'X': X, 'U': U,
        'final_pos_err': float(final_pos_err), 'final_vel': float(final_vel),
        'min_dist_target': min_dist_target, 'koz_margin': float(koz_margin),
        'koz_ok': koz_ok, 'peak_speed': peak_speed, 'vcap_viol': vcap_viol,
        'fuel': fuel, 'effort': effort, 'settling_time': settling_time,
        'converged': converged, 'divergent': divergent, 'n_steps': nsteps,
        # filter stats (zeros when no filter was used)
        'filt_intervene_rate': (n_intervene / nsteps) if nsteps else 0.0,
        'filt_n_fallback': n_fallback,
        'filt_corr_mean': (corr_sum / n_intervene) if n_intervene else 0.0,
        'filt_corr_max': corr_max,
        'filt_solve_us_mean': (solve_us_sum / nsteps) if nsteps else 0.0,
        'filt_solve_us_max': solve_us_max,
    }


# ============================================================================
# Checkpoint loading through the training-identical builder
# ============================================================================
def build_policy(cfg):
    """Build any architecture from its training config, routing through the
    same builder that trained it so construction cannot drift."""
    if cfg.get('arch') == 'history':
        return train_history.build_history_model(cfg)
    return train_policy.build_model(cfg)


def load_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = build_policy(cfg)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, cfg, ck


def latest_checkpoint():
    cks = sorted(glob.glob('checkpoints/*.pt'), key=os.path.getmtime)
    if not cks:
        sys.exit("no checkpoints found in checkpoints/")
    return cks[-1]


# ============================================================================
# E2, the worst case, network vs the cached expert trajectory
# ============================================================================
def run_e2(model, refs, x_mean, x_std, run_id, filt=None):
    x0 = refs['wc_x0'][:6]
    theta0 = float(refs['wc_theta0'])
    a = float(refs['wc_a'])
    e = float(refs['wc_e'])

    print("  E2: flying the worst-case corner...")
    res = fly_network(model, x0, theta0, a, e, x_mean, x_std, filt=filt)

    # deviation from the expert's own path, time-aligned
    Xe = refs['wc_X']                       # (401, 7) expert states
    Ue = refs['wc_U']                       # (400, 3) expert commands
    n = min(res['X'].shape[0], Xe.shape[0])
    dev = np.linalg.norm(res['X'][:n, :3] - Xe[:n, :3], axis=1)
    max_dev = float(np.max(dev))
    mean_dev = float(np.mean(dev))
    m = min(res['U'].shape[0], Ue.shape[0])
    cmd_rms = float(np.sqrt(np.mean(np.sum((res['U'][:m] - Ue[:m]) ** 2, axis=1))))

    print(f"    converged            : {res['converged']}"
          f"{'  (DIVERGENT)' if res['divergent'] else ''}")
    print(f"    final pos err        : {res['final_pos_err']:.4f} m")
    print(f"    min dist to target   : {res['min_dist_target']:.3f} m  "
          f"(keep-out {'OK' if res['koz_ok'] else 'VIOLATED'}, "
          f"margin {res['koz_margin']:+.3f} m)")
    print(f"    vel-cap violation    : {res['vcap_viol']:.4f} m/s")
    print(f"    fuel / effort        : {res['fuel']:.3f} m/s / {res['effort']:.4f}")
    print(f"    settling time        : {res['settling_time']:.1f} s"
          if not np.isnan(res['settling_time']) else
          "    settling time        : not reached")
    print(f"    max deviation from expert path  : {max_dev:.3f} m")
    print(f"    mean deviation from expert path : {mean_dev:.3f} m")
    print(f"    per-step command RMS difference : {cmd_rms:.5f} m/s^2")
    if filt is not None:
        print(f"    filter intervention rate        : "
              f"{100*res['filt_intervene_rate']:.1f}%   "
              f"fallbacks {res['filt_n_fallback']}")
        print(f"    filter correction  mean/max     : "
              f"{res['filt_corr_mean']:.5f} / {res['filt_corr_max']:.5f} m/s^2")
        print(f"    filter solve  mean/max          : "
              f"{res['filt_solve_us_mean']:.1f} / {res['filt_solve_us_max']:.1f} us")

    out = f'cl_E2_{run_id}.npz'
    np.savez_compressed(out, X=res['X'].astype(np.float32),
                        U=res['U'].astype(np.float32),
                        max_dev=max_dev, mean_dev=mean_dev, cmd_rms=cmd_rms,
                        **{k: res[k] for k in
                           ('final_pos_err', 'final_vel', 'min_dist_target',
                            'koz_margin', 'koz_ok', 'peak_speed', 'vcap_viol',
                            'fuel', 'effort', 'settling_time', 'converged',
                            'divergent', 'filt_intervene_rate', 'filt_n_fallback',
                            'filt_corr_mean', 'filt_corr_max',
                            'filt_solve_us_mean', 'filt_solve_us_max')})
    print(f"    saved -> {out}")
    e2_summary = dict(converged=res['converged'], koz_ok=res['koz_ok'],
                      max_dev=max_dev, mean_dev=mean_dev, cmd_rms=cmd_rms,
                      fuel=res['fuel'], settling=res['settling_time'],
                      divergent=res['divergent'],
                      filt_intervene_rate=res['filt_intervene_rate'],
                      filt_n_fallback=res['filt_n_fallback'],
                      filt_solve_us_mean=res['filt_solve_us_mean'],
                      filt_solve_us_max=res['filt_solve_us_max'])
    return e2_summary


# ============================================================================
# E3, the paired thousand, network vs the cached expert references
# ============================================================================
def run_e3(model, refs, x_mean, x_std, run_id, filt=None):
    n = int(refs['n_mc'])
    mc_x0 = refs['mc_x0']; mc_th = refs['mc_theta0']
    mc_a = refs['mc_a']; mc_e = refs['mc_e']

    keys = ['converged', 'koz_ok', 'min_dist_target', 'koz_margin',
            'vcap_viol', 'fuel', 'effort', 'settling_time', 'divergent',
            'final_pos_err', 'n_steps']
    net = {k: np.zeros(n) for k in keys}
    # filter stats per run
    f_rate = np.zeros(n); f_fb = np.zeros(n)
    f_corr = np.zeros(n); f_solve_mean = np.zeros(n); f_solve_max = np.zeros(n)

    print(f"  E3: flying the {n} paired conditions"
          f"{' (filtered)' if filt is not None else ''}...")
    t0 = time.time()
    for i in range(n):
        res = fly_network(model, mc_x0[i], float(mc_th[i]),
                          float(mc_a[i]), float(mc_e[i]), x_mean, x_std,
                          filt=filt)
        for k in keys:
            net[k][i] = float(res[k])
        f_rate[i] = res['filt_intervene_rate']
        f_fb[i] = res['filt_n_fallback']
        f_corr[i] = res['filt_corr_max']
        f_solve_mean[i] = res['filt_solve_us_mean']
        f_solve_max[i] = res['filt_solve_us_max']
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0 + 1e-9)
            eta = (n - i - 1) / (rate + 1e-9) / 60.0
            print(f"    {i + 1}/{n}  rate {rate:.1f}/s  eta {eta:.1f} min")

    conv = net['converged'].astype(bool)
    koz = net['koz_ok'].astype(bool)
    div = net['divergent'].astype(bool)
    n_viol = int((~koz).sum())
    worst_pen = float(np.max(np.maximum(EXP.R_KOZ - net['min_dist_target'], 0.0)))

    # paired deltas, network minus expert on the identical condition
    fuel_d = net['fuel'] - refs['fom_fuel']
    settle_d = net['settling_time'] - refs['fom_settling_time']
    settle_d_valid = settle_d[~np.isnan(settle_d)]

    print("\n  " + "=" * 58)
    print("  E3 PAIRED SUMMARY (network vs expert, identical conditions)")
    print("  " + "=" * 58)
    print(f"  converged            : {conv.sum()}/{n} ({100*conv.mean():.1f}%)   "
          f"[expert: {int(refs['fom_converged'].sum())}/{n}]")
    print(f"  keep-out respected   : {koz.sum()}/{n}   "
          f"violations {n_viol}, worst penetration {worst_pen:.3f} m")
    print(f"  divergent runs       : {int(div.sum())}")
    if div.sum():
        print(f"    divergent indices  : {np.where(div)[0].tolist()}")
    if n_viol:
        print(f"    violation indices  : {np.where(~koz)[0].tolist()}")
    print(f"  keep-out margin [m]  : mean {net['koz_margin'].mean():.3f}   "
          f"min {net['koz_margin'].min():.3f}   "
          f"[expert mean {refs['fom_koz_margin'].mean():.3f}]")
    print(f"  vel-cap viol [m/s]   : mean {net['vcap_viol'].mean():.4f}   "
          f"max {net['vcap_viol'].max():.4f}")
    print(f"  final pos err [m]    : mean {net['final_pos_err'].mean():.4f}   "
          f"max {net['final_pos_err'].max():.4f}")
    print(f"  fuel delta [m/s]     : mean {fuel_d.mean():+.4f}   "
          f"max {fuel_d.max():+.4f}   (net minus expert)")
    if settle_d_valid.size:
        print(f"  settling delta [s]   : mean {settle_d_valid.mean():+.2f}   "
              f"max {settle_d_valid.max():+.2f}   "
              f"({settle_d_valid.size}/{n} runs with both settled)")
    if filt is not None:
        total_steps = int(net['n_steps'].sum())
        print("  " + "-" * 58)
        print(f"  filter intervention  : mean {100*f_rate.mean():.2f}%   "
              f"max {100*f_rate.max():.2f}%   per-flight")
        print(f"  fallback activations : {int(f_fb.sum())} total over "
              f"{total_steps} filtered steps")
        print(f"  filter correction max: {f_corr.max():.5f} m/s^2")
        print(f"  filter solve time    : mean-of-means {f_solve_mean.mean():.1f} us   "
              f"worst-case max {f_solve_max.max():.1f} us")
    print("  " + "=" * 58)

    out = f'cl_E3_{run_id}.npz'
    np.savez_compressed(out, **{f'net_{k}': v for k, v in net.items()},
                        fuel_delta=fuel_d, settling_delta=settle_d,
                        filt_rate=f_rate, filt_fallback=f_fb,
                        filt_corr_max=f_corr, filt_solve_mean=f_solve_mean,
                        filt_solve_max=f_solve_max)
    print(f"  saved -> {out}")
    e3_summary = dict(conv=int(conv.sum()), viol=n_viol,
                      worst_pen=worst_pen, divergent=int(div.sum()),
                      margin_min=float(net['koz_margin'].min()),
                      fuel_d_mean=float(fuel_d.mean()),
                      filt_rate_mean=float(f_rate.mean()),
                      filt_fallback_total=int(f_fb.sum()),
                      filt_solve_mean=float(f_solve_mean.mean()),
                      filt_solve_max=float(f_solve_max.max()))
    return e3_summary


def log_row(run_id, ckpt, cfg, e2, e3):
    header = ['run_id', 'ckpt', 'tag', 'arch',
              'e2_converged', 'e2_koz_ok', 'e2_divergent',
              'e2_max_dev', 'e2_mean_dev', 'e2_cmd_rms',
              'e3_conv', 'e3_viol', 'e3_worst_pen', 'e3_divergent',
              'e3_margin_min', 'e3_fuel_d_mean',
              'e3_filt_rate', 'e3_filt_fallback', 'e3_filt_solve_mean',
              'e3_filt_solve_max']
    row = [run_id, ckpt, cfg.get('tag', ''), cfg.get('arch', ''),
           e2.get('converged'), e2.get('koz_ok'), e2.get('divergent'),
           f"{e2.get('max_dev', np.nan):.4f}", f"{e2.get('mean_dev', np.nan):.4f}",
           f"{e2.get('cmd_rms', np.nan):.6f}",
           e3.get('conv'), e3.get('viol'), f"{e3.get('worst_pen', np.nan):.4f}",
           e3.get('divergent'), f"{e3.get('margin_min', np.nan):.4f}",
           f"{e3.get('fuel_d_mean', np.nan):+.4f}",
           f"{e3.get('filt_rate_mean', np.nan):.4f}",
           e3.get('filt_fallback_total', ''),
           f"{e3.get('filt_solve_mean', np.nan):.2f}",
           f"{e3.get('filt_solve_max', np.nan):.2f}"]
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


# ============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Closed-loop evaluation (E2/E3).")
    p.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
    p.add_argument('--latest', action='store_true',
                   help='use the most recent checkpoint')
    p.add_argument('--mode', choices=['e2', 'e3', 'both'], default='both')
    # CBF safety filter options
    p.add_argument('--filter', action='store_true',
                   help='apply the CBF-CLF-QP safety filter to every command')
    p.add_argument('--alpha1', type=float, default=cbf_filter.ALPHA1_DEFAULT,
                   help='keep-out inner gain (certified <= 0.325)')
    p.add_argument('--alpha2', type=float, default=cbf_filter.ALPHA2_DEFAULT,
                   help='keep-out outer gain (free, tuned in 9.4.2)')
    p.add_argument('--alphav', type=float, default=cbf_filter.ALPHAV_DEFAULT,
                   help='velocity-cap gain (free, tuned in 9.4.2)')
    p.add_argument('--rho', type=float, default=cbf_filter.RHO_DEFAULT,
                   help='CLF slack penalty (tuned in 9.4.2)')
    args = p.parse_args()

    if args.latest:
        ckpt_path = latest_checkpoint()
    elif args.ckpt:
        ckpt_path = args.ckpt
    else:
        p.error("give --ckpt PATH or --latest")

    if not os.path.exists(REFS_FILE):
        sys.exit(f"{REFS_FILE} not found - run cache_expert_references.py first")

    print("=" * 66)
    print("CLOSED-LOOP EVALUATION (network in the expert's place)")
    print("=" * 66)
    model, cfg, ck = load_checkpoint(ckpt_path)
    run_id = ck.get('run_id', os.path.basename(ckpt_path).replace('.pt', ''))
    print(f"  checkpoint : {ckpt_path}")
    print(f"  run id     : {run_id}")
    print(f"  config     : arch={cfg['arch']} depth={cfg['depth']} "
          f"width={cfg['width']} act={cfg['activation']} head={cfg['head']} "
          f"tag='{cfg.get('tag','')}'")
    print(f"  best val   : {ck.get('best_val', float('nan')):.6e}")

    # build the safety filter if requested; tag the run id so filtered outputs
    # do not overwrite the unfiltered ones
    filt = None
    if args.filter:
        filt = cbf_filter.CBFFilter(alpha1=args.alpha1, alpha2=args.alpha2,
                                    alpha_v=args.alphav, rho=args.rho)
        run_id = (f"{run_id}_filt_a2-{args.alpha2}_av-{args.alphav}"
                  f"_rho-{args.rho}")
        print(f"  FILTER     : on   alpha1={args.alpha1} alpha2={args.alpha2} "
              f"alpha_v={args.alphav} rho={args.rho}")
        print(f"  run id     : {run_id}")

    refs = np.load(REFS_FILE)
    x_mean, x_std = bc_data.load_norm_stats()
    print(f"  references : {REFS_FILE} (E2 worst case + {int(refs['n_mc'])} paired)")
    print("=" * 66)

    e2_summary, e3_summary = {}, {}
    if args.mode in ('e2', 'both'):
        e2_summary = run_e2(model, refs, x_mean, x_std, run_id, filt=filt)
    if args.mode in ('e3', 'both'):
        e3_summary = run_e3(model, refs, x_mean, x_std, run_id, filt=filt)

    log_row(run_id, ckpt_path, cfg, e2_summary, e3_summary)
    print(f"\n  summary row appended -> {LOG_FILE}")