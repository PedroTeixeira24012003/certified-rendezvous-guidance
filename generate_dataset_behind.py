"""
generate_dataset_behind.py
Behavioural-cloning dataset generator. Flies the validated cost-shaped
expert from random initial conditions and records, at every closed-loop
step, the state features and the command the expert applied. Checkpoints
incrementally and resumes after a crash.
"""

import os
import sys
import time
import numpy as np

# import the validated expert, dynamics, solver, sampler and constants
import th_mpc_behind_1000run as EXP

# ============================================================================
# Configuration
# ============================================================================
N_TRAJ          = 10000        # trajectories to generate
SEED            = 12345        # RNG seed
DATASET_FILE    = 'th_bc_dataset_behind.npz'
CHECKPOINT_FILE = 'th_bc_dataset_behind_checkpoint.npz'
CHECKPOINT_EVERY = 100         # checkpoint every this many trajectories
PROGRESS_EVERY   = 20          # progress print every this many trajectories

# acceptance filters, only clean expert demonstrations enter the dataset
REQUIRE_CONVERGED   = True     # must reach the hold tolerance
REQUIRE_KEEPOUT     = True     # must never enter the keep-out zone
MAX_SOLVE_FAILS     = 5        # reject beyond this many failed solves


# ============================================================================
# Feature vector from a raw 7-element state plus the orbit.
# Returns [x, y, z, vx, vy, vz, sin(theta), cos(theta), a, e].
# ============================================================================
def make_features(state7, a, e):
    x, y, z, vx, vy, vz, th = state7
    return np.array([x, y, z, vx, vy, vz,
                     np.sin(th), np.cos(th), a, e], dtype=np.float64)


# ============================================================================
# Fly the expert once and return the (inputs, outputs) pairs for every step,
# or None with a reject tag when the acceptance filters fail the trajectory.
# ============================================================================
def trajectory_pairs(solver, x0_rel, theta0, a, e):
    res = EXP.run_closed_loop(
        solver, x0_rel, theta0, a, e,
        sim_steps=EXP.SIM_STEPS,
        stop_on_arrival=False,     # fly to full settling
        verbose=False,
    )

    # acceptance filters
    if REQUIRE_CONVERGED and not res['converged']:
        return None, res, 'not_converged'
    if REQUIRE_KEEPOUT and not res['koz_ok']:
        return None, res, 'keepout_violated'
    if res['solve_fail'] > MAX_SOLVE_FAILS:
        return None, res, 'too_many_solve_fails'

    X = res['X']            # (n_steps+1, 7)
    U = res['U']            # (n_steps,   3)
    n = U.shape[0]

    # one feature row for each state that has a following command
    inputs = np.empty((n, 10), dtype=np.float64)
    for k in range(n):
        inputs[k] = make_features(X[k], a, e)
    outputs = U.astype(np.float64)            # (n, 3)

    # guard against non-finite values
    if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(outputs))):
        return None, res, 'non_finite'

    return (inputs, outputs), res, 'ok'


# ============================================================================
# Save the accumulated dataset and metadata, float32, atomic write
# ============================================================================
def save_dataset(path, inputs_list, outputs_list, meta):
    if inputs_list:
        X = np.concatenate(inputs_list, axis=0).astype(np.float32)
        Y = np.concatenate(outputs_list, axis=0).astype(np.float32)
    else:
        X = np.zeros((0, 10), dtype=np.float32)
        Y = np.zeros((0, 3),  dtype=np.float32)

    tmp = path + '.tmp'
    np.savez_compressed(
        tmp,
        X=X, Y=Y,
        n_traj_done   = meta['n_traj_done'],
        n_traj_target = meta['n_traj_target'],
        n_rejected    = meta['n_rejected'],
        reject_reasons= np.array(meta['reject_reasons']),
        feature_names = np.array(
            ['x','y','z','vx','vy','vz','sin_theta','cos_theta','a','e']),
        output_names  = np.array(['ux','uy','uz']),
        seed          = SEED,
    )
    os.replace(tmp + '.npz', path)    # atomic on the same filesystem
    return X.shape[0]


# ============================================================================
# Per-feature normalization statistics, stored for use at deployment.
# std is floored so a constant feature cannot divide by zero.
# ============================================================================
def save_norm_stats(path, X, Y):
    eps = 1e-8
    x_mean = X.mean(axis=0);  x_std = X.std(axis=0)
    y_mean = Y.mean(axis=0);  y_std = Y.std(axis=0)
    x_std = np.where(x_std < eps, 1.0, x_std)
    y_std = np.where(y_std < eps, 1.0, y_std)
    np.savez(path,
             x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32),
             y_mean=y_mean.astype(np.float32), y_std=y_std.astype(np.float32),
             feature_names=np.array(
                 ['x','y','z','vx','vy','vz','sin_theta','cos_theta','a','e']),
             output_names=np.array(['ux','uy','uz']))


# ============================================================================
# Resume from a checkpoint if one exists
# ============================================================================
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return [], [], 0, 0, []
    try:
        d = np.load(CHECKPOINT_FILE, allow_pickle=True)
        X = d['X']; Y = d['Y']
        n_done = int(d['n_traj_done'])
        n_rej  = int(d['n_rejected'])
        reasons = list(d['reject_reasons'])
        print(f"  resuming from checkpoint: {n_done} trajectories, "
              f"{X.shape[0]} pairs already collected")
        # single-element lists so appending continues to work
        inputs_list = [X.astype(np.float64)] if X.shape[0] else []
        outputs_list = [Y.astype(np.float64)] if Y.shape[0] else []
        return inputs_list, outputs_list, n_done, n_rej, reasons
    except Exception as ex:
        print(f"  checkpoint unreadable ({ex}); starting fresh")
        return [], [], 0, 0, []


# ============================================================================
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("BEHAVIOURAL-CLONING DATASET GENERATION (behind, cost-shaped expert)")
    print(f"  target trajectories : {N_TRAJ}")
    print(f"  input features (10) : x y z vx vy vz sin(theta) cos(theta) a e")
    print(f"  output labels  (3)  : ux uy uz")
    print(f"  checkpoint every    : {CHECKPOINT_EVERY} trajectories -> "
          f"{CHECKPOINT_FILE}")
    print(f"  final dataset       : {DATASET_FILE}")
    print("=" * 70)

    # frame check before a long run
    EXP.verify_frame()
    print("=" * 70)

    solver = EXP.build_solver()

    rng = np.random.default_rng(SEED)

    inputs_list, outputs_list, n_done, n_rejected, reject_reasons = load_checkpoint()

    # on resume, advance the RNG past the trajectories already consumed by
    # re-drawing and discarding, so the stream of initial conditions is
    # identical to an uninterrupted run
    for _ in range(n_done):
        EXP.sample_cylinder_ic(rng)
        rng.uniform(0, 2 * np.pi)          # theta0
        rng.uniform(*EXP.A_RANGE)          # a
        rng.uniform(*EXP.E_RANGE)          # e

    n_pairs = sum(a.shape[0] for a in inputs_list)

    while n_done < N_TRAJ:
        # draw one initial condition from the operating set
        x0 = EXP.sample_cylinder_ic(rng)
        theta0 = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(*EXP.A_RANGE)
        e = rng.uniform(*EXP.E_RANGE)

        try:
            pair, res, tag = trajectory_pairs(solver, x0[:6], theta0, a, e)
        except Exception as ex:
            # one trajectory blew up, log and skip
            n_done += 1
            n_rejected += 1
            reject_reasons.append(f'exception:{type(ex).__name__}')
            print(f"  [traj {n_done}] EXCEPTION {type(ex).__name__}: {ex} -> skipped")
            continue

        n_done += 1
        if pair is None:
            n_rejected += 1
            reject_reasons.append(tag)
        else:
            inputs_list.append(pair[0])
            outputs_list.append(pair[1])
            n_pairs += pair[0].shape[0]

        if n_done % PROGRESS_EVERY == 0:
            rate = n_done / (time.time() - t_start + 1e-9)
            eta_min = (N_TRAJ - n_done) / (rate + 1e-9) / 60.0
            print(f"  {n_done}/{N_TRAJ}  pairs={n_pairs}  "
                  f"rejected={n_rejected}  "
                  f"rate={rate:.1f} traj/s  eta={eta_min:.0f} min")

        if n_done % CHECKPOINT_EVERY == 0:
            meta = dict(n_traj_done=n_done, n_traj_target=N_TRAJ,
                        n_rejected=n_rejected, reject_reasons=reject_reasons)
            save_dataset(CHECKPOINT_FILE, inputs_list, outputs_list, meta)

    # final save
    meta = dict(n_traj_done=n_done, n_traj_target=N_TRAJ,
                n_rejected=n_rejected, reject_reasons=reject_reasons)
    n_final = save_dataset(DATASET_FILE, inputs_list, outputs_list, meta)

    # normalization statistics from the final dataset
    d = np.load(DATASET_FILE)
    save_norm_stats('bc_norm_stats_behind.npz', d['X'], d['Y'])

    # report
    dt_min = (time.time() - t_start) / 60.0
    print("\n" + "=" * 70)
    print("  DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"  trajectories attempted : {n_done}")
    print(f"  trajectories accepted  : {n_done - n_rejected}")
    print(f"  trajectories rejected  : {n_rejected}")
    if n_rejected:
        # tally reject reasons
        reasons, counts = np.unique(np.array(reject_reasons), return_counts=True)
        for r, c in zip(reasons, counts):
            print(f"      {r:24s}: {c}")
    print(f"  total (state,command) pairs : {n_final}")
    print(f"  dataset file  : {DATASET_FILE}")
    print(f"  norm stats    : bc_norm_stats_behind.npz")
    print(f"  wall-clock    : {dt_min:.1f} min")
    print("=" * 70)

    # remove the checkpoint now that the final dataset is written
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print("  checkpoint removed (final dataset written)")
        except OSError:
            pass