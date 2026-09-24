"""
make_tasil_targets.py
Precompute the TaSIL probe targets for the correction-law term. For each dataset row one small probe is drawn in
normalized input space on the six physical state features only, the expert
is queried at the state and at the probed state through
expert_iface.expert_command, and the scaled response difference is stored
aligned with the rows. One probe per row, fixed seed, reproducible targets.
"""

import argparse
import time
import numpy as np
import bc_data

EPS_PROBE = 0.05          # probe radius in normalized units
PROBE_SEED = 21           # fixed seed for the probe draw
OUT_FULL = 'tasil_targets_behind.npz'
OUT_SUB = 'tasil_targets_behind_subset.npz'

try:
    from expert_iface import expert_command
except ImportError:
    expert_command = None


def draw_probes(n_rows, rng):
    """Uniform on the 6-sphere of radius EPS_PROBE, one per row."""
    d = rng.normal(size=(n_rows, 6))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return (EPS_PROBE * d).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', action='store_true',
                    help='only the 2000-trajectory sweep subset (800k rows)')
    ap.add_argument('--start', type=int, default=0,
                    help='resume from this entry index')
    args = ap.parse_args()

    if expert_command is None:
        raise SystemExit(
            "expert_iface.py not found. Create it with a function\n"
            "  expert_command(r, v, theta, a, e) -> np.array (3,)\n"
            "wrapping your th_mpc expert (see the docstring).")

    print("=" * 62)
    print("PRECOMPUTE TaSIL PROBE TARGETS (four-term loss, Ch 21)")
    print("=" * 62)

    if args.subset:
        traj = bc_data.make_subset()
        out_file = OUT_SUB
    else:
        traj = None
        out_file = OUT_FULL

    if traj is not None:
        rows = bc_data.traj_to_rows(traj)
    else:
        data0 = np.load(bc_data.DATASET_FILE, mmap_mode='r')
        rows = np.arange(data0['X'].shape[0], dtype=np.int64)

    n = rows.shape[0]
    print(f"  rows to process: {n:,}  ({'subset' if args.subset else 'FULL'})")

    data = np.load(bc_data.DATASET_FILE, mmap_mode='r')
    X = data['X']                                       # raw features, mmap
    x_mean, x_std = bc_data.load_norm_stats()
    std6 = x_std[:6].astype(np.float64)

    rng = np.random.default_rng(PROBE_SEED)
    delta_norm = draw_probes(n, rng)                    # (n, 6) normalized
    d_exp = np.zeros((n, 3), dtype=np.float32)

    t0 = time.time()
    for j in range(args.start, n):
        i = rows[j]
        xi = np.asarray(X[i], dtype=np.float64)
        r, v = xi[0:3], xi[3:6]
        th = np.arctan2(xi[6], xi[7])
        a, e = xi[8], xi[9]

        dphys = delta_norm[j].astype(np.float64) * std6
        rp, vp = r + dphys[0:3], v + dphys[3:6]

        u_base = np.asarray(expert_command(r, v, th, a, e))
        u_prob = np.asarray(expert_command(rp, vp, th, a, e))
        d_exp[j] = ((u_prob - u_base) / bc_data.U_MAX).astype(np.float32)

        if (j + 1) % 10000 == 0:
            rate = (j + 1 - args.start) / (time.time() - t0)
            eta = (n - j - 1) / max(rate, 1e-9) / 3600.0
            print(f"    {j+1:>9,}/{n:,}  {rate:7.1f} rows/s  ETA {eta:5.1f} h")
            np.savez(out_file + '.partial', delta_norm=delta_norm,
                     d_exp=d_exp, rows=rows, done=j + 1)

    np.savez(out_file, delta_norm=delta_norm, d_exp=d_exp, rows=rows)
    mag = np.linalg.norm(d_exp, axis=1)
    print(f"  |d_exp| (scaled) range [{mag.min():.3e}, {mag.max():.3e}] "
          f"mean {mag.mean():.3e}")
    print(f"  saved -> {out_file}")
    print("  DONE")


if __name__ == "__main__":
    main()