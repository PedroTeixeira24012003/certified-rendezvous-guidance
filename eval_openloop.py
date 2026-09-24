"""
eval_openloop.py
Open-loop evaluation of a trained checkpoint on the test trajectories only.
Reports per-axis RMSE and maximum absolute error in physical m/s^2, split
into transit and holding phases by distance to the hold point, because the
near-zero holding commands dominate the data and a pooled error would hide
the transit accuracy. Appends one summary row per checkpoint to
openloop_log.csv.
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np
import torch

import bc_data
import train_policy

U_MAX = bc_data.U_MAX
HOLD = np.array([-8.0, 0.0, 0.0])       # hold point (matches the expert module)
TRANSIT_THRESH = 15.0                    # m, distance to hold splitting the phases
LOG_FILE = 'openloop_log.csv'
AXES = ['x', 'y', 'z']


def load_checkpoint(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = train_policy.build_model(ck['config'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck['config'], ck


def latest_checkpoint():
    cks = sorted(glob.glob('checkpoints/*.pt'), key=os.path.getmtime)
    if not cks:
        sys.exit("no checkpoints found in checkpoints/")
    return cks[-1]


def evaluate(model):
    """Run the network on the whole test set, return raw per-pair errors in
    m/s^2 plus the distance-to-hold of each pair for the phase split."""
    ds = bc_data.BCDataset('test')          # normalized inputs, raw command
                                            # and raw position also kept
    Xn = ds.X                               # normalized inputs (torch)
    Y_raw = ds.Y_raw.numpy()               # expert command, raw m/s^2
    pos = ds.pos.numpy()                   # raw position, m

    # forward pass in batches, physical output (m/s^2)
    preds = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], 8192):
            u = model(Xn[i:i + 8192], scaled=False)   # m/s^2, in the box
            preds.append(u.numpy())
    U_pred = np.concatenate(preds, axis=0)             # (N, 3) m/s^2

    err = U_pred - Y_raw                                # signed error, m/s^2
    dist_to_hold = np.linalg.norm(pos - HOLD, axis=1)   # m
    return err, dist_to_hold


def phase_stats(err, mask):
    """Per-axis RMSE and max |error| over the pairs selected by mask."""
    e = err[mask]
    if e.shape[0] == 0:
        return None
    rmse = np.sqrt(np.mean(e ** 2, axis=0))        # (3,)
    maxe = np.max(np.abs(e), axis=0)               # (3,)
    return rmse, maxe, e.shape[0]


def print_table(err, dist):
    transit = dist > TRANSIT_THRESH
    holding = ~transit
    n = err.shape[0]

    print("  " + "=" * 62)
    print(f"  E1 OPEN-LOOP ERROR ON TEST PAIRS  (N = {n:,})")
    print(f"  phase split at distance-to-hold = {TRANSIT_THRESH:.0f} m")
    print("  " + "=" * 62)
    print(f"  transit pairs (> {TRANSIT_THRESH:.0f} m): {transit.sum():,} "
          f"({100*transit.mean():.1f}%)")
    print(f"  holding pairs (<= {TRANSIT_THRESH:.0f} m): {holding.sum():,} "
          f"({100*holding.mean():.1f}%)")
    print("  " + "-" * 62)

    results = {}
    for name, mask in [('Transit', transit), ('Holding', holding),
                       ('Pooled', np.ones(n, dtype=bool))]:
        st = phase_stats(err, mask)
        if st is None:
            continue
        rmse, maxe, cnt = st
        results[name] = (rmse, maxe)
        print(f"  {name} phase ({cnt:,} pairs):")
        print(f"    per-axis RMSE   [m/s^2]: "
              f"x {rmse[0]:.5e}   y {rmse[1]:.5e}   z {rmse[2]:.5e}")
        print(f"    per-axis max|e| [m/s^2]: "
              f"x {maxe[0]:.5e}   y {maxe[1]:.5e}   z {maxe[2]:.5e}")
        # express the transit RMSE as a fraction of the thrust limit
        if name == 'Transit':
            frac = rmse / U_MAX * 100
            print(f"    transit RMSE as % of u_max: "
                  f"x {frac[0]:.2f}%   y {frac[1]:.2f}%   z {frac[2]:.2f}%")
        print()
    print("  " + "=" * 62)
    return results


def log_row(run_id, cfg, results):
    tr_rmse = results['Transit'][0]
    ho_rmse = results['Holding'][0]
    po_rmse = results['Pooled'][0]
    header = ['run_id', 'tag', 'arch',
              'transit_rmse_x', 'transit_rmse_y', 'transit_rmse_z',
              'transit_max_worst',
              'holding_rmse_x', 'holding_rmse_y', 'holding_rmse_z',
              'pooled_rmse_x', 'pooled_rmse_y', 'pooled_rmse_z']
    row = [run_id, cfg.get('tag', ''), cfg.get('arch', ''),
           f'{tr_rmse[0]:.6e}', f'{tr_rmse[1]:.6e}', f'{tr_rmse[2]:.6e}',
           f'{results["Transit"][1].max():.6e}',
           f'{ho_rmse[0]:.6e}', f'{ho_rmse[1]:.6e}', f'{ho_rmse[2]:.6e}',
           f'{po_rmse[0]:.6e}', f'{po_rmse[1]:.6e}', f'{po_rmse[2]:.6e}']
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


# ============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Open-loop evaluation (E1).")
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--latest', action='store_true')
    args = p.parse_args()

    if args.latest:
        ckpt_path = latest_checkpoint()
    elif args.ckpt:
        ckpt_path = args.ckpt
    else:
        p.error("give --ckpt PATH or --latest")

    print("=" * 66)
    print("OPEN-LOOP EVALUATION (E1) ON TEST TRAJECTORIES")
    print("=" * 66)
    model, cfg, ck = load_checkpoint(ckpt_path)
    run_id = ck.get('run_id', os.path.basename(ckpt_path).replace('.pt', ''))
    print(f"  checkpoint : {ckpt_path}")
    print(f"  run id     : {run_id}")
    print(f"  config     : arch={cfg['arch']} depth={cfg['depth']} "
          f"width={cfg['width']} act={cfg['activation']} head={cfg['head']} "
          f"tag='{cfg.get('tag','')}'")
    print(f"  best val   : {ck.get('best_val', float('nan')):.6e}")
    print("=" * 66)

    err, dist = evaluate(model)
    results = print_table(err, dist)
    log_row(run_id, cfg, results)
    print(f"  summary row appended -> {LOG_FILE}")