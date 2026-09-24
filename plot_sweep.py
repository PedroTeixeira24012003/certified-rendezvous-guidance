"""
plot_sweep.py
Turns the logged sweep runs into one val-loss-versus-factor figure per
hyperparameter, each with the anchor marked as the shared reference point.
Reads experiments_log.csv and writes sweep_<name>.png per sweep, plus a
printed summary of every run's best validation loss.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

LOG_FILE = 'experiments_log.csv'

# which config field each sweep varies, and whether the x-axis is log-scaled
SWEEP_FIELD = {
    'depth':      ('depth', False),
    'width':      ('width', False),
    'lr':         ('lr', True),
    'batch':      ('batch', True),
    'optimizer':  ('optimizer', False),
    'loss':       ('loss', False),
    'activation': ('activation', False),
    'head':       ('head', False),
}

# the anchor's value on each swept field, so it can be plotted as the centre
ANCHOR_VALUE = {
    'depth': 3, 'width': 256, 'lr': 3e-4, 'batch': 4096,
    'optimizer': 'adam', 'loss': 'mse', 'activation': 'relu', 'head': 'tanh',
}

# axis labels
NICE = {
    'depth': 'Hidden Layers', 'width': 'Units Per Layer',
    'lr': 'Learning Rate', 'batch': 'Batch Size',
    'optimizer': 'Optimizer', 'loss': 'Loss Function',
    'activation': 'Activation', 'head': 'Output Head',
}


def load_runs():
    """Read every logged run into a list of dicts with parsed config + best_val."""
    if not os.path.exists(LOG_FILE):
        sys.exit(f"no {LOG_FILE} found - run the sweeps first")
    runs = []
    with open(LOG_FILE) as f:
        for row in csv.DictReader(f):
            cfg = json.loads(row['config_json'])
            runs.append({'cfg': cfg, 'best_val': float(row['best_val']),
                         'n_params': int(row['n_params'])})
    return runs


def anchor_run(runs):
    """Find the run that matches the anchor on every swept field. If several
    match, return the best of them by validation loss, so a stale partial run
    cannot masquerade as the anchor."""
    matches = [r for r in runs
               if all(r['cfg'].get(f) == ANCHOR_VALUE[f] for f in ANCHOR_VALUE)]
    if not matches:
        return None
    return min(matches, key=lambda r: r['best_val'])


def runs_for_sweep(runs, sweep):
    """Collect the runs that vary only `sweep` off the anchor (plus the anchor
    itself), so the plot shows one factor changing and nothing else."""
    field, _ = SWEEP_FIELD[sweep]
    others = [f for f in ANCHOR_VALUE if f != field]
    pts = []
    for r in runs:
        c = r['cfg']
        # every non-swept field must equal the anchor
        if all(c.get(o) == ANCHOR_VALUE[o] for o in others):
            pts.append((c.get(field), r['best_val']))
    # dedupe by x (keep the best if repeated), sort for a clean line
    best = {}
    for x, v in pts:
        if x not in best or v < best[x]:
            best[x] = v
    xs = sorted(best, key=lambda k: (isinstance(k, str), k))
    return xs, [best[x] for x in xs]


def plot_sweep(sweep, runs):
    field, logx = SWEEP_FIELD[sweep]
    xs, vals = runs_for_sweep(runs, sweep)
    if len(xs) < 2:
        print(f"  [skip] {sweep}: only {len(xs)} point(s) logged")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    numeric = all(not isinstance(x, str) for x in xs)

    if numeric:
        ax.plot(xs, vals, 'o-', color='tab:blue', lw=1.5, ms=7)
        if logx:
            ax.set_xscale('log')
        # mark the anchor point
        av = ANCHOR_VALUE[field]
        if av in xs:
            ai = xs.index(av)
            ax.plot(av, vals[ai], 'o', color='tab:red', ms=11, mfc='none',
                    mew=2, label='Anchor')
    else:
        idx = np.arange(len(xs))
        colors = ['tab:red' if x == ANCHOR_VALUE[field] else 'tab:blue' for x in xs]
        ax.bar(idx, vals, color=colors, width=0.55)
        ax.set_xticks(idx)
        ax.set_xticklabels([str(x).capitalize() for x in xs])
        ax.plot([], [], 's', color='tab:red', label='Anchor')

    ax.set_xlabel(NICE[sweep])
    ax.set_ylabel('Best Validation Loss')
    ax.set_title(f'Validation Loss Versus {NICE[sweep]}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = f'sweep_{sweep}.png'
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  saved {out}   ({NICE[sweep]}: "
          + ", ".join(f'{x}={v:.3e}' for x, v in zip(xs, vals)) + ")")
    return out


# ============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot the OFAT sweeps (7.2).")
    p.add_argument('--only', type=str, default=None,
                   help=f'one sweep only ({", ".join(SWEEP_FIELD)})')
    args = p.parse_args()

    runs = load_runs()
    print("=" * 62)
    print(f"  loaded {len(runs)} runs from {LOG_FILE}")
    a = anchor_run(runs)
    if a:
        print(f"  anchor best val: {a['best_val']:.6e}  ({a['n_params']:,} params)")
    print("=" * 62)

    sweeps = [args.only] if args.only else list(SWEEP_FIELD)
    made = []
    for s in sweeps:
        if s not in SWEEP_FIELD:
            print(f"  unknown sweep '{s}'"); continue
        out = plot_sweep(s, runs)
        if out:
            made.append(out)

    print("=" * 62)
    print(f"  made {len(made)} figures: {', '.join(made)}")
    print("=" * 62)