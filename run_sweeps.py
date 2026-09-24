"""
run_sweeps.py
One-factor-at-a-time hyperparameter campaign. Every run is the anchor
configuration with exactly one field changed, trained on the fixed 25%
subset with the same seed and stopping rules, one row per run in
experiments_log.csv. Already-logged configs are detected and skipped, so
an interrupted campaign resumes by re-running. Supports --dry-run and
--only <sweep>.
"""

import os
import sys
import json
import argparse
import subprocess

import train_policy   # for the ANCHOR dict and to detect already-run configs

PYTHON = sys.executable          # same interpreter running this script
LOG_FILE = 'experiments_log.csv'

# sweep grid, name -> list of (field, value) overrides off the anchor,
# each entry one run; the anchor is added automatically as the shared centre
SWEEPS = {
    'depth':      [('depth', 2), ('depth', 4)],
    'width':      [('width', 128), ('width', 512)],
    'lr':         [('lr', 1e-3), ('lr', 3e-3)],
    'batch':      [('batch', 1024), ('batch', 16384)],
    'optimizer':  [('optimizer', 'adamw'), ('optimizer', 'sgd')],
    'loss':       [('loss', 'huber'), ('loss', 'l1')],
    'activation': [('activation', 'tanh'), ('activation', 'silu')],
    'head':       [('head', 'linear')],
}


def anchor_config():
    """The anchor, forced onto the 25% subset with a fixed seed and a tag."""
    cfg = dict(train_policy.ANCHOR)
    cfg['subset'] = True
    cfg['seed'] = 0
    cfg['tag'] = 'anchor'
    return cfg


def build_run_list():
    """Return a list of (label, config) for the anchor plus every sweep point."""
    runs = [('anchor', anchor_config())]
    for sweep_name, points in SWEEPS.items():
        for field, value in points:
            cfg = anchor_config()
            cfg[field] = value
            cfg['tag'] = f'{sweep_name}={value}'
            runs.append((cfg['tag'], cfg))
    return runs


def already_run(cfg):
    """True if a row with this exact config (ignoring the tag) is already
    logged, so an interrupted campaign resumes without repeating runs."""
    if not os.path.exists(LOG_FILE):
        return False
    import csv
    key = {k: cfg[k] for k in cfg if k != 'tag'}
    key_json = json.dumps(key, sort_keys=True)
    with open(LOG_FILE) as f:
        for row in csv.DictReader(f):
            logged = json.loads(row['config_json'])
            logged.pop('tag', None)
            if json.dumps(logged, sort_keys=True) == key_json:
                return True
    return False


def run_one(cfg):
    """Invoke train_policy.py as a subprocess with this config, via a temp JSON."""
    tmp = '_sweep_cfg.json'
    with open(tmp, 'w') as f:
        json.dump(cfg, f)
    # -u for live output; train_policy reads the JSON, no CLI overrides needed
    cmd = [PYTHON, '-u', 'train_policy.py', '--config', tmp]
    print(f"\n>>> RUN: {cfg['tag']}")
    print(f"    {cmd}")
    subprocess.run(cmd, check=True)
    os.remove(tmp)


# ============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OFAT hyperparameter campaign (7.2).")
    p.add_argument('--dry-run', action='store_true',
                   help='list the runs and exit without training')
    p.add_argument('--only', type=str, default=None,
                   help=f'run one sweep only ({", ".join(SWEEPS)})')
    args = p.parse_args()

    runs = build_run_list()
    if args.only:
        if args.only not in SWEEPS and args.only != 'anchor':
            p.error(f"--only must be one of: anchor, {', '.join(SWEEPS)}")
        runs = [(lbl, c) for lbl, c in runs
                if c['tag'] == 'anchor' and args.only == 'anchor'
                or c['tag'].startswith(f'{args.only}=')]

    print("=" * 62)
    print(f"OFAT CAMPAIGN: {len(runs)} runs (anchor + sweep points)")
    print("=" * 62)
    for lbl, cfg in runs:
        done = already_run(cfg)
        print(f"  [{'DONE' if done else '   '}] {lbl}")
    print("=" * 62)

    if args.dry_run:
        print("  dry run: nothing trained.")
        sys.exit(0)

    n_run, n_skip = 0, 0
    for lbl, cfg in runs:
        if already_run(cfg):
            print(f"\n>>> SKIP (already logged): {lbl}")
            n_skip += 1
            continue
        run_one(cfg)
        n_run += 1

    print("\n" + "=" * 62)
    print(f"  CAMPAIGN COMPLETE: {n_run} trained, {n_skip} skipped (already done)")
    print(f"  results in {LOG_FILE} -> run plot_sweep.py to make the figures")
    print("=" * 62)