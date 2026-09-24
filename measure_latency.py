"""
measure_latency.py
Batch-1 forward-pass latency and parameter counts for the three policy
architectures. Each model is timed on a single sample in the exact shape it
consumes in flight, warmed first, single-threaded, and both the mean and
the worst case are reported, because a guidance evaluation must fit inside
the control interval every cycle, not on average. Writes latency_log.csv.
"""

import os
import csv
import glob
import time
import argparse
import numpy as np
import torch

import train_policy
import train_history
import models

LOG = 'latency_log.csv'

# The three checkpoints. Edit here if run ids differ; --ckpt-dir scans.
DEFAULT_CKPTS = {
    'MLP (champion)':      '20260723-030815-8f9a49',
    'Encoder-Decoder':     '20260723-041105-ce0b73',
    'History Transformer': '20260723-051201-de35ab',
}


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    if cfg.get('arch') == 'history':
        model = train_history.build_history_model(cfg)
    else:
        model = train_policy.build_model(cfg)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, cfg, ck


def make_input(model):
    """One batch-1 input in the exact shape the model consumes in flight."""
    if isinstance(model, models.HistoryPolicy):
        k = model.k
        x_hist = torch.randn(1, k, 13, dtype=torch.float32)
        x_now = torch.randn(1, 10, dtype=torch.float32)
        return ('history', x_hist, x_now)
    return ('plain', torch.randn(1, 10, dtype=torch.float32), None)


def time_model(model, reps, warmup):
    torch.set_num_threads(1)                     # single-cycle cost, not parallel
    kind, a, b = make_input(model)

    with torch.no_grad():
        for _ in range(warmup):
            model(a, b, scaled=False) if kind == 'history' else model(a, scaled=False)

        samples = np.empty(reps)
        for i in range(reps):
            t0 = time.perf_counter()
            model(a, b, scaled=False) if kind == 'history' else model(a, scaled=False)
            samples[i] = (time.perf_counter() - t0) * 1e6   # microseconds

    return samples


def main():
    p = argparse.ArgumentParser(description="Batch-1 latency of the bracket models.")
    p.add_argument('--reps', type=int, default=400)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--ckpt-dir', type=str, default='checkpoints')
    args = p.parse_args()

    print("=" * 70)
    print("BATCH-1 FORWARD LATENCY  (Section 7.3.3 / Chapter 12)")
    print("=" * 70)
    print(f"  reps {args.reps}, warmup {args.warmup}, single thread\n")

    rows = []
    header = f"  {'Architecture':<22}{'Params':>10}{'Mean [us]':>12}" \
             f"{'Max [us]':>11}{'Std [us]':>11}"
    print(header)
    print("  " + "-" * 64)

    for name, run_id in DEFAULT_CKPTS.items():
        path = os.path.join(args.ckpt_dir, f'{run_id}.pt')
        if not os.path.exists(path):
            print(f"  {name:<22}  checkpoint not found: {path}")
            continue
        model, cfg, ck = load_model(path)
        n_params = models.count_params(model)
        s = time_model(model, args.reps, args.warmup)
        mean, mx, sd = float(s.mean()), float(s.max()), float(s.std())
        print(f"  {name:<22}{n_params:>10,}{mean:>12.2f}{mx:>11.2f}{sd:>11.2f}")
        rows.append(dict(architecture=name, run_id=run_id, params=n_params,
                         mean_us=mean, max_us=mx, std_us=sd,
                         reps=args.reps))

    print("  " + "-" * 64)
    if rows:
        base = rows[0]['mean_us']
        print("\n  relative to the MLP champion (mean):")
        for r in rows:
            print(f"    {r['architecture']:<22} {r['mean_us']/base:>5.2f} x")

    with open(LOG, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['architecture', 'run_id', 'params',
                                          'mean_us', 'max_us', 'std_us', 'reps'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  saved -> {LOG}")


if __name__ == "__main__":
    main()