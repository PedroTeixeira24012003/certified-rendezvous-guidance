"""
summarize_medians.py
Read the per-condition ladder outputs (rob_<rung>_<ctrl>_<runid>.npz) already
on disk and print median and mean fuel and final position error for each
policy in the manifest plus the expert, at the requested rungs. No re-flying,
these arrays were saved by robustness_ladder.py. Also writes
medians_summary.csv.
"""

import os
import csv
import argparse
import numpy as np

# label -> run_id used in the npz filename (ckpt basename without .pt)
EXPERT_NPZ_RUNID = "20260813-161306-4t-cea077"   # ckpt passed to the expert run
MANIFEST = [
    ("ta=0.01", "raw",    "20260813-141246-4t-cc7328"),
    ("ta=0.03", "raw",    "20260813-152458-4t-6fc9e3"),
    ("ta=0.1",  "raw",    "20260813-154058-4t-68990a"),
    ("ta=0.3",  "raw",    "20260813-161306-4t-cea077"),
    ("ta=1.0",  "raw",    "20260813-163542-4t-d2afae"),
    ("ta=5.0",  "raw",    "20260813-171313-4t-b1fc55"),
    ("ta=8.0",  "raw",    "20260813-183326-4t-828492"),
    ("expert",  "expert", EXPERT_NPZ_RUNID),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", default=["L2", "L3"])
    ap.add_argument("--conv-only", action="store_true",
                    help="take medians over converged flights only")
    args = ap.parse_args()

    rows = []
    header = f"{'label':>9} {'rung':>4} {'n':>5} {'conv':>5} " \
             f"{'med_fpe[m]':>11} {'med_fuel[m/s]':>14} " \
             f"{'mean_fpe[m]':>12} {'mean_fuel[m/s]':>15}"
    print(header)
    print("-" * len(header))

    for label, ctrl, runid in MANIFEST:
        for rung in args.rungs:
            path = f"rob_{rung}_{ctrl}_{runid}.npz"
            if not os.path.exists(path):
                print(f"{label:>9} {rung:>4}  [missing {path}]")
                continue
            d = np.load(path)
            fpe = np.asarray(d["final_pos_err"], float)
            fuel = np.asarray(d["fuel"], float)
            conv = np.asarray(d["converged"], float).astype(bool)
            n = len(fpe)
            nconv = int(conv.sum())

            if args.conv_only:
                mask = conv
            else:
                mask = np.ones(n, bool)
            if mask.sum() == 0:
                med_fpe = med_fuel = mean_fpe = mean_fuel = float("nan")
            else:
                med_fpe = float(np.median(fpe[mask]))
                med_fuel = float(np.median(fuel[mask]))
                mean_fpe = float(np.mean(fpe[mask]))
                mean_fuel = float(np.mean(fuel[mask]))

            print(f"{label:>9} {rung:>4} {n:>5} {nconv:>5} "
                  f"{med_fpe:>11.4f} {med_fuel:>14.4f} "
                  f"{mean_fpe:>12.4f} {mean_fuel:>15.4f}")
            rows.append([label, rung, n, nconv, args.conv_only,
                         med_fpe, med_fuel, mean_fpe, mean_fuel])

    with open("medians_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "rung", "n", "converged", "conv_only",
                    "median_final_pos_err_m", "median_fuel_mps",
                    "mean_final_pos_err_m", "mean_fuel_mps"])
        w.writerows(rows)
    print("\n  saved -> medians_summary.csv")


if __name__ == "__main__":
    main()