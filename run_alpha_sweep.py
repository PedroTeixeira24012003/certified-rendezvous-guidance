"""
run_alpha_sweep.py
Filter tuning sweep. With alpha1 fixed by the feasibility verification, this
driver sweeps alpha2, alpha_v and the CLF slack penalty rho over a grid,
flies a filtered paired-thousand at each setting through
deploy_closedloop.run_e3, and writes the intervention-versus-margin tradeoff
table to alpha_sweep.csv, one row per setting.
"""

import os
import sys
import csv
import time
import argparse
import itertools
import numpy as np

import deploy_closedloop as DCL
import cbf_filter
import bc_data

# fixed by the feasibility verification
ALPHA1_FIXED = 0.1

# the free tuning grid
ALPHA2_GRID = [0.5, 1.0, 2.0]
ALPHAV_GRID = [0.5, 1.0, 2.0]
RHO_GRID = [0.1, 1.0, 10.0]

# coarse grid for a quick look
ALPHA2_COARSE = [0.5, 2.0]
ALPHAV_COARSE = [0.5, 2.0]
RHO_COARSE = [0.1, 1.0]

SWEEP_CSV = 'alpha_sweep.csv'


def one_setting(model, refs, x_mean, x_std, base_run_id, a2, av, rho):
    """Fly the filtered paired-thousand at one gain setting; return a summary."""
    filt = cbf_filter.CBFFilter(alpha1=ALPHA1_FIXED, alpha2=a2,
                                alpha_v=av, rho=rho)
    run_id = f"{base_run_id}_sweep_a2-{a2}_av-{av}_rho-{rho}"
    # run_e3 prints its own block and writes cl_E3_<run_id>.npz
    summ = DCL.run_e3(model, refs, x_mean, x_std, run_id, filt=filt)
    # augment with the per-setting means for the tradeoff table
    e3 = np.load(f'cl_E3_{run_id}.npz')
    corr_max = float(np.max(e3['filt_corr_max']))
    rate_mean = float(np.mean(e3['filt_rate']))
    vcap_max = float(np.max(e3['net_vcap_viol']))
    summ.update(alpha2=a2, alpha_v=av, rho=rho,
                corr_max=corr_max, rate_mean=rate_mean, vcap_max=vcap_max)
    return summ


def main():
    p = argparse.ArgumentParser(description="Filter tuning sweep (Section 9.4.2).")
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--latest', action='store_true')
    p.add_argument('--coarse', action='store_true',
                   help='8-setting grid instead of the full 27')
    args = p.parse_args()

    if args.latest:
        ckpt_path = DCL.latest_checkpoint()
    elif args.ckpt:
        ckpt_path = args.ckpt
    else:
        p.error("give --ckpt PATH or --latest")

    if not os.path.exists(DCL.REFS_FILE):
        sys.exit(f"{DCL.REFS_FILE} not found - run cache_expert_references.py first")

    a2s = ALPHA2_COARSE if args.coarse else ALPHA2_GRID
    avs = ALPHAV_COARSE if args.coarse else ALPHAV_GRID
    rhos = RHO_COARSE if args.coarse else RHO_GRID
    settings = list(itertools.product(a2s, avs, rhos))

    print("=" * 70)
    print("FILTER TUNING SWEEP  (Section 9.4.2)")
    print("=" * 70)
    print(f"  checkpoint : {ckpt_path}")
    print(f"  alpha1 fixed at {ALPHA1_FIXED} (feasibility, Section 9.3)")
    print(f"  grid       : alpha2 {a2s} x alpha_v {avs} x rho {rhos}")
    print(f"  settings   : {len(settings)}   (each is a filtered E3 of 1000 flights)")
    print("=" * 70)

    model, cfg, ck = DCL.load_checkpoint(ckpt_path)
    base_run_id = ck.get('run_id',
                         os.path.basename(ckpt_path).replace('.pt', ''))
    refs = np.load(DCL.REFS_FILE)
    x_mean, x_std = bc_data.load_norm_stats()

    rows = []
    t_start = time.time()
    for idx, (a2, av, rho) in enumerate(settings, 1):
        el = (time.time() - t_start) / 60.0
        print(f"\n[{idx}/{len(settings)}]  alpha2={a2} alpha_v={av} rho={rho}"
              f"   (elapsed {el:.1f} min)")
        summ = one_setting(model, refs, x_mean, x_std, base_run_id, a2, av, rho)
        rows.append(summ)

    # write the tradeoff table
    fields = ['alpha2', 'alpha_v', 'rho', 'conv', 'viol', 'divergent',
              'margin_min', 'vcap_max', 'rate_mean', 'corr_max',
              'filt_fallback_total', 'filt_solve_mean', 'filt_solve_max',
              'fuel_d_mean']
    with open(SWEEP_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # console summary table
    print("\n" + "=" * 70)
    print("TRADEOFF TABLE  (Section 9.4.2)")
    print("=" * 70)
    hdr = (f"{'a2':>4} {'av':>4} {'rho':>5} | {'conv':>5} {'viol':>4} "
           f"{'margin':>7} {'vcap':>7} | {'rate%':>6} {'corr_max':>9} "
           f"{'fb':>3} {'solve_us':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['alpha2']:>4} {r['alpha_v']:>4} {r['rho']:>5} | "
              f"{r['conv']:>5} {r['viol']:>4} "
              f"{r['margin_min']:>7.3f} {r['vcap_max']:>7.4f} | "
              f"{100*r['rate_mean']:>6.2f} {r['corr_max']:>9.5f} "
              f"{r['filt_fallback_total']:>3} {r['filt_solve_mean']:>9.1f}")
    print("=" * 70)
    print(f"  wrote {SWEEP_CSV}")
    print(f"  total time: {(time.time()-t_start)/60.0:.1f} min")

    # least-intrusive setting that held safety with no fallbacks
    safe = [r for r in rows if r['viol'] == 0 and r['filt_fallback_total'] == 0]
    if safe:
        knee = min(safe, key=lambda r: (r['rate_mean'], r['corr_max']))
        print(f"  least-intrusive safe setting: alpha2={knee['alpha2']} "
              f"alpha_v={knee['alpha_v']} rho={knee['rho']}  "
              f"(rate {100*knee['rate_mean']:.2f}%, corr_max {knee['corr_max']:.5f})")


if __name__ == "__main__":
    main()