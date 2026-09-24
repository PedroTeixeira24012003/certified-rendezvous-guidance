"""
fly_finale.py
Wrapper around robustness_ladder that adds stopwatches at the three command
points, the policy forward pass, the CBF-CLF QP, and the NMPC solve, and
prints a per-step compute time summary against the control period. The
flights themselves are robustness_ladder untouched, same tapes, rungs and
figures of merit; any robustness_ladder argument passes straight through.
"""
import sys, time
import numpy as np

import robustness_ladder as RL
import deploy_closedloop as DCL
import cbf_filter
import th_mpc_behind_1000run as EXP
from acados_template import AcadosOcpSolver

TIMES = {'net': [], 'filter': [], 'expert': []}

_orig_cmd = DCL.policy_command_history
def _timed_cmd(*a, **k):
    t0 = time.perf_counter()
    out = _orig_cmd(*a, **k)
    TIMES['net'].append(time.perf_counter() - t0)
    return out
DCL.policy_command_history = _timed_cmd

_orig_cmd1 = DCL.policy_command
def _timed_cmd1(*a, **k):
    t0 = time.perf_counter()
    out = _orig_cmd1(*a, **k)
    TIMES['net'].append(time.perf_counter() - t0)
    return out
DCL.policy_command = _timed_cmd1

_orig_filter = cbf_filter.CBFFilter.filter
def _timed_filter(self, *a, **k):
    t0 = time.perf_counter()
    out = _orig_filter(self, *a, **k)
    TIMES['filter'].append(time.perf_counter() - t0)
    return out
cbf_filter.CBFFilter.filter = _timed_filter

_orig_solve = AcadosOcpSolver.solve
def _timed_solve(self, *a, **k):
    t0 = time.perf_counter()
    out = _orig_solve(self, *a, **k)
    TIMES['expert'].append(time.perf_counter() - t0)
    return out
AcadosOcpSolver.solve = _timed_solve

SUMMARY = []

def _stats(arr):
    if len(arr) == 0:
        return None
    a = np.asarray(arr)
    return dict(mean=a.mean(), p99=np.percentile(a, 99), mx=a.max(), n=len(a))

_orig_cell = RL.run_cell
def _timed_cell(rung, controller, *a, **k):
    marks = {kk: len(vv) for kk, vv in TIMES.items()}
    out = _orig_cell(rung, controller, *a, **k)
    seg = {kk: np.asarray(TIMES[kk][marks[kk]:]) for kk in TIMES}
    if controller == 'expert':
        tot = seg['expert']
    elif controller == 'raw':
        tot = seg['net']
    else:                                  # certified: policy + QP, per step
        n = min(len(seg['net']), len(seg['filter']))
        tot = seg['net'][:n] + seg['filter'][:n]
    st = _stats(tot)
    if st:
        print(f"    [time] {rung:>3} {controller:>9}: per-step "
              f"mean {st['mean']*1e3:7.3f} ms   p99 {st['p99']*1e3:7.3f} ms   "
              f"max {st['mx']*1e3:7.3f} ms   ({st['n']:,} steps)")
        SUMMARY.append((rung, controller, st))
    return out
RL.run_cell = _timed_cell

if __name__ == '__main__':
    RL.main()
    print()
    print("=" * 74)
    print("PER-STEP COMPUTE TIME SUMMARY   (control period DT = "
          f"{EXP.DT:.2f} s = {EXP.DT*1e3:.0f} ms)")
    print("=" * 74)
    print(f"{'Rung':>4}  {'Controller':>10}  {'Mean (ms)':>10}  "
          f"{'P99 (ms)':>10}  {'Max (ms)':>10}  {'Budget used (max)':>18}")
    for rung, ctrl, st in SUMMARY:
        frac = st['mx'] / EXP.DT * 100.0
        print(f"{rung:>4}  {ctrl:>10}  {st['mean']*1e3:>10.3f}  "
              f"{st['p99']*1e3:>10.3f}  {st['mx']*1e3:>10.3f}  {frac:>17.2f}%")
    print()
    print("net = one policy forward pass. certified = forward pass + one")
    print("CBF-CLF QP. expert = one NMPC solve. All wall-clock on this CPU.")