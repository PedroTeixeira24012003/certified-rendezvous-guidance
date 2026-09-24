"""
expert_iface.py
One-function wrapper around the acados TH-NMPC expert, imported by
make_tasil_targets.py. The solver is built once at import and reused; each
query pins the state through lbx/ubx on stage 0, pushes p=[a,e] to every
stage, solves and returns the first command. acados retains the previous
solution internally, so queries issued in trajectory order inherit a warm
start from the neighbouring state. Nonzero solver statuses are counted and
reported at exit.
"""

import atexit
import numpy as np

# take whichever expert module name is present
try:
    import th_mpc_behind_1000run as _exp
except ImportError:
    import th_mpc_expert as _exp

_solver = _exp.build_solver()
_N_P = _exp.N_P
_n_calls = 0
_n_fail = 0


def expert_command(r, v, theta, a, e):
    """Solve the expert OCP once from the given state, return the first command."""
    global _n_calls, _n_fail
    state = np.array([r[0], r[1], r[2], v[0], v[1], v[2], theta], dtype=float)
    p_val = np.array([a, e], dtype=float)
    for j in range(_N_P + 1):
        _solver.set(j, "p", p_val)
    _solver.set(0, "lbx", state)
    _solver.set(0, "ubx", state)
    status = _solver.solve()
    _n_calls += 1
    if status != 0:
        _n_fail += 1
    return np.array(_solver.get(0, "u"), dtype=float)


@atexit.register
def _report():
    if _n_calls:
        print(f"[expert_iface] {_n_calls:,} solves, {_n_fail:,} nonzero-status "
              f"({100.0 * _n_fail / _n_calls:.3f}%)")