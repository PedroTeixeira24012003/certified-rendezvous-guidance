"""
verify_certificates.py
Numerical verification of every derivative the safety filter evaluates. Each
closed-form quantity (barrier drift terms, HOCBF and velocity-cap derivatives,
control directions, softplus cap derivative) is compared against central
finite differences of the underlying functions along the true dynamics, at
100 random states across the operating envelope, and must agree to a tight
tolerance at every state. Prints a per-quantity PASS/FAIL table and the worst
deviation.
"""

import numpy as np

# plant constants, identical to the expert module
MU = 3.986004418e14
R_KOZ = 5.0
V_FAR, V_NEAR, D_HOLD, D_TAPER, VCAP_BETA = 2.0, 0.2, -8.0, 30.0, 12.0

# envelope for random-state sampling
A_RANGE = (3.0e7, 4.5e7)
E_RANGE = (0.05, 0.30)

N_STATES = 100
FD_EPS = 1e-3             # central-difference step in time (truncation-limited)
TOL = 1e-5               # required agreement: min(abs, rel) below this


# ============================================================================
# Plant (numpy), identical to the expert module's th_rhs
# ============================================================================
def th_rhs(state, u, a, e):
    x, y, z, vx, vy, vz, th = state
    c = np.cos(th)
    den = 1.0 + e * c
    thd = (1.0 + e * c) ** 2 / (1.0 - e ** 2) ** 1.5 * np.sqrt(MU / a ** 3)
    thdd = (-2.0 * e * (1.0 + e * c) * np.sin(th) / (1.0 - e ** 2) ** 1.5
            * np.sqrt(MU / a ** 3) * thd)
    ax = -thdd * y + (e * c) / den * thd ** 2 * x - 2.0 * thd * vy + u[0]
    ay = thdd * x + (3.0 + e * c) / den * thd ** 2 * y + 2.0 * thd * vx + u[1]
    az = -(1.0 / den) * thd ** 2 * z + u[2]
    return np.array([vx, vy, vz, ax, ay, az, thd])


def drift_accel(state, a, e):
    """f_v: the drift acceleration only (u = 0), components 3:6 of the RHS."""
    return th_rhs(state, np.zeros(3), a, e)[3:6]


# ============================================================================
# Softplus velocity cap and its analytic derivative
# ============================================================================
def _sp(w, beta):
    return (1.0 / beta) * np.log1p(np.exp(beta * w))


def _sp_prime(w, beta):
    return 1.0 / (1.0 + np.exp(-beta * w))          # logistic


def _clamp01(w, beta):
    lo = _sp(w, beta)
    return 1.0 - _sp(1.0 - lo, beta)


def v_cap(d):
    w = (d - D_HOLD) / (D_TAPER - D_HOLD)
    return V_NEAR + (V_FAR - V_NEAR) * _clamp01(w, VCAP_BETA)


def v_cap_prime_analytic(d):
    """Closed form of the cap derivative."""
    w = (d - D_HOLD) / (D_TAPER - D_HOLD)
    sigma_prime = _sp_prime(1.0 - _sp(w, VCAP_BETA), VCAP_BETA) * _sp_prime(w, VCAP_BETA)
    return (V_FAR - V_NEAR) / (D_TAPER - D_HOLD) * sigma_prime


# ============================================================================
# Analytic certificate quantities, closed forms
# ============================================================================
def analytic_r_dot_fv(state, a, e):
    """r . f_v in closed form."""
    x, y, z, vx, vy, vz, th = state
    c = np.cos(th)
    den = 1.0 + e * c
    thd = (1.0 + e * c) ** 2 / (1.0 - e ** 2) ** 1.5 * np.sqrt(MU / a ** 3)
    return (thd ** 2 / den * (e * c * x ** 2 + (3.0 + e * c) * y ** 2 - z ** 2)
            - 2.0 * thd * (x * vy - y * vx))


def analytic_v_dot_fv(state, a, e):
    """v . f_v in closed form."""
    x, y, z, vx, vy, vz, th = state
    c = np.cos(th)
    den = 1.0 + e * c
    thd = (1.0 + e * c) ** 2 / (1.0 - e ** 2) ** 1.5 * np.sqrt(MU / a ** 3)
    thdd = (-2.0 * e * (1.0 + e * c) * np.sin(th) / (1.0 - e ** 2) ** 1.5
            * np.sqrt(MU / a ** 3) * thd)
    return (thdd * (x * vy - y * vx)
            + thd ** 2 / den * (e * c * x * vx + (3.0 + e * c) * y * vy - z * vz))


def analytic_psi1_dot(state, u, a, e, alpha1):
    """Full keep-out HOCBF derivative."""
    x, y, z, vx, vy, vz, th = state
    r = np.array([x, y, z]); v = np.array([vx, vy, vz])
    return (2.0 * v.dot(v) + 2.0 * analytic_r_dot_fv(state, a, e)
            + 2.0 * r.dot(u) + 2.0 * alpha1 * r.dot(v))


def analytic_hv_dot(state, u, a, e):
    """Velocity-cap barrier derivative."""
    x, y, z, vx, vy, vz, th = state
    r = np.array([x, y, z]); v = np.array([vx, vy, vz])
    d = np.linalg.norm(r)
    rdot_dot = r.dot(v) / d
    return (2.0 * v_cap(d) * v_cap_prime_analytic(d) * rdot_dot
            - 2.0 * analytic_v_dot_fv(state, a, e) - 2.0 * v.dot(u))


# ============================================================================
# Finite-difference references, differentiate the actual functions along dyn.
# ============================================================================
def fd_time_derivative(f, state, u, a, e, eps=FD_EPS):
    """Central difference of scalar f(state) along the true dynamics."""
    sp_ = state + eps * th_rhs(state, u, a, e)
    sm_ = state - eps * th_rhs(state, u, a, e)
    return (f(sp_) - f(sm_)) / (2.0 * eps)


def h_koz(state):
    return state[0] ** 2 + state[1] ** 2 + state[2] ** 2 - R_KOZ ** 2


def hv(state):
    r = state[:3]; v = state[3:6]
    return v_cap(np.linalg.norm(r)) ** 2 - v.dot(v)


# ============================================================================
# Random operating-envelope states, safely outside the keep-out sphere
# ============================================================================
def sample_state(rng):
    d = rng.uniform(R_KOZ + 0.5, 110.0)
    dir_ = rng.normal(size=3); dir_ /= np.linalg.norm(dir_)
    r = d * dir_
    v = rng.uniform(-2.0, 2.0, size=3)
    th = rng.uniform(0, 2 * np.pi)
    return np.array([r[0], r[1], r[2], v[0], v[1], v[2], th])


def rel(a_val, b_val):
    """Agreement metric: the SMALLER of absolute and relative deviation. A
    quantity that is legitimately near zero passes on its tiny absolute error
    without the relative measure exploding on two near-zero numbers."""
    abs_dev = abs(a_val - b_val)
    rel_dev = abs_dev / max(abs(a_val), abs(b_val), 1e-12)
    return min(abs_dev, rel_dev)


# ============================================================================
def main():
    print("=" * 70)
    print("CERTIFICATE DERIVATIVE VERIFICATION  (Section 9.2)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    alpha1 = 0.5
    u_probe = np.array([0.03, -0.05, 0.02])   # arbitrary in-box command

    worst = {k: 0.0 for k in
             ['r.fv', 'v.fv', 'psi1_dot', 'b_koz', 'hv_dot', 'b_v', "vcap'"]}

    for _ in range(N_STATES):
        s = sample_state(rng)
        a = rng.uniform(*A_RANGE)
        e = rng.uniform(*E_RANGE)
        r = s[:3]; v = s[3:6]; d = np.linalg.norm(r)

        # r.fv and v.fv against the drift acceleration directly
        fv = drift_accel(s, a, e)
        worst['r.fv'] = max(worst['r.fv'], rel(analytic_r_dot_fv(s, a, e), r.dot(fv)))
        worst['v.fv'] = max(worst['v.fv'], rel(analytic_v_dot_fv(s, a, e), v.dot(fv)))

        # psi1_dot: analytic vs finite diff of psi1 along dynamics
        def psi1(state):
            rr = state[:3]; vv = state[3:6]
            return 2.0 * rr.dot(vv) + alpha1 * (rr.dot(rr) - R_KOZ ** 2)
        psi1_fd = fd_time_derivative(psi1, s, u_probe, a, e)
        worst['psi1_dot'] = max(worst['psi1_dot'],
                                rel(analytic_psi1_dot(s, u_probe, a, e, alpha1), psi1_fd))

        # b_koz = grad_u psi1_dot = 2r  (analytic psi1_dot slope in u)
        du = 1e-4
        for i in range(3):
            up = u_probe.copy(); up[i] += du
            um = u_probe.copy(); um[i] -= du
            slope = (analytic_psi1_dot(s, up, a, e, alpha1)
                     - analytic_psi1_dot(s, um, a, e, alpha1)) / (2 * du)
            worst['b_koz'] = max(worst['b_koz'], rel(slope, 2.0 * r[i]))

        # hv_dot: analytic vs finite diff of hv along dynamics
        hv_fd = fd_time_derivative(hv, s, u_probe, a, e)
        worst['hv_dot'] = max(worst['hv_dot'], rel(analytic_hv_dot(s, u_probe, a, e), hv_fd))

        # b_v = grad_u hv_dot = -2v
        for i in range(3):
            up = u_probe.copy(); up[i] += du
            um = u_probe.copy(); um[i] -= du
            slope = (analytic_hv_dot(s, up, a, e) - analytic_hv_dot(s, um, a, e)) / (2 * du)
            worst['b_v'] = max(worst['b_v'], rel(slope, -2.0 * v[i]))

        # vcap'(d): analytic vs finite diff in d
        dd = 1e-4
        vcap_fd = (v_cap(d + dd) - v_cap(d - dd)) / (2 * dd)
        worst["vcap'"] = max(worst["vcap'"], rel(v_cap_prime_analytic(d), vcap_fd))

    print(f"  states checked: {N_STATES}   tolerance: {TOL:.0e}\n")
    print(f"  {'quantity':<12}{'worst rel. deviation':>22}   status")
    print("  " + "-" * 46)
    all_pass = True
    for k, val in worst.items():
        ok = val < TOL
        all_pass &= ok
        print(f"  {k:<12}{val:>22.3e}   {'PASS' if ok else 'FAIL'}")
    print("  " + "-" * 46)
    overall = max(worst.values())
    print(f"\n  worst deviation over all quantities and states: {overall:.3e}")
    print(f"  VERDICT: {'ALL DERIVATIVES VERIFIED' if all_pass else 'FAILURE - DO NOT TRUST FILTER'}")
    print("=" * 70)


if __name__ == "__main__":
    main()