"""
cbf_filter.py
CBF-CLF-QP safety filter. Build one CBFFilter and call filter() each control
step with the current state and the network's proposed command; it returns
the nearest admissible command via a small OSQP quadratic program with hard
barrier constraints, a soft CLF, the thrust box, and a logged fallback ladder.
"""

import numpy as np
import osqp
from scipy import sparse


# ============================================================================
# Plant and scenario constants (identical to the expert module)
# ============================================================================
MU = 3.986004418e14
R_KOZ = 5.0
U_MAX = 0.082
V_FAR, V_NEAR, D_HOLD, D_TAPER, VCAP_BETA = 2.0, 0.2, -8.0, 30.0, 12.0
HOLD = np.array([-8.0, 0.0, 0.0])

# certified / default gains
ALPHA1_DEFAULT = 0.1        # certified feasible for alpha1 <= 0.325
ALPHA2_DEFAULT = 1.0        # free tuning knob
ALPHAV_DEFAULT = 1.0        # free tuning knob
GAMMA_DEFAULT = 0.001       # CLF decay rate, gentle backstop
RHO_DEFAULT = 0.1           # CLF slack penalty, kept small so the soft term
                            # neither dominates nor ill-conditions the QP

# CLF weights (expert relative weighting), nominal orbit for the CARE
A_NOM = 3.75e7
Q_POS_CLF, Q_VEL_CLF, R_CTRL_CLF = 2.0, 1000.0, 135000.0


# ============================================================================
# Softplus velocity cap and derivative
# ============================================================================
def _sp(w, b): return (1.0 / b) * np.log1p(np.exp(b * w))
def _sp_prime(w, b): return 1.0 / (1.0 + np.exp(-b * w))
def _clamp01(w, b):
    lo = _sp(w, b)
    return 1.0 - _sp(1.0 - lo, b)


def v_cap(d):
    w = (d - D_HOLD) / (D_TAPER - D_HOLD)
    return V_NEAR + (V_FAR - V_NEAR) * _clamp01(w, VCAP_BETA)


def v_cap_prime(d):
    w = (d - D_HOLD) / (D_TAPER - D_HOLD)
    sig = _sp_prime(1.0 - _sp(w, VCAP_BETA), VCAP_BETA) * _sp_prime(w, VCAP_BETA)
    return (V_FAR - V_NEAR) / (D_TAPER - D_HOLD) * sig


# ============================================================================
# Barrier drift terms, closed forms
# ============================================================================
def _theta_dot(a, e, th):
    c = np.cos(th)
    return (1.0 + e * c) ** 2 / (1.0 - e ** 2) ** 1.5 * np.sqrt(MU / a ** 3)


def _theta_ddot(a, e, th):
    c = np.cos(th)
    return (-2.0 * e * (1.0 + e * c) * np.sin(th) / (1.0 - e ** 2) ** 1.5
            * np.sqrt(MU / a ** 3) * _theta_dot(a, e, th))


def _drift_accel(r, v, a, e, th):
    """f_v: drift acceleration (u = 0), from th_rhs with the command removed."""
    x, y, z = r; vx, vy, vz = v
    c = np.cos(th); den = 1.0 + e * c
    thd = _theta_dot(a, e, th); thdd = _theta_ddot(a, e, th)
    ax = -thdd * y + (e * c) / den * thd ** 2 * x - 2.0 * thd * vy
    ay = thdd * x + (3.0 + e * c) / den * thd ** 2 * y + 2.0 * thd * vx
    az = -(1.0 / den) * thd ** 2 * z
    return np.array([ax, ay, az])


# ============================================================================
# Build P once from the CARE, and the CW A, B for V_dot
# ============================================================================
def _build_clf():
    from scipy.linalg import solve_continuous_are
    n = np.sqrt(MU / A_NOM ** 3)
    A = np.zeros((6, 6))
    A[0, 3] = 1; A[1, 4] = 1; A[2, 5] = 1
    A[3, 4] = -2 * n
    A[4, 1] = 3 * n ** 2; A[4, 3] = 2 * n
    A[5, 2] = -n ** 2
    B = np.zeros((6, 3)); B[3, 0] = 1; B[4, 1] = 1; B[5, 2] = 1
    Qc = np.diag([Q_POS_CLF] * 3 + [Q_VEL_CLF] * 3)
    Rc = np.diag([R_CTRL_CLF] * 3)
    P = solve_continuous_are(A, B, Qc, Rc)
    # Normalise P to unit spectral norm. Scaling P leaves the CLF condition
    # unchanged (both sides scale), so the Lyapunov property is preserved
    # while the CLF stays on a numeric scale comparable to the thrust.
    P = P / np.linalg.norm(P, 2)
    return P


_P_CLF = _build_clf()


# ============================================================================
# The filter
# ============================================================================
class CBFFilter:
    """CBF-CLF-QP safety filter. Construct once, call filter() each step.

    Parameters
    ----------
    alpha1, alpha2, alpha_v : barrier gains. alpha1 defaults to 0.1, inside
        the certified bound alpha1 <= 0.325; alpha2 and alpha_v are free
        tuning knobs.
    gamma : CLF decay rate.
    rho   : CLF slack penalty.
    eps   : small inner margin on the barrier conditions, absorbing
        inter-sample excursion between control instants.
    """

    def __init__(self, alpha1=ALPHA1_DEFAULT, alpha2=ALPHA2_DEFAULT,
                 alpha_v=ALPHAV_DEFAULT, gamma=GAMMA_DEFAULT, rho=RHO_DEFAULT,
                 eps=0.0, verbose=False):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha_v = alpha_v
        self.gamma = gamma
        self.rho = rho
        self.eps = eps
        self.P = _P_CLF

        # instrumentation
        self.n_steps = 0
        self.n_intervene = 0
        self.n_fallback_clf = 0      # stage 2 activations
        self.n_fallback_viol = 0     # stage 3 activations
        self.n_solver_fail = 0
        self.correction_mags = []    # ||u_filt - u_net|| when it intervened
        self.solve_times_us = []

        # OSQP is set up once with the fixed sparsity and warm-started via update
        self._prob = None
        self._setup_osqp()

    # -- assemble the per-step data ------------------------------------------
    def _rows(self, state):
        r = state[:3]; v = state[3:6]; th = state[6]
        # a, e are needed for the drift terms; carried on the filter per call
        a, e = self._a, self._e
        d = np.linalg.norm(r) + 1e-12
        fv = _drift_accel(r, v, a, e, th)

        # keep-out HOCBF condition:  b_koz . u >= -rhs_koz
        psi0 = r.dot(r) - R_KOZ ** 2
        psi1 = 2.0 * r.dot(v) + self.alpha1 * psi0
        a_koz = (2.0 * v.dot(v) + 2.0 * r.dot(fv) + 2.0 * self.alpha1 * r.dot(v))
        b_koz = 2.0 * r
        rhs_koz = a_koz + self.alpha2 * psi1 - self.eps

        # velocity-cap CBF condition:  b_v . u >= -rhs_v
        hv = v_cap(d) ** 2 - v.dot(v)
        a_vcap = 2.0 * v_cap(d) * v_cap_prime(d) * (r.dot(v) / d) - 2.0 * v.dot(fv)
        b_v = -2.0 * v
        rhs_v = a_vcap + self.alpha_v * hv - self.eps

        # CLF condition:  grad_u(V_dot) . u <= -gamma V - (drift V_dot) + delta
        xi = np.concatenate([r - HOLD, v])
        Pxi = self.P @ xi
        # V_dot = 2 xi^T P [v; fv] + 2 xi^T P B u ; B picks the velocity rows
        drift_Vdot = 2.0 * (Pxi[:3].dot(v) + Pxi[3:].dot(fv))
        grad_u_V = 2.0 * Pxi[3:]                    # coefficient on u
        V = xi.dot(Pxi)
        clf_rhs = -self.gamma * V - drift_Vdot      # grad_u_V . u - delta <= clf_rhs

        return b_koz, rhs_koz, b_v, rhs_v, grad_u_V, clf_rhs

    # -- OSQP setup (full sparsity pattern; values updated each step) --------
    def _setup_osqp(self):
        # decision z = [ux, uy, uz, delta]
        Pmat = sparse.diags([2.0, 2.0, 2.0, 2.0 * self.rho]).tocsc()
        q = np.zeros(4)
        # Fixed pattern: every entry that can be nonzero is marked. To keep the
        # nnz fixed at exactly these positions regardless of the numeric values
        # (a zero b_koz component must NOT collapse the pattern), the constraint
        # matrix is carried as a dense array and converted to CSC through a fixed
        # boolean mask each step, so the value vector always has the same length.
        self._mask = np.array([
            [True,  True,  True,  False],   # keep-out u-cols
            [True,  True,  True,  False],   # vel-cap  u-cols
            [True,  True,  True,  True],    # CLF      u-cols + delta
            [True,  False, False, False],   # box ux
            [False, True,  False, False],   # box uy
            [False, False, True,  False],   # box uz
            [False, False, False, True],    # delta >= 0
        ])
        A_init = np.where(self._mask, 1e-30, 0.0)
        A_init[3, 0] = 1.0; A_init[4, 1] = 1.0; A_init[5, 2] = 1.0
        A_init[2, 3] = -1.0; A_init[6, 3] = 1.0
        A = sparse.csc_matrix(A_init)
        l = np.array([-np.inf, -np.inf, -np.inf, -U_MAX, -U_MAX, -U_MAX, 0.0])
        u = np.array([np.inf, np.inf, np.inf, U_MAX, U_MAX, U_MAX, np.inf])
        self._prob = osqp.OSQP()
        self._prob.setup(Pmat, q, A, l, u, warm_start=True, verbose=False,
                         eps_abs=1e-6, eps_rel=1e-6, max_iter=6000, polish=False)
        # CSC value order for our pattern, computed once from the mask
        self._csc_from_dense = lambda M: M.T[self._mask.T]   # column-major (CSC) order
        self._nnz = int(self._mask.sum())

    def _solve(self, u_net, rows, use_clf=True):
        import time
        b_koz, rhs_koz, b_v, rhs_v, grad_u_V, clf_rhs = rows
        q = np.array([-2.0 * u_net[0], -2.0 * u_net[1], -2.0 * u_net[2], 0.0])
        A = np.zeros((7, 4))
        A[0, :3] = -b_koz
        A[1, :3] = -b_v
        A[2, :3] = grad_u_V
        A[2, 3] = -1.0
        A[3, 0] = 1.0; A[4, 1] = 1.0; A[5, 2] = 1.0
        A[6, 3] = 1.0
        Ax = self._csc_from_dense(A)               # fixed-length value vector
        u_r3 = clf_rhs if use_clf else np.inf
        u_bounds = np.array([rhs_koz, rhs_v, u_r3, U_MAX, U_MAX, U_MAX, np.inf])
        l_bounds = np.array([-np.inf, -np.inf, -np.inf, -U_MAX, -U_MAX, -U_MAX, 0.0])
        self._prob.update(q=q, Ax=Ax, l=l_bounds, u=u_bounds)
        t0 = time.perf_counter()
        res = self._prob.solve()
        dt_us = (time.perf_counter() - t0) * 1e6
        return res, dt_us

    # -- the public per-step call --------------------------------------------
    def filter(self, state, u_net, a, e):
        """Return the certified command for one step.

        Returns dict with:
            u        : the filtered command (np.array, 3)
            intervened : bool
            correction : ||u - u_net||
            solve_us   : QP solve time [microseconds]
            fallback   : 0 (primary), 2 (safety-only), 3 (min-violation)
        """
        self._a, self._e = a, e
        self.n_steps += 1
        rows = self._rows(state)

        # stage 1: primary QP (barriers hard + CLF soft)
        res, dt_us = self._solve(u_net, rows, use_clf=True)
        fallback = 0

        if res.info.status_val not in (1, 2):     # not solved / solved-inaccurate
            # stage 2: drop the CLF (safety-only)
            self.n_fallback_clf += 1
            fallback = 2
            res, dt2 = self._solve(u_net, rows, use_clf=False)
            dt_us += dt2
            if res.info.status_val not in (1, 2):
                # stage 3: minimise the worst barrier violation
                self.n_fallback_viol += 1
                fallback = 3
                u_out = self._min_violation(state, rows)
                self.solve_times_us.append(dt_us)
                corr = float(np.linalg.norm(u_out - u_net))
                self.n_intervene += 1
                self.correction_mags.append(corr)
                return dict(u=u_out, intervened=True, correction=corr,
                            solve_us=dt_us, fallback=fallback)

        u_out = np.clip(res.x[:3], -U_MAX, U_MAX)
        self.solve_times_us.append(dt_us)
        corr = float(np.linalg.norm(u_out - u_net))
        intervened = corr > 1e-9
        if intervened:
            self.n_intervene += 1
            self.correction_mags.append(corr)
        return dict(u=u_out, intervened=intervened, correction=corr,
                    solve_us=dt_us, fallback=fallback)

    # -- stage-3 fallback: closed-form best-effort braking -------------------
    def _min_violation(self, state, rows):
        """If even the safety-only QP is infeasible, push the command to the
        box corner that best satisfies the more violated barrier."""
        b_koz, rhs_koz, b_v, rhs_v, _, _ = rows
        # residual of each barrier at u = 0
        res_koz = rhs_koz
        res_v = rhs_v
        b = b_koz if res_koz < res_v else b_v
        return U_MAX * np.sign(b)

    # -- summary --------------------------------------------------------------
    def summary(self):
        st = np.array(self.solve_times_us) if self.solve_times_us else np.array([0.0])
        cm = np.array(self.correction_mags) if self.correction_mags else np.array([0.0])
        return dict(
            n_steps=self.n_steps,
            n_intervene=self.n_intervene,
            intervention_rate=self.n_intervene / max(self.n_steps, 1),
            n_fallback_clf=self.n_fallback_clf,
            n_fallback_viol=self.n_fallback_viol,
            correction_mean=float(cm.mean()),
            correction_max=float(cm.max()),
            solve_us_mean=float(st.mean()),
            solve_us_max=float(st.max()),
            solve_us_p99=float(np.percentile(st, 99)),
        )


# ============================================================================
# Self-test, filter a few states, confirm it runs and respects the box
# ============================================================================
if __name__ == "__main__":
    print("CBF filter self-test")
    print(f"  P built from CARE, expert weights; ||P|| = {np.linalg.norm(_P_CLF):.1f}")
    filt = CBFFilter()
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.uniform(6.0, 100.0); dirn = rng.normal(size=3); dirn /= np.linalg.norm(dirn)
        r = d * dirn; v = rng.uniform(-0.5, 0.5, size=3); th = rng.uniform(0, 2*np.pi)
        state = np.array([*r, *v, th])
        u_net = np.clip(rng.normal(0, 0.03, size=3), -U_MAX, U_MAX)
        out = filt.filter(state, u_net, a=3.5e7, e=0.2)
        assert np.all(np.abs(out['u']) <= U_MAX + 1e-6), "box violated!"
        print(f"  d={d:6.1f}  corr={out['correction']:.4f}  "
              f"solve={out['solve_us']:6.1f} us  fallback={out['fallback']}")
    s = filt.summary()
    print(f"  intervention rate: {s['intervention_rate']*100:.0f}%   "
          f"fallbacks: clf={s['n_fallback_clf']} viol={s['n_fallback_viol']}   "
          f"solve mean {s['solve_us_mean']:.1f} us")
    print("  PASS: filter runs, respects the thrust box.")