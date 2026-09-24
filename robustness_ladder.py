"""
robustness_ladder.py
Disturbance driver. Flies the three controllers, the expert, the raw network,
and the certified policy, through a six-rung ladder of unmodelled disturbances
(navigation error, thrust error, environmental accelerations) over the paired
thousand. Every disturbance enters the true propagation and nowhere else: nav
error corrupts only the state the controller reads, thrust error corrupts the
command after the controller issues it, environmental acceleration is added
inside the true RK4 step only, and all figures of merit are measured on the
true state. Writes per-cell FoM arrays and a summary CSV.
"""

import os
import sys
import csv
import time
import zlib
import argparse
import numpy as np

import th_mpc_behind_1000run as EXP     # frozen plant + expert loop
import deploy_closedloop as DCL         # frozen network flight
import bc_data                          # normalization (training-identical)
import cbf_filter                       # frozen safety filter


# ============================================================================
# LADDER DEFINITION
# ============================================================================
KAPPA = 3.0        # margin inflation, 3-sigma
BETA = 0.5         # nav bias fraction (bias drawn at half the white sigma)

# rung -> dict of severities. lam scales sigma_r(R); env toggles the three
# environmental perturbations; sigv/gain/bias/sigu are the feedback scales.
LADDER = {
    'L0': dict(lam=0.0, sigv=0.0,   gain=0.00, bias=0.000, sigu=0.0000, env=False),
    'L1': dict(lam=0.0, sigv=0.0,   gain=0.00, bias=0.000, sigu=0.0000, env=True),
    'L2': dict(lam=0.2, sigv=0.001, gain=0.02, bias=0.001, sigu=0.0005, env=True),
    'L3': dict(lam=1.0, sigv=0.005, gain=0.05, bias=0.002, sigu=0.0010, env=True),
    'L4': dict(lam=4.0, sigv=0.020, gain=0.10, bias=0.004, sigu=0.0020, env=True),
    'L5': dict(lam=12.0, sigv=0.050, gain=0.15, bias=0.008, sigu=0.0040, env=True),
}
RUNG_ORDER = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']
CONTROLLERS = ['expert', 'raw', 'certified']


# ============================================================================
# NAVIGATION ERROR MODEL: Madonna et al. 2025 along-boresight curve
# sigma_r(R) = -3.352e-2 + 3.858e-2 * R^0.24   [m], R in metres
# adopted isotropically (conservative: worst sensor axis on all components)
# ============================================================================
def sigma_r_of_R(R):
    """Range-dependent position sigma [m], Madonna along-boresight, floored at 0."""
    s = -3.352e-2 + 3.858e-2 * (max(R, 1e-6) ** 0.24)
    return max(s, 0.0)


# ============================================================================
# ENVIRONMENTAL DIFFERENTIAL ACCELERATION
# Added to the TRUE acceleration only. Three terms, each a differential across
# the servicer-target separation rho = ||r||. Directions are fixed per flight
# from the disturbance tape; all terms are far below u_max by construction.
# ============================================================================
J2_E = 1.08263e-3
RE_E = 6.378137e6
MU_SUN = 1.32712440018e20
MU_MOON = 4.9028e12
AU_M = 1.495978707e11
D_MOON = 3.844e8
P_SRP = 4.57e-6            # N/m^2 (Madonna Table 1)


def env_accel(r_vec, a, e, th, dir_j2, dir_srp, dir_3b, srp_mismatch):
    """Differential environmental acceleration [m/s^2, 3-vector] at the true state.
    r_vec: relative position (m). a,e,th: orbit. dir_*: fixed unit directions for
    this flight (from the tape). srp_mismatch: |Delta(Cr A/m)| for this flight."""
    rho = np.linalg.norm(r_vec)
    # target radius from the orbit equation
    c = np.cos(th)
    p = a * (1.0 - e ** 2)
    r_t = p / (1.0 + e * c)
    # J2 differential: 4 a_J2 / r_t * rho, a_J2 = 1.5 J2 mu RE^2 / r_t^4
    a_j2 = 1.5 * J2_E * EXP.MU * RE_E ** 2 / r_t ** 4
    d_j2 = 4.0 * a_j2 / r_t * rho
    # SRP differential: P * |Delta(Cr A/m)|  (independent of r_t and rho)
    d_srp = P_SRP * srp_mismatch
    # third-body differential: 2 mu3 / D^3 * rho (Sun + Moon)
    d_3b = 2.0 * (MU_SUN / AU_M ** 3 + MU_MOON / D_MOON ** 3) * rho
    return d_j2 * dir_j2 + d_srp * dir_srp + d_3b * dir_3b


def rk4_step_disturbed(state, u, dt, a, e, env_fn):
    """RK4 that adds env acceleration to the TRUE derivative only. env_fn(state)
    returns the 3-vector environmental accel; it is added to rows 3:6 of the
    derivative at every stage. Identical to EXP.rk4_step when env_fn returns 0."""
    def rhs(s):
        d = EXP.th_rhs_numpy(s, u, a, e)
        d[3:6] = d[3:6] + env_fn(s)
        return d
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================================
# DISTURBANCE TAPE: one per (rung, flight), replayed identically for all three
# controllers, so they meet an identical world. The seed is zlib.crc32 of
# "rung-index", a fixed function of its input, unlike hash(), which Python
# salts per interpreter session; tapes are therefore byte-identical across
# sessions and machines, and results from separate runs are comparable.
# ============================================================================
class DisturbanceTape:
    def __init__(self, rung, flight_idx, cfg):
        # deterministic per-(rung,flight) seed, stable across sessions/machines
        seed = zlib.crc32(f"{rung}-{flight_idx}".encode()) & 0x7FFFFFFF
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.lam = cfg['lam']
        self.sigv = cfg['sigv']
        self.env_on = cfg['env']

        # per-flight constant draws
        # nav bias (position + velocity), drawn once, held through the flight
        if self.lam > 0.0:
            # bias scaled to BETA * (a representative sigma); position bias uses
            # sigma_r at mid-range so the bias is a fixed offset, not range-varying
            sr_ref = self.lam * sigma_r_of_R(60.0)
            self.bias_r = self.rng.normal(0.0, BETA * sr_ref, size=3)
            self.bias_v = self.rng.normal(0.0, BETA * self.sigv, size=3)
        else:
            self.bias_r = np.zeros(3)
            self.bias_v = np.zeros(3)

        # thrust gain + bias, drawn once per flight
        gm = cfg['gain']
        bm = cfg['bias']
        self.gamma = self.rng.uniform(-gm, gm, size=3) if gm > 0 else np.zeros(3)
        self.tbias = self.rng.uniform(-bm, bm, size=3) if bm > 0 else np.zeros(3)
        self.sigu = cfg['sigu']

        # environmental fixed directions + SRP mismatch, drawn once per flight
        if self.env_on:
            self.dir_j2 = self._unit()
            self.dir_srp = self._unit()
            self.dir_3b = self._unit()
            self.srp_mismatch = abs(self.rng.normal(0.002, 0.0005))
        else:
            self.dir_j2 = self.dir_srp = self.dir_3b = np.zeros(3)
            self.srp_mismatch = 0.0

    def _unit(self):
        v = self.rng.normal(size=3)
        return v / (np.linalg.norm(v) + 1e-12)

    def nav_noise(self, R):
        """White nav error at range R: range-dependent position, fixed velocity."""
        if self.lam <= 0.0:
            return np.zeros(6)
        sr = self.lam * sigma_r_of_R(R)
        wr = self.rng.normal(0.0, sr, size=3) + self.bias_r
        wv = self.rng.normal(0.0, self.sigv, size=3) + self.bias_v
        return np.concatenate([wr, wv])

    def thrust_apply(self, u_cmd):
        """Apply gain, bias, white to the command; clip to the box."""
        if self.cfg['gain'] == 0 and self.cfg['bias'] == 0 and self.sigu == 0:
            return u_cmd
        w = self.rng.normal(0.0, self.sigu, size=3) if self.sigu > 0 else np.zeros(3)
        u = (1.0 + self.gamma) * u_cmd + self.tbias + w
        return np.clip(u, -EXP.U_MAX, EXP.U_MAX)

    def env_fn(self, a, e):
        """Return an env-accel closure over the true state, or zero if env off."""
        if not self.env_on:
            return lambda s: np.zeros(3)
        return lambda s: env_accel(s[:3], a, e, s[6],
                                   self.dir_j2, self.dir_srp, self.dir_3b,
                                   self.srp_mismatch)


# ============================================================================
# INFLATED-RADIUS FILTER SHIM. Subclass the frozen CBFFilter and override ONLY
# the keep-out radius used in _rows, reading a per-step attribute. The verified
# barrier maths, drift terms, QP, and fallback ladder are inherited unchanged.
# r_koz_star = R_KOZ + kappa * lambda * sigma_r(R).
# ============================================================================
class InflatedCBFFilter(cbf_filter.CBFFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._r_koz_star = cbf_filter.R_KOZ   # updated each step by the driver

    def _rows(self, state):
        r = state[:3]; v = state[3:6]; th = state[6]
        a, e = self._a, self._e
        d = np.linalg.norm(r) + 1e-12
        fv = cbf_filter._drift_accel(r, v, a, e, th)
        rkoz = self._r_koz_star                      # <-- inflated radius

        psi0 = r.dot(r) - rkoz ** 2
        psi1 = 2.0 * r.dot(v) + self.alpha1 * psi0
        a_koz = (2.0 * v.dot(v) + 2.0 * r.dot(fv) + 2.0 * self.alpha1 * r.dot(v))
        b_koz = 2.0 * r
        rhs_koz = a_koz + self.alpha2 * psi1 - self.eps

        hv = cbf_filter.v_cap(d) ** 2 - v.dot(v)
        a_vcap = (2.0 * cbf_filter.v_cap(d) * cbf_filter.v_cap_prime(d)
                  * (r.dot(v) / d) - 2.0 * v.dot(fv))
        b_v = -2.0 * v
        rhs_v = a_vcap + self.alpha_v * hv - self.eps

        xi = np.concatenate([r - cbf_filter.HOLD, v])
        Pxi = self.P @ xi
        drift_Vdot = 2.0 * (Pxi[:3].dot(v) + Pxi[3:].dot(fv))
        grad_u_V = 2.0 * Pxi[3:]
        V = xi.dot(Pxi)
        clf_rhs = -self.gamma * V - drift_Vdot
        return b_koz, rhs_koz, b_v, rhs_v, grad_u_V, clf_rhs


# ============================================================================
# FIGURES OF MERIT on the TRUE trajectory, never the measured state.
# Same definitions as EXP.run_closed_loop / DCL.fly_network.
# ============================================================================
def compute_fom(X, U, divergent):
    final_pos_err = float(np.linalg.norm(X[-1, :3] - EXP.HOLD))
    final_vel = float(np.linalg.norm(X[-1, 3:6]))
    dist_target = np.linalg.norm(X[:, :3], axis=1)
    min_dist = float(np.min(dist_target))
    koz_ok = bool(min_dist >= EXP.R_KOZ - 1e-6)
    koz_margin = float(min_dist - EXP.R_KOZ)
    speed = np.linalg.norm(X[:, 3:6], axis=1)
    peak_speed = float(np.max(speed))
    caps = np.array([EXP.v_cap_np(d) for d in dist_target])
    vcap_viol = float(np.max(np.maximum(speed - caps, 0.0)))
    fuel = float(np.sum(np.linalg.norm(U, axis=1)) * EXP.DT) if len(U) else 0.0
    converged = bool((not divergent) and final_pos_err < EXP.TOL_POS
                     and final_vel < EXP.TOL_VEL)
    return dict(final_pos_err=final_pos_err, final_vel=final_vel,
                min_dist_target=min_dist, koz_ok=koz_ok, koz_margin=koz_margin,
                peak_speed=peak_speed, vcap_viol=vcap_viol, fuel=fuel,
                converged=converged, divergent=bool(divergent), n_steps=len(U))


# ============================================================================
# FLY THE RAW / CERTIFIED NETWORK under a disturbance tape.
# Mirrors DCL.fly_network but: reads measured state, applies thrust error,
# propagates disturbed true dynamics, feeds inflated radius to the filter,
# measures FoM on the TRUE state.
# ============================================================================
def fly_network_dist(model, x0, theta0, a, e, x_mean, x_std, tape, filt=None):
    import models
    state = np.concatenate([x0, [theta0]])          # TRUE state
    X_hist = [state.copy()]
    U_hist = []
    divergent = False
    env_fn = tape.env_fn(a, e)

    is_history = isinstance(model, models.HistoryPolicy)
    if is_history:
        feats_list, acts_scaled_list = [], []
        k_win = model.k

    for k in range(EXP.SIM_STEPS):
        R = np.linalg.norm(state[:3])
        state_meas = state.copy()
        state_meas[:6] = state[:6] + tape.nav_noise(R)   # NAV ERROR (read only)

        if is_history:
            u = DCL.policy_command_history(model, state_meas, a, e, x_mean, x_std,
                                           feats_list, acts_scaled_list, k_win)
        else:
            u = DCL.policy_command(model, state_meas, a, e, x_mean, x_std)

        if filt is not None:
            filt._r_koz_star = cbf_filter.R_KOZ + KAPPA * tape.lam * sigma_r_of_R(R)
            fout = filt.filter(state_meas, u, a, e)      # filter on measured state
            u = fout['u']

        u_applied = tape.thrust_apply(u)                 # THRUST ERROR (after ctrl)
        U_hist.append(u_applied.copy())
        state = rk4_step_disturbed(state, u_applied, EXP.DT, a, e, env_fn)  # TRUE
        X_hist.append(state.copy())

        if not np.all(np.isfinite(state)):
            divergent = True; break
        if np.linalg.norm(state[:3] - EXP.HOLD) > DCL.DIVERGE_DIST:
            divergent = True; break

    return compute_fom(np.array(X_hist), np.array(U_hist), divergent)


# ============================================================================
# FLY THE EXPERT under a disturbance tape.
# Solver reads the MEASURED state (via lbx/ubx); the true state is propagated
# with the disturbed dynamics and thrust error. Metrics on the TRUE state.
# ============================================================================
def fly_expert_dist(solver, x0, theta0, a, e, tape):
    state = np.concatenate([x0, [theta0]])          # TRUE state
    X_hist = [state.copy()]
    U_hist = []
    divergent = False
    env_fn = tape.env_fn(a, e)
    p_val = np.array([a, e])

    for k in range(EXP.SIM_STEPS):
        R = np.linalg.norm(state[:3])
        state_meas = state.copy()
        state_meas[:6] = state[:6] + tape.nav_noise(R)   # NAV ERROR (read only)

        for j in range(EXP.N_P + 1):
            solver.set(j, "p", p_val)
        solver.set(0, "lbx", state_meas)                 # solver reads MEASURED
        solver.set(0, "ubx", state_meas)
        solver.solve()
        u = solver.get(0, "u")

        u_applied = tape.thrust_apply(u)                 # THRUST ERROR
        U_hist.append(u_applied.copy())
        state = rk4_step_disturbed(state, u_applied, EXP.DT, a, e, env_fn)  # TRUE
        X_hist.append(state.copy())

        if not np.all(np.isfinite(state)):
            divergent = True; break
        if np.linalg.norm(state[:3] - EXP.HOLD) > DCL.DIVERGE_DIST:
            divergent = True; break

    return compute_fom(np.array(X_hist), np.array(U_hist), divergent)


# ============================================================================
# RUN ONE (rung, controller) OVER THE PAIRED THOUSAND
# ============================================================================
def run_cell(rung, controller, refs, model, x_mean, x_std, solver, filt, n_max):
    cfg = LADDER[rung]
    n = min(int(refs['n_mc']), n_max)
    mc_x0 = refs['mc_x0']; mc_th = refs['mc_theta0']
    mc_a = refs['mc_a']; mc_e = refs['mc_e']

    keys = ['final_pos_err', 'final_vel', 'min_dist_target', 'koz_ok',
            'koz_margin', 'peak_speed', 'vcap_viol', 'fuel', 'converged',
            'divergent', 'n_steps']
    out = {k: np.zeros(n) for k in keys}

    t0 = time.time()
    for i in range(n):
        tape = DisturbanceTape(rung, i, cfg)
        if controller == 'expert':
            res = fly_expert_dist(solver, mc_x0[i], float(mc_th[i]),
                                  float(mc_a[i]), float(mc_e[i]), tape)
        elif controller == 'raw':
            res = fly_network_dist(model, mc_x0[i], float(mc_th[i]),
                                   float(mc_a[i]), float(mc_e[i]),
                                   x_mean, x_std, tape, filt=None)
        else:  # certified
            res = fly_network_dist(model, mc_x0[i], float(mc_th[i]),
                                   float(mc_a[i]), float(mc_e[i]),
                                   x_mean, x_std, tape, filt=filt)
        for k in keys:
            out[k][i] = float(res[k])
    dt = time.time() - t0
    return out, dt, n


# ============================================================================
# MAIN
# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Chapter 10 robustness ladder.")
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--latest', action='store_true')
    p.add_argument('--smoke', action='store_true',
                   help='10 flights per cell, print timed ETA for the full run')
    p.add_argument('--rungs', nargs='+', default=RUNG_ORDER)
    p.add_argument('--controllers', nargs='+', default=CONTROLLERS)
    # filter gains, the deployed defaults
    p.add_argument('--alpha1', type=float, default=cbf_filter.ALPHA1_DEFAULT)
    p.add_argument('--alpha2', type=float, default=cbf_filter.ALPHA2_DEFAULT)
    p.add_argument('--alphav', type=float, default=cbf_filter.ALPHAV_DEFAULT)
    p.add_argument('--rho', type=float, default=cbf_filter.RHO_DEFAULT)
    args = p.parse_args()

    ckpt = args.ckpt or (DCL.latest_checkpoint() if args.latest else None)
    if ckpt is None:
        p.error("give --ckpt PATH or --latest")
    if not os.path.exists(DCL.REFS_FILE):
        sys.exit(f"{DCL.REFS_FILE} not found - run cache_expert_references.py first")

    n_max = 10 if args.smoke else 10 ** 9

    print("=" * 70)
    print("CHAPTER 10 ROBUSTNESS LADDER" + ("  [SMOKE]" if args.smoke else ""))
    print("=" * 70)

    model, cfg, ck = DCL.load_checkpoint(ckpt)
    run_id = ck.get('run_id', os.path.basename(ckpt).replace('.pt', ''))
    refs = np.load(DCL.REFS_FILE)
    x_mean, x_std = bc_data.load_norm_stats()
    print(f"  checkpoint : {ckpt}")
    print(f"  rungs      : {args.rungs}")
    print(f"  controllers: {args.controllers}")
    print(f"  conditions : {min(int(refs['n_mc']), n_max)} per cell")

    # the expert solver is (re)built fresh per rung inside the loop below, so no
    # warm-start state carries across rungs; here we only note it will be built
    solver = None
    if 'expert' in args.controllers:
        print("  acados expert solver will be rebuilt fresh at each rung")

    # build the inflated-radius filter only if the certified policy is flown
    filt = None
    if 'certified' in args.controllers:
        filt = InflatedCBFFilter(alpha1=args.alpha1, alpha2=args.alpha2,
                                 alpha_v=args.alphav, rho=args.rho)
    print("=" * 70)

    summary_rows = []
    cell_times = {}
    for rung in args.rungs:
        # Rebuild the expert solver fresh at the start of each rung so that no
        # warm-start state from a previous (possibly severe) rung leaks into
        # this one: every rung's expert starts from a clean internal state and
        # the result is independent of rung order, so the L0 baseline matches
        # the nominal run no matter how --rungs is passed. Within a rung the
        # solver is reused across the 1000 flights, exactly as in the
        # validation study, so the expert's own behaviour is unchanged.
        if 'expert' in args.controllers:
            solver = EXP.build_solver()
        for controller in args.controllers:
            print(f"  {rung} / {controller} ...", end='', flush=True)
            out, dt, n = run_cell(rung, controller, refs, model, x_mean, x_std,
                                  solver, filt, n_max)
            per_flight = dt / max(n, 1)
            cell_times[(rung, controller)] = per_flight
            conv = int(out['converged'].sum())
            viol = int((~out['koz_ok'].astype(bool)).sum())
            div = int(out['divergent'].sum())
            marg = float(out['koz_margin'].min())
            print(f" {conv}/{n} conv, {viol} viol, {div} div, "
                  f"min margin {marg:+.3f} m  [{per_flight:.3f} s/flight]")
            if not args.smoke:
                np.savez_compressed(f'rob_{rung}_{controller}_{run_id}.npz', **out)
            summary_rows.append([rung, controller, n, conv, viol, div,
                                 marg, float(out['fuel'].mean()),
                                 float(out['final_pos_err'].mean()), per_flight])

    # write summary
    if not args.smoke:
        with open(f'rob_summary_{run_id}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['rung', 'controller', 'n', 'converged', 'violations',
                        'divergent', 'min_margin', 'mean_fuel',
                        'mean_final_pos_err', 's_per_flight'])
            w.writerows(summary_rows)
        print(f"\n  summary -> rob_summary_{run_id}.csv")

    # ETA projection from the smoke timings
    if args.smoke:
        print("=" * 70)
        print("  ETA PROJECTION FOR THE FULL 1000-CONDITION RUN")
        print("=" * 70)
        n_full = int(refs['n_mc'])
        total = 0.0
        for controller in CONTROLLERS:
            times = [cell_times[(r, controller)] for r in RUNG_ORDER
                     if (r, controller) in cell_times]
            if not times:
                continue
            avg = float(np.mean(times))
            col = avg * n_full * len(RUNG_ORDER)
            total += col
            print(f"    {controller:>10}: {avg:.3f} s/flight x {n_full} x "
                  f"{len(RUNG_ORDER)} rungs = {col/3600:.2f} h")
        print("  " + "-" * 60)
        print(f"    TOTAL (all three controllers, six rungs): {total/3600:.2f} h")
        print("=" * 70)


if __name__ == "__main__":
    main()