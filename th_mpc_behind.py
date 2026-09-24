"""
th_mpc_expert_behind.py
Closed-loop Tschauner-Hempel NMPC expert for a V-bar proximity approach to a
target on an elliptical orbit. Solves the OCP at every control instant,
applies the first command, propagates one step, repeats.
"""

import time
import numpy as np
import casadi as ca

# ============================================================================
# Configuration
# ============================================================================
MODE = 'single'          # 'single' or 'randomized'

USE_KEEPOUT  = True       # spherical keep-out ||r|| >= R_KOZ
USE_VCAP     = True       # tapered velocity cap ||v|| <= v_cap(d)
USE_CORRIDOR = False      # approach corridor (not used)

N_RANDOM = 1000

# physical and orbital constants
MU = 3.986004418e14       # Earth gravitational parameter [m^3/s^2]

# nominal orbit for the single run, sampling ranges for the randomized study
A_SMA_NOM = 3.5e7         # semi-major axis [m]
ECC_NOM   = 0.30          # eccentricity
A_RANGE   = (3.0e7, 4.5e7)
E_RANGE   = (0.05, 0.30)

# scenario geometry, V-bar approach along +x
R_KOZ   = 5.0             # keep-out sphere radius [m]
D_HOLD  = -8.0            # hold point on -x, behind the target [m]
HOLD    = np.array([D_HOLD, 0.0, 0.0])
CONE_HALF_ANGLE = np.deg2rad(30.0)       # corridor half-angle (unused)
CONE_AXIS = np.array([-1.0, 0.0, 0.0])   # corridor axis (unused)

# initial-condition cylinder, behind the target on -x
IC_X_RANGE = (-110.0, -90.0)
IC_RHO_MAX = 90.0 * np.tan(np.deg2rad(30.0))   # 51.96 m disc radius
IC_V_MAX   = 0.2

# tapered velocity cap, constant V_FAR far out, smooth taper to V_NEAR at hold
V_FAR    = 2.0            # far-field speed limit [m/s]
V_NEAR   = 0.2            # near-field floor [m/s]
D_TAPER  = 30.0           # separation where the cap starts tightening [m]
VCAP_BETA = 12.0          # softplus sharpness of the smooth clamp

# MPC tuning
N_P   = 80               # prediction horizon (steps)
DT    = 0.5              # control interval [s]
T_HORIZON = N_P * DT
U_MAX = 0.082            # per-axis thrust limit [m/s^2]

SIM_STEPS = 400          # closed-loop steps

# cost weights
Q_POS, Q_VEL, R_CTRL = 1.0, 1.0, 1.0
QN_SCALE = 50.0

# soft-constraint penalties, keep-out firm, velocity cap unpenalised
KOZ_ZL, KOZ_ZU = 1e4, 1e4
KOZ_ZL2, KOZ_ZU2 = 1e3, 1e3
VCAP_ZL, VCAP_ZU = 0.0, 0.0
VCAP_ZL2, VCAP_ZU2 = 0.0, 0.0

# arrival tolerances at the hold point
TOL_POS = 0.5            # [m]
TOL_VEL = 0.05           # [m/s]


# ============================================================================
# Dynamics, identical equations in CasADi symbolic and numpy form.
# a and e are parameters so one solver covers the sampled orbit range.
# ============================================================================
def th_rhs_casadi(state, u, a_sym, e_sym):
    x, y, z, vx, vy, vz, th = ca.vertsplit(state)
    c = ca.cos(th)
    den = 1.0 + e_sym * c
    thd = (1.0 + e_sym * c) ** 2 / (1.0 - e_sym ** 2) ** 1.5 * ca.sqrt(MU / a_sym ** 3)
    thdd = (-2.0 * e_sym * (1.0 + e_sym * c) * ca.sin(th) / (1.0 - e_sym ** 2) ** 1.5
            * ca.sqrt(MU / a_sym ** 3) * thd)
    # x = V-bar (along-track)
    ax = -thdd * y + (e_sym * c) / den * thd ** 2 * x - 2.0 * thd * vy + u[0]
    # y = R-bar (radial, outward)
    ay = thdd * x + (3.0 + e_sym * c) / den * thd ** 2 * y + 2.0 * thd * vx + u[1]
    # z = H-bar (cross-track, decoupled)
    az = -(1.0 / den) * thd ** 2 * z + u[2]
    return ca.vertcat(vx, vy, vz, ax, ay, az, thd)


def th_rhs_numpy(state, u, a, e):
    x, y, z, vx, vy, vz, th = state
    c = np.cos(th)
    den = 1.0 + e * c
    thd = (1.0 + e * c) ** 2 / (1.0 - e ** 2) ** 1.5 * np.sqrt(MU / a ** 3)
    thdd = (-2.0 * e * (1.0 + e * c) * np.sin(th) / (1.0 - e ** 2) ** 1.5
            * np.sqrt(MU / a ** 3) * thd)
    # x = V-bar (along-track)
    ax = -thdd * y + (e * c) / den * thd ** 2 * x - 2.0 * thd * vy + u[0]
    # y = R-bar (radial, outward)
    ay = thdd * x + (3.0 + e * c) / den * thd ** 2 * y + 2.0 * thd * vx + u[1]
    # z = H-bar (cross-track, decoupled)
    az = -(1.0 / den) * thd ** 2 * z + u[2]
    return np.array([vx, vy, vz, ax, ay, az, thd])


def rk4_step(state, u, dt, a, e):
    k1 = th_rhs_numpy(state, u, a, e)
    k2 = th_rhs_numpy(state + 0.5 * dt * k1, u, a, e)
    k3 = th_rhs_numpy(state + 0.5 * dt * k2, u, a, e)
    k4 = th_rhs_numpy(state + dt * k3, u, a, e)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================================
# Smooth velocity cap, softplus-based clamps so the taper has no kinks
# ============================================================================
def _softplus_np(x, beta):
    return (1.0 / beta) * np.log1p(np.exp(beta * x))


def _smooth_clamp01_np(x, beta):
    lo = _softplus_np(x, beta)                 # smooth max(x, 0)
    return 1.0 - _softplus_np(1.0 - lo, beta)  # smooth min(lo, 1)


def v_cap_np(d):
    """Distance-dependent speed limit, numpy form, matches the OCP expression."""
    frac = (d - D_HOLD) / (D_TAPER - D_HOLD)
    fr = _smooth_clamp01_np(frac, VCAP_BETA)
    return V_NEAR + (V_FAR - V_NEAR) * fr


def _softplus_ca(x, beta):
    return (1.0 / beta) * ca.log(1.0 + ca.exp(beta * x))


def _smooth_clamp01_ca(x, beta):
    lo = _softplus_ca(x, beta)                 # smooth max(x, 0)
    return 1.0 - _softplus_ca(1.0 - lo, beta)  # smooth min(lo, 1)


def v_cap_casadi(d):
    """Distance-dependent speed limit as a smooth CasADi expression."""
    frac = (d - D_HOLD) / (D_TAPER - D_HOLD)
    fr = _smooth_clamp01_ca(frac, VCAP_BETA)
    return V_NEAR + (V_FAR - V_NEAR) * fr


# ============================================================================
# Build the acados OCP solver, a and e enter as online parameters
# ============================================================================
def build_solver():
    from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

    nx, nu = 7, 3
    state = ca.SX.sym('state', nx)
    u = ca.SX.sym('u', nu)
    a_sym = ca.SX.sym('a_sym')
    e_sym = ca.SX.sym('e_sym')

    model = AcadosModel()
    model.name = 'th_expert'
    model.x = state
    model.u = u
    model.p = ca.vertcat(a_sym, e_sym)
    model.f_expl_expr = th_rhs_casadi(state, u, a_sym, e_sym)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N_P
    ocp.solver_options.tf = T_HORIZON
    ocp.parameter_values = np.array([A_SMA_NOM, ECC_NOM])

    # least-squares cost tracking the hold point
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    W = np.diag([Q_POS] * 3 + [Q_VEL] * 3 + [R_CTRL] * 3)
    W_e = np.diag([Q_POS] * 3 + [Q_VEL] * 3) * QN_SCALE
    ocp.cost.W, ocp.cost.W_e = W, W_e

    ny, ny_e = 9, 6
    Vx = np.zeros((ny, nx)); Vx[:6, :6] = np.eye(6)
    Vu = np.zeros((ny, nu)); Vu[6:, :] = np.eye(nu)
    ocp.cost.Vx, ocp.cost.Vu = Vx, Vu
    Vx_e = np.zeros((ny_e, nx)); Vx_e[:6, :6] = np.eye(6)
    ocp.cost.Vx_e = Vx_e

    yref = np.zeros(ny); yref[:3] = HOLD
    yref_e = np.zeros(ny_e); yref_e[:3] = HOLD
    ocp.cost.yref, ocp.cost.yref_e = yref, yref_e

    ocp.constraints.x0 = np.zeros(nx)

    # thrust box
    ocp.constraints.idxbu = np.array([0, 1, 2])
    ocp.constraints.lbu = np.array([-U_MAX] * 3)
    ocp.constraints.ubu = np.array([U_MAX] * 3)

    # nonlinear path constraints in a fixed order, keep-out then velocity cap,
    # so the per-constraint penalty split is unambiguous
    h_list = []
    zl_vec, zu_vec, Zl_vec, Zu_vec = [], [], [], []

    if USE_KEEPOUT:
        # ||r||^2 - R_KOZ^2 >= 0
        h_list.append(state[0] ** 2 + state[1] ** 2 + state[2] ** 2 - R_KOZ ** 2)
        zl_vec.append(KOZ_ZL);  zu_vec.append(KOZ_ZU)
        Zl_vec.append(KOZ_ZL2); Zu_vec.append(KOZ_ZU2)

    if USE_VCAP:
        # v_cap(d)^2 - ||v||^2 >= 0
        r = state[0:3]
        v = state[3:6]
        d = ca.sqrt(ca.dot(r, r) + 1e-9)
        vc = v_cap_casadi(d)
        h_list.append(vc ** 2 - ca.dot(v, v))
        zl_vec.append(VCAP_ZL);  zu_vec.append(VCAP_ZU)
        Zl_vec.append(VCAP_ZL2); Zu_vec.append(VCAP_ZU2)

    if USE_CORRIDOR:
        r = state[0:3]; axis = ca.DM(CONE_AXIS)
        rnorm = ca.sqrt(ca.dot(r, r) + 1e-9)
        h_list.append(ca.dot(r, axis) - rnorm * np.cos(CONE_HALF_ANGLE))
        zl_vec.append(VCAP_ZL);  zu_vec.append(VCAP_ZU)
        Zl_vec.append(VCAP_ZL2); Zu_vec.append(VCAP_ZU2)

    if h_list:
        model.con_h_expr = ca.vertcat(*h_list)
        nh = len(h_list)
        ocp.constraints.lh = np.zeros(nh)
        ocp.constraints.uh = 1e9 * np.ones(nh)
        # soft constraints with per-constraint penalties
        ocp.constraints.idxsh = np.arange(nh)
        ocp.cost.zl = np.array(zl_vec)
        ocp.cost.zu = np.array(zu_vec)
        ocp.cost.Zl = np.array(Zl_vec)
        ocp.cost.Zu = np.array(Zu_vec)

    # solver options, full SQP iterated to convergence
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1

    return AcadosOcpSolver(ocp, json_file='th_expert.json')


# ============================================================================
# Frame verification, free drift on a circular orbit with no thrust
# ============================================================================
def verify_frame(a=3.5e7, verbose=True):
    n = np.sqrt(MU / a ** 3)
    T = 2.0 * np.pi / n
    dt = 2.0
    u0 = np.zeros(3)
    out = {}
    for key, s0 in [('x', np.array([100., 0., 0., 0., 0., 0., 0.])),
                    ('y', np.array([0., 100., 0., 0., 0., 0., 0.]))]:
        s = s0.copy()
        for _ in range(int(T / dt)):
            s = rk4_step(s, u0, dt, a, 0.0)
        out[key] = s[:3].copy()
    if verbose:
        print("  frame check (circular orbit, free drift, one full orbit):")
        print(f"    +100 m on x (V-bar): -> ({out['x'][0]:8.2f}, {out['x'][1]:9.2f}, "
              f"{out['x'][2]:5.2f})   expect ( 100.00,     0.00): passively stable")
        print(f"    +100 m on y (R-bar): -> ({out['y'][0]:8.2f}, {out['y'][1]:9.2f}, "
              f"{out['y'][2]:5.2f})   expect (-3770.0,   100.00): along-track drift")
        ok = (abs(out['x'][0] - 100.0) < 1.0 and abs(out['x'][1]) < 1.0
              and out['y'][0] < -3000.0 and abs(out['y'][1] - 100.0) < 1.0)
        print(f"    verdict: {'PASS - x is V-bar' if ok else 'FAIL - axes are wrong'}")
    return out


# ============================================================================
# Closed-loop simulation from one initial condition
# ============================================================================
def run_closed_loop(solver, x0_rel, theta0, a, e, sim_steps=SIM_STEPS,
                    stop_on_arrival=True, verbose=True):
    state = np.concatenate([x0_rel, [theta0]])
    X_hist = [state.copy()]
    U_hist = []
    solve_times = []
    solve_fail = 0
    p_val = np.array([a, e])

    arrival_step = None

    for k in range(sim_steps):
        # push orbital parameters to every stage
        for j in range(N_P + 1):
            solver.set(j, "p", p_val)
        # pin the initial predicted state to the current true state
        solver.set(0, "lbx", state)
        solver.set(0, "ubx", state)

        # acados retains the previous solution internally, which warm starts

        t0 = time.perf_counter()
        status = solver.solve()
        solve_times.append((time.perf_counter() - t0) * 1e3)   # ms
        if status != 0:
            solve_fail += 1

        u0 = solver.get(0, "u")
        U_hist.append(u0.copy())

        # propagate one step
        state = rk4_step(state, u0, DT, a, e)
        X_hist.append(state.copy())

        # arrival check
        pos_err = np.linalg.norm(state[:3] - HOLD)
        vel = np.linalg.norm(state[3:6])
        if arrival_step is None and pos_err < TOL_POS and vel < TOL_VEL:
            arrival_step = k + 1
            if stop_on_arrival:
                break

    X = np.array(X_hist)
    U = np.array(U_hist)
    solve_times = np.array(solve_times)

    # figures of merit
    err_to_hold = X[:, :3] - HOLD
    final_pos_err = np.linalg.norm(err_to_hold[-1])
    final_vel = np.linalg.norm(X[-1, 3:6])

    dist_target = np.linalg.norm(X[:, :3], axis=1)
    min_dist_target = np.min(dist_target)
    koz_ok = min_dist_target >= R_KOZ - 1e-6
    koz_margin = min_dist_target - R_KOZ

    speed = np.linalg.norm(X[:, 3:6], axis=1)
    peak_speed = np.max(speed)
    caps = np.array([v_cap_np(d) for d in dist_target])
    vcap_viol = np.max(np.maximum(speed - caps, 0.0))

    fuel = np.sum(np.linalg.norm(U, axis=1)) * DT
    effort = np.sum(np.sum(U ** 2, axis=1)) * DT

    settling_time = arrival_step * DT if arrival_step is not None else np.nan
    converged = (final_pos_err < TOL_POS) and (final_vel < TOL_VEL)

    res = {
        'X': X, 'U': U,
        'solve_fail': solve_fail,
        'final_pos_err': final_pos_err,
        'final_vel': final_vel,
        'min_dist_target': min_dist_target,
        'koz_margin': koz_margin,
        'koz_ok': koz_ok,
        'peak_speed': peak_speed,
        'vcap_viol': vcap_viol,
        'fuel': fuel,
        'effort': effort,
        'settling_time': settling_time,
        'converged': converged,
        't_solve_mean': solve_times.mean(),
        't_solve_max': solve_times.max(),
        't_solve_std': solve_times.std(),
        'n_steps': len(U),
        'a': a, 'e': e, 'theta0': theta0,
    }

    if verbose:
        print(f"  orbit:  a={a:.3e} m  e={e:.3f}  theta0={theta0:.3f} rad")
        print(f"  closed-loop steps:          {len(U)}")
        print(f"  solver failures:            {solve_fail}")
        print(f"  converged (<{TOL_POS} m, <{TOL_VEL} m/s): {converged}")
        print(f"  settling time:              "
              f"{settling_time:.1f} s" if not np.isnan(settling_time) else
              "  settling time:              not reached")
        print(f"  final dist to hold point:   {final_pos_err:.4f} m")
        print(f"  final speed:                {final_vel:.5f} m/s")
        print(f"  min dist to target:         {min_dist_target:.3f} m  "
              f"(keep-out {R_KOZ} m: {'OK' if koz_ok else 'VIOLATED'}, "
              f"margin {koz_margin:+.3f} m)")
        print(f"  peak speed:                 {peak_speed:.3f} m/s")
        print(f"  velocity-cap violation:     {vcap_viol:.4f} m/s "
              f"({'clean' if vcap_viol < 1e-3 else 'violated'})")
        print(f"  fuel (integral |u| dt):     {fuel:.3f} m/s")
        print(f"  control effort (|u|^2 dt):  {effort:.5f}")
        print(f"  solve time [ms]: mean {solve_times.mean():.2f}  "
              f"max {solve_times.max():.2f}  std {solve_times.std():.2f}")

    return res


# ============================================================================
# Cylinder initial-condition sampler
# ============================================================================
def sample_cylinder_ic(rng):
    x = rng.uniform(*IC_X_RANGE)
    rho = IC_RHO_MAX * np.sqrt(rng.uniform(0.0, 1.0))   # area-uniform disc
    phi = rng.uniform(0.0, 2 * np.pi)
    y = rho * np.cos(phi)
    z = rho * np.sin(phi)
    r0 = np.array([x, y, z])
    v0 = rng.uniform(-IC_V_MAX, IC_V_MAX, size=3)
    return np.concatenate([r0, v0])


# ============================================================================
# Time-history panels for one run
# ============================================================================
def plot_run(res, fname='th_expert_unpenalised_behind.png'):
    import matplotlib.pyplot as plt
    X, U = res['X'], res['U']
    t = np.arange(X.shape[0]) * DT
    tu = np.arange(U.shape[0]) * DT

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    axs[0, 0].plot(t, X[:, 0], label='X (V-bar)')
    axs[0, 0].plot(t, X[:, 1], label='Y (R-bar)')
    axs[0, 0].plot(t, X[:, 2], label='Z (H-bar)')
    axs[0, 0].axhline(HOLD[0], ls='--', c='gray', lw=0.8)
    axs[0, 0].set_title('Relative position [m]')
    axs[0, 0].legend(); axs[0, 0].grid(True); axs[0, 0].set_xlabel('T [s]')

    axs[0, 1].plot(t, X[:, 3], label='Vx')
    axs[0, 1].plot(t, X[:, 4], label='Vy')
    axs[0, 1].plot(t, X[:, 5], label='Vz')
    axs[0, 1].set_title('Relative velocity [m/s]')
    axs[0, 1].legend(); axs[0, 1].grid(True); axs[0, 1].set_xlabel('T [s]')

    # speed against the tapered cap
    dist = np.linalg.norm(X[:, :3], axis=1)
    speed = np.linalg.norm(X[:, 3:6], axis=1)
    caps = np.array([v_cap_np(d) for d in dist])
    axs[0, 2].plot(t, speed, label='||V||')
    axs[0, 2].plot(t, caps, 'r--', lw=1, label='V_cap(d)')
    axs[0, 2].set_title('Speed vs tapered cap [m/s]')
    axs[0, 2].legend(); axs[0, 2].grid(True); axs[0, 2].set_xlabel('T [s]')

    axs[1, 0].step(tu, U[:, 0], label='Ux')
    axs[1, 0].step(tu, U[:, 1], label='Uy')
    axs[1, 0].step(tu, U[:, 2], label='Uz')
    axs[1, 0].axhline(U_MAX, ls=':', c='r', lw=0.8)
    axs[1, 0].axhline(-U_MAX, ls=':', c='r', lw=0.8)
    axs[1, 0].set_title('Control [m/s^2]')
    axs[1, 0].legend(); axs[1, 0].grid(True); axs[1, 0].set_xlabel('T [s]')

    # distance to target against the keep-out radius
    axs[1, 1].plot(t, dist, label='||R||')
    axs[1, 1].axhline(R_KOZ, ls='--', c='r', lw=1, label='Keep-out')
    axs[1, 1].set_title('Distance to target [m]')
    axs[1, 1].legend(); axs[1, 1].grid(True); axs[1, 1].set_xlabel('T [s]')

    # trajectory in the x-y plane
    axs[1, 2].plot(X[:, 0], X[:, 1])
    th = np.linspace(0, 2 * np.pi, 100)
    axs[1, 2].plot(R_KOZ * np.cos(th), R_KOZ * np.sin(th), 'r--', lw=1, label='Keep-out')
    axs[1, 2].scatter([HOLD[0]], [HOLD[1]], c='g', marker='*', s=150, label='Hold point')
    axs[1, 2].scatter([0], [0], c='k', marker='x', s=60, label='Target')
    axs[1, 2].set_aspect('equal')
    axs[1, 2].set_title('X-Y plane (V-bar / R-bar)')
    axs[1, 2].legend(fontsize=8); axs[1, 2].grid(True)
    axs[1, 2].set_xlabel('X [m]'); axs[1, 2].set_ylabel('Y [m]')

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    print(f"  saved figure -> {fname}")
    plt.show()


# ============================================================================
# 3D scene, reference frame, starting cylinder, keep-out sphere, trajectory
# ============================================================================
def plot_scene_3d(res=None, fname='th_expert_scene3d.png'):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # keep-out sphere
    us = np.linspace(0, 2 * np.pi, 40)
    vs = np.linspace(0, np.pi, 20)
    sx = R_KOZ * np.outer(np.cos(us), np.sin(vs))
    sy = R_KOZ * np.outer(np.sin(us), np.sin(vs))
    sz = R_KOZ * np.outer(np.ones_like(us), np.cos(vs))
    ax.plot_surface(sx, sy, sz, color='red', alpha=0.15, linewidth=0)
    ax.plot_wireframe(sx, sy, sz, color='red', lw=0.3, alpha=0.4)

    # starting cylinder
    x_lo, x_hi = IC_X_RANGE
    phi = np.linspace(0, 2 * np.pi, 60)
    for xf in (x_lo, x_hi):
        yc = IC_RHO_MAX * np.cos(phi)
        zc = IC_RHO_MAX * np.sin(phi)
        ax.plot(np.full_like(phi, xf), yc, zc, color='tab:blue', lw=1.0, alpha=0.7)
    for pth in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        yy = IC_RHO_MAX * np.cos(pth)
        zz = IC_RHO_MAX * np.sin(pth)
        ax.plot([x_lo, x_hi], [yy, yy], [zz, zz], color='tab:blue', lw=0.4, alpha=0.35)
    rr = np.linspace(0, IC_RHO_MAX, 8)
    RR, PP = np.meshgrid(rr, phi)
    ax.plot_surface(np.full_like(RR, x_lo), RR * np.cos(PP), RR * np.sin(PP),
                    color='tab:blue', alpha=0.06, linewidth=0)

    # reference-frame axes drawn from the target
    L = 0.7 * abs(x_hi)
    ax.quiver(0, 0, 0, L, 0, 0, color='k', lw=1.5, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, IC_RHO_MAX * 1.3, 0, color='k', lw=1.2, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 0, IC_RHO_MAX * 1.3, color='k', lw=1.2, arrow_length_ratio=0.08)
    ax.text(L * 1.02, 0, 0, 'X  V-bar', fontsize=11)
    ax.text(0, IC_RHO_MAX * 1.35, 0, 'Y  R-bar', fontsize=11)
    ax.text(0, 0, IC_RHO_MAX * 1.35, 'Z  H-bar', fontsize=11)

    # target and hold point
    ax.scatter([0], [0], [0], c='k', marker='x', s=80, label='Target')
    ax.scatter([HOLD[0]], [HOLD[1]], [HOLD[2]], c='g', marker='*', s=200, label='Hold point')

    # flown trajectory
    if res is not None:
        X = res['X']
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color='tab:orange', lw=1.6, label='Trajectory')
        ax.scatter([X[0, 0]], [X[0, 1]], [X[0, 2]], c='tab:blue', marker='o', s=50,
                   label='Start')

    ax.set_xlabel('X  V-bar [m]', labelpad=12)
    ax.set_ylabel('Y  R-bar [m]', labelpad=12)
    ax.set_zlabel('Z  H-bar [m]', labelpad=12)
    ax.set_title('Reference frame, starting cylinder and keep-out zone', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)

    try:
        ax.set_box_aspect((abs(x_hi), 2 * IC_RHO_MAX, 2 * IC_RHO_MAX))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-60)

    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    print(f"  saved figure -> {fname}")
    plt.show()


# ============================================================================
# Randomized study over sampled initial conditions and orbits
# ============================================================================
def run_randomized(solver, n=N_RANDOM, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    print(f"  running {n} randomized initial conditions...\n")
    for i in range(n):
        x0 = sample_cylinder_ic(rng)
        theta0 = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(*A_RANGE)
        e = rng.uniform(*E_RANGE)
        res = run_closed_loop(solver, x0[:6], theta0, a, e,
                              stop_on_arrival=True, verbose=False)
        res['x0'] = x0
        results.append(res)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n} done")

    fpe = np.array([r['final_pos_err'] for r in results])
    fv = np.array([r['final_vel'] for r in results])
    koz = np.array([r['koz_ok'] for r in results])
    kozm = np.array([r['koz_margin'] for r in results])
    fail = np.array([r['solve_fail'] for r in results])
    fuel = np.array([r['fuel'] for r in results])
    effort = np.array([r['effort'] for r in results])
    vcv = np.array([r['vcap_viol'] for r in results])
    settle = np.array([r['settling_time'] for r in results])
    tmean = np.array([r['t_solve_mean'] for r in results])
    tmax = np.array([r['t_solve_max'] for r in results])
    conv = np.array([r['converged'] for r in results])

    print("\n" + "=" * 60)
    print("  RANDOMIZED FIGURES-OF-MERIT SUMMARY")
    print("=" * 60)
    print(f"  runs:                          {n}")
    print(f"  converged (<{TOL_POS} m, <{TOL_VEL} m/s):  "
          f"{conv.sum()}/{n} ({100 * conv.mean():.1f}%)")
    print(f"  keep-out respected:            {koz.sum()}/{n} ({100 * koz.mean():.1f}%)")
    print(f"  runs with any solve fail:      {(fail > 0).sum()}/{n}")
    print("  " + "-" * 56)
    print(f"  final pos err [m]:   mean {fpe.mean():.3f}   "
          f"median {np.median(fpe):.3f}   max {fpe.max():.3f}")
    print(f"  final speed [m/s]:   mean {fv.mean():.4f}   max {fv.max():.4f}")
    print(f"  keep-out margin [m]: mean {kozm.mean():.3f}   min {kozm.min():.3f}")
    print(f"  vel-cap viol [m/s]:  mean {vcv.mean():.4f}   max {vcv.max():.4f}")
    print(f"  fuel [m/s]:          mean {fuel.mean():.3f}   "
          f"min {fuel.min():.3f}   max {fuel.max():.3f}")
    print(f"  effort |u|^2 dt:     mean {effort.mean():.4f}   max {effort.max():.4f}")
    valid_settle = settle[~np.isnan(settle)]
    if valid_settle.size:
        print(f"  settling time [s]:   mean {valid_settle.mean():.1f}   "
              f"max {valid_settle.max():.1f}")
    print("  " + "-" * 56)
    print(f"  solve time [ms]:     mean-of-means {tmean.mean():.2f}   "
          f"worst-case max {tmax.max():.2f}")
    print("=" * 60)

    return results


# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(f"TH-MPC EXPERT (x = V-bar)  |  keepout={USE_KEEPOUT} vcap={USE_VCAP} "
          f"corridor={USE_CORRIDOR} mode={MODE}")
    print(f"u_max={U_MAX}  N_p={N_P} (no control horizon, free commands)  "
          f"dt={DT}  sim_steps={SIM_STEPS}")
    print(f"tapered vcap (SMOOTH): {V_FAR} -> {V_NEAR} m/s below {D_TAPER} m  "
          f"(beta={VCAP_BETA})")
    print(f"penalties: keep-out firm (zl={KOZ_ZL:.0e})  vcap gentle (zl={VCAP_ZL:.0e})")
    print("=" * 70)

    verify_frame()
    print("=" * 70)

    solver = build_solver()

    if MODE == 'single':
        # worst-case corner start, near face, full off-axis radius split 45 deg
        # between y and z, inward velocity on every axis, perigee start
        phi = np.deg2rad(45.0)
        x0 = np.array([-90.0,
                       IC_RHO_MAX * np.cos(phi),
                       IC_RHO_MAX * np.sin(phi),
                       +IC_V_MAX,
                       -IC_V_MAX * np.cos(phi),
                       -IC_V_MAX * np.sin(phi)])
        res = run_closed_loop(solver, x0[:6], theta0=0.0,
                              a=A_SMA_NOM, e=ECC_NOM, stop_on_arrival=False)
        plot_run(res)
        plot_scene_3d(res)

    elif MODE == 'randomized':
        run_randomized(solver)