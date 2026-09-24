"""
verify_filter.py
Correctness suite for the deployed safety filter. On random certified states
across the operating envelope it checks A barrier enforcement at the filter
output, A' zero fallback activations, B pass-through of already-safe commands,
C active correction of a command that would violate the keep-out condition,
and D determinism. Prints a PASS/FAIL verdict per check.
"""

import numpy as np
from cbf_filter import CBFFilter, U_MAX, R_KOZ, v_cap, HOLD, ALPHA1_DEFAULT

TOL = 1e-4


def certified(r, v, alpha1=ALPHA1_DEFAULT):
    psi0 = r.dot(r) - R_KOZ ** 2
    psi1 = 2.0 * r.dot(v) + alpha1 * psi0
    return psi0 >= 0.0 and psi1 >= 0.0


def sample_certified(rng):
    """A random certified state across the operating envelope."""
    while True:
        d = rng.uniform(R_KOZ, 100.0)
        dirn = rng.normal(size=3); dirn /= np.linalg.norm(dirn)
        r = d * dirn
        cap = v_cap(d)
        v = rng.normal(size=3); v = v / np.linalg.norm(v) * rng.uniform(0.0, cap)
        if certified(r, v):
            th = rng.uniform(0, 2 * np.pi)
            a = rng.uniform(3.0e7, 4.5e7); e = rng.uniform(0.05, 0.30)
            return np.array([*r, *v, th]), a, e


def check_A_barriers(n=5000):
    print(f"CHECK A/A'  barrier enforcement + zero fallbacks ({n} certified states)")
    f = CBFFilter(); rng = np.random.default_rng(2)
    worst_koz = 0.0; worst_vcap = 0.0; n_fb = 0; solves = []
    for _ in range(n):
        state, a, e = sample_certified(rng)
        u_net = rng.uniform(-U_MAX, U_MAX, size=3)
        out = f.filter(state, u_net, a, e)
        u = out['u']; solves.append(out['solve_us'])
        if out['fallback'] != 0:
            n_fb += 1
        b_koz, rhs_koz, b_v, rhs_v, _, _ = f._rows(state)
        worst_koz = min(worst_koz, b_koz.dot(u) + rhs_koz)
        worst_vcap = min(worst_vcap, b_v.dot(u) + rhs_v)
    solves = np.array(solves)
    okA = worst_koz > -TOL and worst_vcap > -TOL
    okAp = n_fb == 0
    print(f"  worst keep-out condition at output : {worst_koz:.3e}")
    print(f"  worst vel-cap  condition at output : {worst_vcap:.3e}")
    print(f"  fallback activations               : {n_fb}")
    print(f"  solve time  mean {solves.mean():.1f} us   p99 {np.percentile(solves,99):.1f} us   max {solves.max():.1f} us")
    print(f"  A  barriers enforced : {'PASS' if okA else 'FAIL'}")
    print(f"  A' zero fallbacks    : {'PASS' if okAp else 'FAIL'}")
    return okA and okAp


def check_B_passthrough(n=1000):
    print(f"\nCHECK B  pass-through on safe commands ({n} states, far from constraints)")
    f = CBFFilter(); rng = np.random.default_rng(7); corrs = []
    for _ in range(n):
        d = rng.uniform(40, 100); dirn = rng.normal(size=3); dirn /= np.linalg.norm(dirn)
        r = d * dirn; v = rng.uniform(-0.1, 0.1, size=3)
        th = rng.uniform(0, 2 * np.pi); a = rng.uniform(3e7, 4.5e7); e = rng.uniform(0.05, 0.3)
        err = r - HOLD
        u_net = np.clip(-0.01 * err / np.linalg.norm(err), -U_MAX, U_MAX)
        out = f.filter(np.array([*r, *v, th]), u_net, a, e)
        corrs.append(out['correction'])
    corrs = np.array(corrs)
    ok = corrs.mean() < 0.01
    print(f"  mean correction : {corrs.mean():.5f}   max : {corrs.max():.5f}")
    print(f"  B  passes safe commands through : {'PASS' if ok else 'FAIL'}")
    return ok


def check_C_protection():
    print("\nCHECK C  active keep-out protection where the raw command would violate")
    f = CBFFilter()
    # A certified state right at the sphere where the network's command, applied
    # raw, would violate the keep-out condition, so the filter must correct it.
    # A far or outward-moving state would pass the command through trivially and
    # prove nothing; this state is chosen so the barrier genuinely binds.
    r = np.array([0.74, 3.28, -3.83]); v = np.array([0.0, -0.01, -0.01])
    assert certified(r, v), "test state must be certified"
    state = np.array([*r, *v, 0.5]); a = 3.5e7; e = 0.2
    u_net = -U_MAX * r / np.linalg.norm(r)          # net drives hard at the target
    f._a, f._e = a, e
    b_koz, rhs_koz, _, _, _, _ = f._rows(state)
    val_raw = b_koz.dot(u_net) + rhs_koz            # what the raw command would give
    out = f.filter(state, u_net, a, e)
    val_filtered = b_koz.dot(out['u']) + rhs_koz
    ok = (val_raw < -1e-3                            # the raw command really violates
          and val_filtered > -TOL                   # the filter fixes it
          and out['correction'] > 1e-3              # the filter actually acted
          and out['fallback'] == 0)
    print(f"  raw command keep-out value : {val_raw:+.4f}  (< 0, would violate)")
    print(f"  filtered keep-out value    : {val_filtered:+.4f}  (>= 0, enforced)")
    print(f"  correction applied         : {out['correction']:.4f}")
    print(f"  C  filter corrects a violating command : {'PASS' if ok else 'FAIL'}")
    return ok


def check_D_determinism():
    print("\nCHECK D  determinism")
    f = CBFFilter()
    s = np.array([20., 5, 3, -0.2, 0.1, 0, 1.0]); un = np.array([0.03, -0.02, 0.01])
    o1 = f.filter(s, un, 3.5e7, 0.2); o2 = f.filter(s, un, 3.5e7, 0.2)
    diff = np.linalg.norm(o1['u'] - o2['u'])
    ok = diff < 1e-6
    print(f"  |u1 - u2| = {diff:.2e}")
    print(f"  D  deterministic : {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 70)
    print("CBF FILTER CORRECTNESS SUITE  (Section 9.4)")
    print("=" * 70)
    results = [check_A_barriers(), check_B_passthrough(),
               check_C_protection(), check_D_determinism()]
    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL CHECKS PASS - filter is correct on the certified set'
                       if all(results) else 'FAILURE - do not deploy'}")
    print("=" * 70)


if __name__ == "__main__":
    main()