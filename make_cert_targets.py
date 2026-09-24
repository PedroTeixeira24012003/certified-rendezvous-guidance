"""
make_cert_targets.py
Precompute the certificate gradient vectors for the four-term loss. Both
certificate rates are linear in the command, and the network and the expert
are evaluated at the same dataset state, so the drift terms cancel in the
difference and only the per-row u-gradients g_V and g_H are needed. Computes
them fully vectorized for every dataset row and stores them aligned with the
rows in cert_targets_behind.npz.
"""

import numpy as np
import bc_data
from cbf_filter import _P_CLF, HOLD

OUT_FILE = 'cert_targets_behind.npz'


def main():
    print("=" * 62)
    print("PRECOMPUTE CERTIFICATE GRADIENTS (four-term loss, Ch 21)")
    print("=" * 62)

    data = np.load(bc_data.DATASET_FILE, mmap_mode='r')
    X = np.asarray(data['X'], dtype=np.float64)          # (N, 10) RAW features
    N = X.shape[0]
    print(f"  dataset rows: {N:,}")

    r = X[:, 0:3]                                        # position [m]
    v = X[:, 3:6]                                        # velocity [m/s]

    # g_V = 2 (P xi)[3:],  xi = [r - HOLD, v]
    xi = np.concatenate([r - HOLD[None, :], v], axis=1)  # (N, 6)
    Pxi = xi @ _P_CLF.T                                  # (N, 6)
    g_V = 2.0 * Pxi[:, 3:]                               # (N, 3)

    # g_H = 2 r  (keep-out HOCBF, the braking law near the sphere)
    g_H = 2.0 * r                                        # (N, 3)

    # sanity: magnitudes
    nv = np.linalg.norm(g_V, axis=1)
    nh = np.linalg.norm(g_H, axis=1)
    print(f"  |g_V| range: [{nv.min():.3e}, {nv.max():.3e}]  mean {nv.mean():.3e}")
    print(f"  |g_H| range: [{nh.min():.3e}, {nh.max():.3e}]  mean {nh.mean():.3e}")
    assert np.all(np.isfinite(g_V)) and np.all(np.isfinite(g_H))

    np.savez(OUT_FILE,
             g_V=g_V.astype(np.float32),
             g_H=g_H.astype(np.float32))
    print(f"  saved -> {OUT_FILE}")

    # self-check: recompute one row through the filter's own machinery
    import cbf_filter
    i = 12345
    state = np.array([*X[i, 0:3], *X[i, 3:6],
                      np.arctan2(X[i, 6], X[i, 7])])
    filt = cbf_filter.CBFFilter()
    filt._a, filt._e = X[i, 8], X[i, 9]
    b_koz, _, _, _, grad_u_V, _ = filt._rows(state)
    assert np.allclose(grad_u_V, g_V[i], atol=1e-6), "g_V mismatch vs filter"
    assert np.allclose(b_koz, g_H[i], atol=1e-6), "g_H mismatch vs filter"
    print("  cross-check vs cbf_filter._rows : OK")
    print("  DONE")


if __name__ == "__main__":
    main()