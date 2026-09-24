"""
bc_data_aug.py
Augmented data module for the four-term loss. Wraps bc_data and adds, per
row, the precomputed targets the extra loss terms need, the certificate
gradients g_V and g_H from make_cert_targets.py and the TaSIL probe dlt plus
expert response dexp from make_tasil_targets.py (the probe never touches the
sin/cos theta, a, e slots). Split, normalization, label scaling and subset
are bc_data's, imported, never redefined. Rows without a TaSIL target carry
dexp = NaN; the training loss masks those rows out of the TaSIL term only.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import bc_data

CERT_FILE = 'cert_targets_behind.npz'
TASIL_FULL = 'tasil_targets_behind.npz'
TASIL_SUB = 'tasil_targets_behind_subset.npz'


def _load_tasil_map(n_total):
    """Return (delta_norm10, d_exp) as full-length arrays with NaN where no
    TaSIL target exists. Prefers the full file, falls back to the subset."""
    path = TASIL_FULL if os.path.exists(TASIL_FULL) else (
        TASIL_SUB if os.path.exists(TASIL_SUB) else None)
    delta10 = np.zeros((n_total, 10), dtype=np.float32)
    dexp = np.full((n_total, 3), np.nan, dtype=np.float32)
    if path is None:
        print("  [bc_data_aug] no TaSIL target file found: TaSIL term will "
              "train on ZERO rows (lambda_ta effectively off). Run "
              "make_tasil_targets.py first.")
        return delta10, dexp, None
    d = np.load(path)
    rows = d['rows']
    delta10[rows, :6] = d['delta_norm']
    dexp[rows] = d['d_exp']
    return delta10, dexp, path


class BCDatasetAug(Dataset):
    """(x, y, g_V, g_H, dlt, dexp) over a chosen set of trajectories."""

    def __init__(self, which='train', traj_indices=None):
        base = bc_data.BCDataset(which=which, traj_indices=traj_indices)
        self.X, self.Y = base.X, base.Y
        self.n = base.n
        rows = bc_data.traj_to_rows(base.traj)

        assert os.path.exists(CERT_FILE), \
            f"{CERT_FILE} missing: run make_cert_targets.py first"
        cert = np.load(CERT_FILE, mmap_mode='r')
        self.gV = torch.from_numpy(np.asarray(cert['g_V'][rows],
                                              dtype=np.float32))
        self.gH = torch.from_numpy(np.asarray(cert['g_H'][rows],
                                              dtype=np.float32))

        n_total = np.load(bc_data.DATASET_FILE, mmap_mode='r')['X'].shape[0]
        delta10, dexp, src = _load_tasil_map(n_total)
        self.dlt = torch.from_numpy(delta10[rows])
        self.dexp = torch.from_numpy(dexp[rows])
        n_tasil = int(torch.isfinite(self.dexp[:, 0]).sum())
        print(f"  [bc_data_aug] {which if traj_indices is None else 'subset'}: "
              f"{self.n:,} rows, TaSIL targets on {n_tasil:,} "
              f"({src or 'none'})")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (self.X[i], self.Y[i], self.gV[i], self.gH[i],
                self.dlt[i], self.dexp[i])


def make_loaders_aug(batch_size=4096, num_workers=0, subset=False):
    """(train, val, test) loaders serving the six-tensor items."""
    if subset:
        train_ds = BCDatasetAug(traj_indices=bc_data.make_subset())
    else:
        train_ds = BCDatasetAug('train')
    val_ds = BCDatasetAug('val')
    test_ds = BCDatasetAug('test')
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


# ============================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("bc_data_aug.py SELF-TEST")
    print("=" * 62)
    ds = BCDatasetAug('val')
    x, y, gv, gh, dl, de = ds[0]
    assert x.shape == (10,) and y.shape == (3,)
    assert gv.shape == (3,) and gh.shape == (3,)
    assert dl.shape == (10,) and de.shape == (3,)
    assert torch.all(dl[6:] == 0), "probe must never touch theta/a/e slots"
    # alignment check: g_H must equal 2*raw position of the same row
    base = bc_data.BCDataset('val')
    assert torch.allclose(gh, 2.0 * base.pos[0], atol=1e-4), "g_H misaligned"
    print("  shapes, probe zeros, g_H alignment : OK")
    print("  ALL PASS")