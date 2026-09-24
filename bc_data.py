"""
bc_data.py
Shared data module for the behavioural-cloning campaign. Defines the frozen
trajectory-level split, the stored input normalization, the label scaling and
the torch datasets and loaders in one place, so every training and evaluation
script applies the identical transforms. Run directly for a self-test.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Constants
# ============================================================================
DATASET_FILE = 'th_bc_dataset_behind.npz'
NORM_FILE    = 'bc_norm_stats_behind.npz'
SPLIT_FILE   = 'split_indices.npz'
SUBSET_FILE  = 'sweep_subset.npz'          # written once by make_subset()

STEPS_PER_TRAJ = 400                        # rows per trajectory
U_MAX          = 0.082                      # per-axis thrust limit [m/s^2]
CLAMP          = 0.999                      # label clamp inside the tanh range
SUBSET_FRAC    = 0.25                       # sweep subset fraction of train
SUBSET_SEED    = 7                          # fixed seed for the subset draw

FEATURE_NAMES = ['x', 'y', 'z', 'vx', 'vy', 'vz',
                 'sin_theta', 'cos_theta', 'a', 'e']
OUTPUT_NAMES  = ['ux', 'uy', 'uz']


# ============================================================================
# Block rule, trajectory indices to row indices.
# Trajectory i owns rows [400*i, 400*(i+1)).
# ============================================================================
def traj_to_rows(traj_indices):
    """Return the flat array of all row indices the given trajectories own,
    in trajectory-contiguous order."""
    traj_indices = np.asarray(traj_indices, dtype=np.int64)
    base = (traj_indices[:, None] * STEPS_PER_TRAJ)          # (T, 1)
    offs = np.arange(STEPS_PER_TRAJ, dtype=np.int64)[None, :]  # (1, 400)
    return (base + offs).reshape(-1)                          # (T*400,)


# ============================================================================
# Loaders for the frozen files
# ============================================================================
def load_split():
    """Return (train_traj, val_traj, test_traj) trajectory-index arrays."""
    d = np.load(SPLIT_FILE)
    return d['train_traj'], d['val_traj'], d['test_traj']


def load_norm_stats():
    """Return the stored input (x_mean, x_std) as float32 arrays of length 10."""
    d = np.load(NORM_FILE)
    x_mean = d['x_mean'].astype(np.float32)
    x_std  = d['x_std'].astype(np.float32)
    # no zero std, a constant feature would divide by zero
    x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
    return x_mean, x_std


# ============================================================================
# The exact transforms, exposed so deployment and evaluation reuse them
# ============================================================================
def normalize_x(X, x_mean, x_std):
    """z-score the 10 input features. Accepts numpy or torch, (..., 10)."""
    return (X - x_mean) / x_std


def scale_y(Y):
    """Scale raw commands in m/s^2 to targets in [-1, 1], clamped to +-CLAMP."""
    t = Y / U_MAX
    return np.clip(t, -CLAMP, CLAMP)


def denormalize_u(t):
    """Map a network output in [-1, 1] back to a command in m/s^2."""
    return t * U_MAX


# ============================================================================
# Fixed 25% sweep subset of the training trajectories, drawn once and saved
# ============================================================================
def make_subset(force=False):
    """Return the fixed sweep subset of train trajectories, drawing and saving
    it on first use. An existing file is loaded, not redrawn, unless force."""
    import os
    if os.path.exists(SUBSET_FILE) and not force:
        return np.load(SUBSET_FILE)['subset_traj']
    train_traj, _, _ = load_split()
    n_sub = int(round(SUBSET_FRAC * train_traj.shape[0]))
    rng = np.random.default_rng(SUBSET_SEED)
    pick = rng.choice(train_traj, size=n_sub, replace=False)
    subset_traj = np.sort(pick).astype(np.int64)
    np.savez(SUBSET_FILE, subset_traj=subset_traj,
             seed=SUBSET_SEED, frac=SUBSET_FRAC)
    return subset_traj


# ============================================================================
# Dataset over a chosen set of trajectories, memory-mapped so only the rows
# a split needs are loaded
# ============================================================================
class BCDataset(Dataset):
    """Behavioural-cloning dataset over a chosen set of trajectories.

    which : 'train', 'val', 'test', or an explicit trajectory-index array.
    Inputs are z-scored with the stored stats, targets scaled and clamped,
    everything returned as float32 torch tensors.
    """
    def __init__(self, which='train', traj_indices=None):
        # resolve which trajectories this dataset covers
        if traj_indices is not None:
            traj = np.asarray(traj_indices, dtype=np.int64)
        else:
            train_traj, val_traj, test_traj = load_split()
            traj = {'train': train_traj, 'val': val_traj,
                    'test': test_traj}[which]

        rows = traj_to_rows(traj)

        # memory-map the big arrays, gather only the rows we need
        data = np.load(DATASET_FILE, mmap_mode='r')
        X = np.asarray(data['X'][rows], dtype=np.float32)   # (n, 10)
        Y = np.asarray(data['Y'][rows], dtype=np.float32)   # (n, 3)

        x_mean, x_std = load_norm_stats()
        Xn = normalize_x(X, x_mean, x_std).astype(np.float32)
        Yt = scale_y(Y).astype(np.float32)

        self.X = torch.from_numpy(Xn)
        self.Y = torch.from_numpy(Yt)
        self.n = self.X.shape[0]
        self.traj = traj

        # raw command and position kept for phase-split evaluation in
        # physical units
        self.Y_raw = torch.from_numpy(Y)                    # m/s^2
        self.pos   = torch.from_numpy(X[:, :3].copy())      # m (raw, unnormalized)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ============================================================================
# One-call loader factory
# ============================================================================
def make_loaders(batch_size=4096, num_workers=0, subset=False):
    """Return (train_loader, val_loader, test_loader). subset=True routes
    train through the fixed sweep subset. Only train is shuffled."""
    if subset:
        train_ds = BCDataset(traj_indices=make_subset())
    else:
        train_ds = BCDataset('train')
    val_ds  = BCDataset('val')
    test_ds = BCDataset('test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ============================================================================
# Windowed dataset for the history policy. Each item is the last k steps of
# (features, action) plus the current features and target. Three properties
# hold, windows never cross a trajectory boundary, the first k-1 steps are
# left-padded by repeating the initial state with zero actions, and the
# current step's action is zeroed in the window so the target never appears
# in the input.
#   x_hist : (k, 13) normalized features ++ scaled actions, oldest first
#   x_now  : (10,)   the current normalized features
#   y      : (3,)    the scaled target command at the current step
# ============================================================================
class BCWindowDataset(Dataset):
    def __init__(self, which='train', traj_indices=None, k=16):
        if traj_indices is not None:
            traj = np.asarray(traj_indices, dtype=np.int64)
        else:
            train_traj, val_traj, test_traj = load_split()
            traj = {'train': train_traj, 'val': val_traj,
                    'test': test_traj}[which]
        self.k = int(k)
        self.traj = traj
        self.n_traj = traj.shape[0]

        rows = traj_to_rows(traj)
        data = np.load(DATASET_FILE, mmap_mode='r')
        X = np.asarray(data['X'][rows], dtype=np.float32)
        Y = np.asarray(data['Y'][rows], dtype=np.float32)

        x_mean, x_std = load_norm_stats()
        Xn = normalize_x(X, x_mean, x_std).astype(np.float32)
        Yt = scale_y(Y).astype(np.float32)

        # reshape to (n_traj, 400, .) so trajectory structure is explicit and
        # a window can never silently cross a boundary
        self.Xn = torch.from_numpy(Xn).view(self.n_traj, STEPS_PER_TRAJ, -1)
        self.Yt = torch.from_numpy(Yt).view(self.n_traj, STEPS_PER_TRAJ, -1)
        self.n = self.n_traj * STEPS_PER_TRAJ

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        t, s = divmod(i, STEPS_PER_TRAJ)      # trajectory index, step in trajectory
        k = self.k
        feats = self.Xn[t]                    # (400, 10)
        acts = self.Yt[t]                     # (400, 3)

        lo = s - k + 1
        if lo >= 0:
            win_f = feats[lo:s + 1]                       # (k, 10)
            win_a = acts[lo:s + 1].clone()                # (k, 3)
        else:
            pad = -lo                                     # steps to left-pad
            # pad repeats the initial state with zero actions
            pad_f = feats[0:1].expand(pad, -1)            # (pad, 10)
            pad_a = torch.zeros(pad, acts.shape[1])
            win_f = torch.cat([pad_f, feats[0:s + 1]], dim=0)
            win_a = torch.cat([pad_a, acts[0:s + 1].clone()], dim=0)

        # the current step's action is the target, zero its slot in the window
        win_a[-1] = 0.0

        x_hist = torch.cat([win_f, win_a], dim=-1)        # (k, 13)
        x_now = feats[s]                                  # (10,)
        y = acts[s]                                       # (3,)
        return x_hist, x_now, y


def make_window_loaders(batch_size=4096, num_workers=0, subset=False, k=16):
    """Return (train, val, test) loaders serving (x_hist, x_now, y) windows."""
    if subset:
        train_ds = BCWindowDataset(traj_indices=make_subset(), k=k)
    else:
        train_ds = BCWindowDataset('train', k=k)
    val_ds = BCWindowDataset('val', k=k)
    test_ds = BCWindowDataset('test', k=k)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ============================================================================
# Self-test, run this file directly to verify every invariant on the real data
# ============================================================================
if __name__ == "__main__":
    import os
    print("=" * 62)
    print("bc_data.py SELF-TEST")
    print("=" * 62)

    # files present?
    for f in (DATASET_FILE, NORM_FILE, SPLIT_FILE):
        print(f"  {'FOUND' if os.path.exists(f) else 'MISSING'} : {f}")
    assert os.path.exists(SPLIT_FILE), "run make_split.py first"

    # block rule
    rows = traj_to_rows(np.array([0, 1, 9999]))
    assert rows[0] == 0 and rows[399] == 399
    assert rows[400] == 400 and rows[799] == 799
    assert rows[800] == 9999 * 400 and rows[-1] == 10000 * 400 - 1
    print("  block rule (traj -> rows)       : OK")

    # split partitions cleanly and by trajectory
    tr, va, te = load_split()
    assert tr.shape[0] == 8000 and va.shape[0] == 1000 and te.shape[0] == 1000
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(va, te)) == 0
    print(f"  split disjoint 8000/1000/1000   : OK")

    # no row overlap between splits
    tr_rows = set(traj_to_rows(tr).tolist())
    va_rows = set(traj_to_rows(va).tolist())
    te_rows = set(traj_to_rows(te).tolist())
    assert tr_rows.isdisjoint(va_rows)
    assert tr_rows.isdisjoint(te_rows)
    assert va_rows.isdisjoint(te_rows)
    print("  row-level disjoint (no leakage) : OK")

    # subset draws, is a subset of train, is deterministic
    s1 = make_subset(force=True)
    s2 = make_subset()                       # loads the saved file
    assert np.array_equal(s1, s2)
    assert set(s1.tolist()).issubset(set(tr.tolist()))
    assert s1.shape[0] == 2000
    print(f"  sweep subset (2000, seed 7)     : OK (subset of train, deterministic)")

    # label scaling round-trips and clamps
    y = np.array([[0.0, U_MAX, -U_MAX], [2 * U_MAX, -2 * U_MAX, 0.041]],
                 dtype=np.float32)
    t = scale_y(y)
    assert t.max() <= CLAMP + 1e-9 and t.min() >= -CLAMP - 1e-9
    back = denormalize_u(t)
    assert abs(back[1, 2] - 0.041) < 1e-6         # in-range value preserved
    print("  label scale/clamp/denorm        : OK")

    # a small dataset actually builds and normalizes
    print("  building val dataset (400k rows) ...")
    ds = BCDataset('val')
    xb, yb = ds[0]
    assert xb.shape == (10,) and yb.shape == (3,)
    Xall = ds.X.numpy()
    print(f"    val inputs  mean~0  : {np.abs(Xall.mean(0)).max():.3f} (max abs feature mean)")
    print(f"    val inputs  std~1   : range [{Xall.std(0).min():.2f}, {Xall.std(0).max():.2f}]")
    print(f"    val targets range   : [{ds.Y.min():.3f}, {ds.Y.max():.3f}]  (within +-{CLAMP})")
    assert ds.Y.min() >= -CLAMP - 1e-6 and ds.Y.max() <= CLAMP + 1e-6

    # windowed dataset invariants
    print("  building windowed val dataset (k=16) ...")
    K = 16
    wds = BCWindowDataset('val', k=K)
    xh, xn, yy = wds[0]
    assert xh.shape == (K, 13) and xn.shape == (10,) and yy.shape == (3,)
    print(f"    window shapes (k,13)/(10,)/(3,) : OK")

    # leakage guard, the last action slot must be exactly zero everywhere
    for probe in (0, 5, 399, 400, 400 + 17, len(wds) - 1):
        xh_p, _, _ = wds[probe]
        assert torch.all(xh_p[-1, 10:] == 0), f"current action leaked at {probe}"
    print("    current action never in window   : OK (leakage guard)")

    # padding convention at the very start of a trajectory
    xh0, xn0, _ = wds[0]                      # step 0 of the first trajectory
    assert torch.all(xh0[:, 10:] == 0), "all actions must be zero at step 0"
    first_feat = xh0[-1, :10]
    assert torch.allclose(xh0[0, :10], first_feat), "pad must repeat initial state"
    assert torch.allclose(xn0, first_feat), "x_now must be the current features"
    print("    pad = initial state, zero action : OK")

    # trajectory boundaries, the window at step 0 of traj 1 must not contain
    # any features from traj 0
    idx_t1_s0 = STEPS_PER_TRAJ                # first row of the second trajectory
    xh1, xn1, _ = wds[idx_t1_s0]
    t0_last = wds.Xn[0, -1]                   # last features of trajectory 0
    assert not torch.allclose(xh1[0, :10], t0_last), "window crossed a boundary"
    assert torch.allclose(xh1[0, :10], wds.Xn[1, 0]), "pad must use own traj start"
    print("    windows respect traj boundaries  : OK")

    # a mid-trajectory window is a true contiguous slice, correctly aligned
    s = 100
    xh_m, xn_m, y_m = wds[s]                  # trajectory 0, step 100
    assert torch.allclose(xh_m[:, :10], wds.Xn[0, s - K + 1:s + 1])
    assert torch.allclose(xh_m[:-1, 10:], wds.Yt[0, s - K + 1:s])
    assert torch.allclose(y_m, wds.Yt[0, s])
    print("    mid-window slice + alignment     : OK")

    print("=" * 62)
    print("  ALL INVARIANTS PASS")
    print("=" * 62)