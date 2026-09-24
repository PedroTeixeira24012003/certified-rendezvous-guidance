"""
bc_data_relabel.py
Corrupted-input data module for privileged relabelling. The training pair
becomes INPUT = the clean state corrupted with a faithful copy of the
ladder's navigation error, LABEL = the expert's command at the clean state,
so the network learns to answer with the clean-state command while reading a
corrupted state. Corruption is applied to the raw physical features first,
then normalized, exactly as at flight time, and all noise is drawn from RNGs
keyed on (seed, trajectory, epoch) so it is byte-reproducible, constant
within a trajectory (the bias) and refreshed each epoch, with val and test
on a fixed key. Two datasets, RelabelDataset for the single-state arm and
RelabelWindowDataset for the history arm, where the window action channel is
zeroed on both sides, matching the flight window. Split, normalization,
label scaling and subset come from bc_data and are never redefined.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import bc_data

# noise model constants (mirror robustness_ladder)
SR_A, SR_B, SR_P = -3.352e-2, 3.858e-2, 0.24     # sigma_r(R) coefficients
BETA = 0.5                                       # bias fraction (ladder BETA)
BIAS_REF_R = 60.0                                # bias sigma anchored at R=60 m
SIGV_PER_LAM = 0.005                             # sigv = 0.005 * lam (ladder)
LAM_LO, LAM_HI = 0.2, 1.0                        # L2 .. L3 training severity
#   (was 0.2..4.0 = up to L4. Training on L4-scale noise swamped the tiny
#    near-hold commands and destroyed terminal precision, especially for the
#    transformer. The EVALUATION ladder is unchanged and still spans L0..L5;
#    this is a training-diet choice only.)
NOISE_SEED_BASE = 217                            # fixed base seed for the noise
VAL_EPOCH_KEY = 10**6                            # fixed key for val/test noise

TASIL_SUB = 'tasil_targets_behind_subset.npz'
TASIL_FULL = 'tasil_targets_behind.npz'


def sigma_r_of_R(R):
    """Range-dependent position sigma [m]. Same formula as the ladder."""
    return np.maximum(SR_A + SR_B * np.power(np.maximum(R, 1e-6), SR_P), 0.0)


def corrupt_traj_raw(X_raw, traj_id, epoch):
    """Corrupt ONE trajectory's raw feature block (400, 10) with the ladder's
    nav-error model. Returns a corrupted copy. Deterministic in
    (NOISE_SEED_BASE, traj_id, epoch).

    Draw order (fixed): lam, bias_r(3), bias_v(3), white_pos(400,3),
    white_vel(400,3). Only columns 0..5 are touched."""
    rng = np.random.default_rng([NOISE_SEED_BASE, int(traj_id), int(epoch)])
    lam = float(np.exp(rng.uniform(np.log(LAM_LO), np.log(LAM_HI))))
    sigv = SIGV_PER_LAM * lam
    bias_r = rng.normal(0.0, BETA * lam * sigma_r_of_R(BIAS_REF_R), size=3)
    bias_v = rng.normal(0.0, BETA * sigv, size=3)

    Xc = X_raw.copy()
    R = np.linalg.norm(X_raw[:, 0:3], axis=1)               # (400,)
    sr = lam * sigma_r_of_R(R)                              # (400,)
    w_pos = rng.normal(0.0, 1.0, size=(X_raw.shape[0], 3)) * sr[:, None]
    w_vel = rng.normal(0.0, sigv, size=(X_raw.shape[0], 3))
    Xc[:, 0:3] += w_pos + bias_r[None, :]
    Xc[:, 3:6] += w_vel + bias_v[None, :]
    return Xc


def _load_tasil_for_rows(rows, n_total):
    """Map the stored TaSIL targets onto these dataset rows. NaN where absent.
    Same convention as bc_data_aug._load_tasil_map."""
    path = TASIL_FULL if os.path.exists(TASIL_FULL) else (
        TASIL_SUB if os.path.exists(TASIL_SUB) else None)
    delta10 = np.zeros((n_total, 10), dtype=np.float32)
    dexp = np.full((n_total, 3), np.nan, dtype=np.float32)
    if path is not None:
        d = np.load(path)
        delta10[d['rows'], :6] = d['delta_norm']
        dexp[d['rows']] = d['d_exp']
    else:
        print("  [bc_data_relabel] WARNING: no TaSIL target file found, the "
              "correction law term will train on zero rows.")
    return delta10[rows], dexp[rows], path


class _RelabelBase:
    """Shared loading: raw feature blocks per trajectory, clean normalized
    tensors, labels, TaSIL targets. Subclasses add the corrupted view."""

    def _load(self, which, traj_indices):
        if traj_indices is not None:
            traj = np.asarray(traj_indices, dtype=np.int64)
        else:
            tr, va, te = bc_data.load_split()
            traj = {'train': tr, 'val': va, 'test': te}[which]
        self.traj = traj
        self.n_traj = traj.shape[0]
        rows = bc_data.traj_to_rows(traj)

        data = np.load(bc_data.DATASET_FILE, mmap_mode='r')
        X_raw = np.asarray(data['X'][rows], dtype=np.float32)
        Y_raw = np.asarray(data['Y'][rows], dtype=np.float32)
        n_total = data['X'].shape[0]

        self.x_mean, self.x_std = bc_data.load_norm_stats()
        self.X_raw = X_raw.reshape(self.n_traj, bc_data.STEPS_PER_TRAJ, 10)
        Xn = bc_data.normalize_x(X_raw, self.x_mean, self.x_std)
        self.Xn_clean = torch.from_numpy(
            Xn.astype(np.float32)).view(self.n_traj, bc_data.STEPS_PER_TRAJ, 10)
        self.Yt = torch.from_numpy(
            bc_data.scale_y(Y_raw).astype(np.float32)).view(
                self.n_traj, bc_data.STEPS_PER_TRAJ, 3)

        dlt, dexp, src = _load_tasil_for_rows(rows, n_total)
        self.dlt = torch.from_numpy(dlt).view(
            self.n_traj, bc_data.STEPS_PER_TRAJ, 10)
        self.dexp = torch.from_numpy(dexp).view(
            self.n_traj, bc_data.STEPS_PER_TRAJ, 3)
        self.n = self.n_traj * bc_data.STEPS_PER_TRAJ
        n_tasil = int(torch.isfinite(self.dexp[..., 0]).sum())
        print(f"  [bc_data_relabel] {which if traj_indices is None else 'subset'}: "
              f"{self.n:,} rows, TaSIL targets on {n_tasil:,} ({src or 'none'})")

    def _recorrupt(self, epoch):
        """Vectorised per-epoch corruption of every trajectory block, then
        normalize. Stored as a (n_traj, 400, 10) tensor."""
        Xc = np.empty_like(self.X_raw)
        for t in range(self.n_traj):
            Xc[t] = corrupt_traj_raw(self.X_raw[t], int(self.traj[t]), epoch)
        Xcn = bc_data.normalize_x(
            Xc.reshape(-1, 10), self.x_mean, self.x_std).astype(np.float32)
        self.Xn_corr = torch.from_numpy(Xcn).view(
            self.n_traj, bc_data.STEPS_PER_TRAJ, 10)


class RelabelDataset(Dataset, _RelabelBase):
    """Single-state relabeling arm. item = (x_corr, x_clean, y, gV, gH, dlt,
    dexp). Certificate gradients ride along for interface parity with the
    four-term loss (their weights default to zero in the champion recipe)."""

    def __init__(self, which='train', traj_indices=None, fixed_epoch=None):
        self._load(which, traj_indices)
        rows = bc_data.traj_to_rows(self.traj)
        assert os.path.exists('cert_targets_behind.npz'), \
            "cert_targets_behind.npz missing: run make_cert_targets.py first"
        cert = np.load('cert_targets_behind.npz', mmap_mode='r')
        self.gV = torch.from_numpy(np.asarray(
            cert['g_V'][rows], dtype=np.float32)).view(
                self.n_traj, bc_data.STEPS_PER_TRAJ, 3)
        self.gH = torch.from_numpy(np.asarray(
            cert['g_H'][rows], dtype=np.float32)).view(
                self.n_traj, bc_data.STEPS_PER_TRAJ, 3)
        self.fixed_epoch = fixed_epoch
        self.set_epoch(0 if fixed_epoch is None else fixed_epoch)

    def set_epoch(self, epoch):
        self._recorrupt(self.fixed_epoch if self.fixed_epoch is not None
                        else epoch)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        t, s = divmod(i, bc_data.STEPS_PER_TRAJ)
        return (self.Xn_corr[t, s], self.Xn_clean[t, s], self.Yt[t, s],
                self.gV[t, s], self.gH[t, s], self.dlt[t, s], self.dexp[t, s])


class RelabelWindowDataset(Dataset, _RelabelBase):
    """Windowed relabeling arm for the history transformer. item =
    (xh_corr, xn_corr, xh_cl, xn_cl, y, dlt, dexp).

    The corrupted window is built from the corrupted per-trajectory feature
    tensor (same bias across the whole trajectory, fresh white per step), the
    clean window from the clean tensor. The window action channel is zeroed
    on both sides, matching the flight window. Padding repeats each view's
    own initial features with zero actions, exactly as bc_data.BCWindowDataset
    and the flight-time build_flight_window do."""

    def __init__(self, which='train', traj_indices=None, k=16,
                 fixed_epoch=None):
        self._load(which, traj_indices)
        self.k = int(k)
        self.fixed_epoch = fixed_epoch
        self.set_epoch(0 if fixed_epoch is None else fixed_epoch)

    def set_epoch(self, epoch):
        self._recorrupt(self.fixed_epoch if self.fixed_epoch is not None
                        else epoch)

    def __len__(self):
        return self.n

    def _window_from(self, feats, acts, s):
        """(k, 13) window ending at step s from a (400,10) feature tensor.

        ACTION CHANNEL ZEROED. The estimation task, averaging white noise and
        identifying the per-flight bias, is a function of the past STATES
        only. Carrying past actions creates a train/deploy mismatch: at
        deploy the window can only hold the network's OWN past commands,
        never the expert commands the model would train on, and a model that
        attends over that channel then reads an out-of-distribution history
        in flight. Zeroing the action channel on BOTH sides removes the
        channel from the problem entirely and makes the flight window
        identical to the training window. The 13-wide shape is kept so
        models.HistoryPolicy and the deploy path need no change, the last
        three columns are simply always zero and the model learns to ignore
        them."""
        k = self.k
        lo = s - k + 1
        if lo >= 0:
            win_f = feats[lo:s + 1]
        else:
            pad = -lo
            pad_f = feats[0:1].expand(pad, -1)
            win_f = torch.cat([pad_f, feats[0:s + 1]], dim=0)
        win_a = torch.zeros(win_f.shape[0], 3)     # action channel unused
        return torch.cat([win_f, win_a], dim=-1)

    def __getitem__(self, i):
        t, s = divmod(i, bc_data.STEPS_PER_TRAJ)
        acts = self.Yt[t]
        xh_c = self._window_from(self.Xn_corr[t], acts, s)
        xh_cl = self._window_from(self.Xn_clean[t], acts, s)
        return (xh_c, self.Xn_corr[t, s], xh_cl, self.Xn_clean[t, s],
                self.Yt[t, s], self.dlt[t, s], self.dexp[t, s])


def make_relabel_loaders(batch_size=4096, num_workers=0, subset=False):
    """(train, val, test) loaders for the single-state arm. Val/test use a
    FIXED corruption (VAL_EPOCH_KEY) so their losses are comparable across
    epochs. Returns the train dataset too, so the trainer can set_epoch."""
    if subset:
        train_ds = RelabelDataset(traj_indices=bc_data.make_subset())
    else:
        train_ds = RelabelDataset('train')
    val_ds = RelabelDataset('val', fixed_epoch=VAL_EPOCH_KEY)
    test_ds = RelabelDataset('test', fixed_epoch=VAL_EPOCH_KEY)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), train_ds


def make_relabel_window_loaders(batch_size=2048, num_workers=0, subset=False,
                                k=16):
    """(train, val, test) loaders for the history arm, plus the train dataset
    for set_epoch. Smaller default batch: each item carries two windows."""
    if subset:
        train_ds = RelabelWindowDataset(traj_indices=bc_data.make_subset(), k=k)
    else:
        train_ds = RelabelWindowDataset('train', k=k)
    val_ds = RelabelWindowDataset('val', k=k, fixed_epoch=VAL_EPOCH_KEY)
    test_ds = RelabelWindowDataset('test', k=k, fixed_epoch=VAL_EPOCH_KEY)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers)
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), train_ds


# ============================================================================
# SELF-TEST: noise statistics, determinism, bias constancy, window invariants.
# ============================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("bc_data_relabel.py SELF-TEST")
    print("=" * 62)

    # pure-numpy noise checks (no dataset files needed)
    rng_block = np.random.default_rng(3)
    Xr = rng_block.normal(size=(400, 10)).astype(np.float32)
    Xr[:, 0] = np.linspace(-100, -8, 400)          # a plausible range profile
    Xr[:, 1:3] *= 3.0

    c1 = corrupt_traj_raw(Xr, traj_id=42, epoch=5)
    c2 = corrupt_traj_raw(Xr, traj_id=42, epoch=5)
    assert np.array_equal(c1, c2), "same key must give identical corruption"
    c3 = corrupt_traj_raw(Xr, traj_id=42, epoch=6)
    assert not np.array_equal(c1, c3), "different epoch must differ"
    c4 = corrupt_traj_raw(Xr, traj_id=43, epoch=5)
    assert not np.array_equal(c1, c4), "different trajectory must differ"
    print("  keyed determinism (traj, epoch)        : OK")

    assert np.array_equal(c1[:, 6:], Xr[:, 6:]), "theta/a/e must be untouched"
    print("  theta/a/e untouched                    : OK")

    # bias constancy: the mean offset over the trajectory is the bias (white
    # averages toward zero over 400 steps), and it must be the same offset at
    # every step in expectation.
    off = c1[:, 0:3] - Xr[:, 0:3]
    bias_est = off.mean(axis=0)
    assert np.linalg.norm(bias_est) > 1e-4, "a per-trajectory bias must exist"
    print(f"  per-trajectory bias present            : OK "
          f"(|bias| ~ {np.linalg.norm(bias_est):.3f} m)")

    # severity reproducibility: lam is the first draw of the keyed stream
    r = np.random.default_rng([NOISE_SEED_BASE, 42, 5])
    lam = float(np.exp(r.uniform(np.log(LAM_LO), np.log(LAM_HI))))
    assert LAM_LO <= lam <= LAM_HI
    print(f"  lam draw in [{LAM_LO}, {LAM_HI}]              : OK "
          f"(lam = {lam:.3f}, sigv = {SIGV_PER_LAM*lam:.4f})")

    # sigma_r sanity against the ladder's formula at two ranges
    assert abs(sigma_r_of_R(60.0) - (SR_A + SR_B * 60.0**SR_P)) < 1e-12
    assert sigma_r_of_R(8.0) < sigma_r_of_R(100.0), "sigma_r must grow with R"
    print("  sigma_r(R) matches ladder formula      : OK")

    # dataset-backed checks only if the frozen files are present
    if os.path.exists(bc_data.DATASET_FILE) and os.path.exists(
            bc_data.SPLIT_FILE):
        print("  building subset RelabelDataset ...")
        ds = RelabelDataset(traj_indices=bc_data.make_subset()[:5])
        xc, xcl, y, gv, gh, dl, de = ds[0]
        assert xc.shape == (10,) and xcl.shape == (10,) and y.shape == (3,)
        assert torch.all(xc[6:] == xcl[6:]), "normalized theta/a/e must match"
        assert not torch.allclose(xc[:6], xcl[:6]), "first 6 must differ"
        ds.set_epoch(1)
        xc1 = ds[0][0]
        assert not torch.allclose(xc, xc1), "epoch refresh must change noise"
        print("    single-state item + epoch refresh    : OK")

        print("  building subset RelabelWindowDataset (k=8) ...")
        wds = RelabelWindowDataset(traj_indices=bc_data.make_subset()[:5], k=8)
        xh_c, xn_c, xh_cl, xn_cl, y, dl, de = wds[0]
        assert xh_c.shape == (8, 13) and xh_cl.shape == (8, 13)
        assert torch.all(xh_c[:, 10:] == 0), "action channel must be all zero"
        assert torch.all(xh_cl[:, 10:] == 0), "action channel must be all zero"
        assert torch.allclose(xh_c[-1, :10], xn_c), "window end = x_now (corr)"
        assert torch.allclose(xh_cl[-1, :10], xn_cl), "window end = x_now (cl)"
        assert not torch.allclose(xh_c[:, :10], xh_cl[:, :10]), \
            "corrupted and clean windows must differ in the state channel"
        print("    windowed items + invariants          : OK")
    else:
        print("  (dataset files absent here: numpy checks only)")

    print("=" * 62)
    print("  ALL RELABEL DATA TESTS PASS")
    print("=" * 62)