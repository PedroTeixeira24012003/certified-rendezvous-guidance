"""
probe_bias_observability.py
Observability probe. Trains a small regressor whose input is a window of k
corrupted states and whose target is the true per-flight position bias added
that flight, both replayed exactly from bc_data_relabel's keyed corruption.
Nothing here concerns control. The baseline to beat is always guessing zero
bias, and the verdict is the held-out R^2 against that baseline, split by
trajectory so held-out means flights never seen.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

import bc_data
import bc_data_relabel as R


def true_bias_for(traj_id, epoch):
    """Replay corrupt_traj_raw's fixed draw order to recover bias_r exactly."""
    rng = np.random.default_rng([R.NOISE_SEED_BASE, int(traj_id), int(epoch)])
    lam = float(np.exp(rng.uniform(np.log(R.LAM_LO), np.log(R.LAM_HI))))
    sigv = R.SIGV_PER_LAM * lam
    bias_r = rng.normal(0.0, R.BETA * lam * R.sigma_r_of_R(R.BIAS_REF_R), size=3)
    return bias_r.astype(np.float32)          # (3,) the per-flight position bias


def build_dataset(n_trajs, k, epoch=0):
    """For each trajectory: take windows of k corrupted states from mid-flight
    and pair them with that flight's true bias. Mirrors the training path
    exactly: raw X per trajectory -> corrupt_traj_raw -> normalize_x.
    Returns X (n,k,10), y (n,3)."""
    train_traj, _, _ = bc_data.load_split()
    traj = np.asarray(train_traj, dtype=np.int64)[:n_trajs]
    rows = bc_data.traj_to_rows(traj)
    data = np.load(bc_data.DATASET_FILE, mmap_mode='r')
    Xraw = np.asarray(data['X'][rows], dtype=np.float32)      # (n_traj*400, 10)
    Xraw = Xraw.reshape(len(traj), bc_data.STEPS_PER_TRAJ, -1)  # (n_traj,400,10)
    x_mean, x_std = bc_data.load_norm_stats()

    Xs, ys = [], []
    for i, t in enumerate(traj):
        clean = Xraw[i][:, :10].astype(np.float32)           # raw, un-normalized
        corr = R.corrupt_traj_raw(clean, int(t), epoch)      # (400,10) corrupted RAW
        b = true_bias_for(int(t), epoch)                     # (3,) truth for THIS traj
        for s in (60, 150, 260, 360):                        # fully-filled windows
            win = corr[s - k + 1:s + 1]                      # (k,10) raw corrupted
            win_n = bc_data.normalize_x(win, x_mean, x_std)  # normalize like training
            Xs.append(win_n.astype(np.float32))
            ys.append(b)
    return np.stack(Xs), np.stack(ys)


class BiasProbe(nn.Module):
    """Small permutation-agnostic reader: mean-pool the window, MLP to bias(3)."""
    def __init__(self, k, d=10, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):            # x: (B,k,10)
        h = self.net(x)              # (B,k,hidden)
        h = h.mean(dim=1)            # pool over the window
        return self.head(h)          # (B,3) bias estimate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trajs', type=int, default=4000)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--lam', type=float, default=None,
                    help='force a fixed lambda (e.g. 4.0 for L4). Overrides the '
                         'training LAM range so we probe the RIGHT severity.')
    args = ap.parse_args()

    # Force the corruption severity to a fixed rung lambda if requested, so the
    # probe tests the bias magnitude that actually appears at that ladder rung.
    if args.lam is not None:
        R.LAM_LO = args.lam
        R.LAM_HI = args.lam
        print(f"** forcing lambda = {args.lam} (fixed) so the probe tests that "
              f"exact severity **")

    print(f"building dataset: {args.trajs} trajectories, k={args.k} ...")
    X, y = build_dataset(args.trajs, args.k, args.epoch)
    print(f"  {X.shape[0]} windows, X{X.shape}, y{y.shape}")
    print(f"  true |bias| mean = {np.linalg.norm(y, axis=1).mean():.3f} m "
          f"(this is the scale we must explain)")

    # split by trajectory-block into train / held-out (windows from a traj stay
    # together so we test on FLIGHTS never seen, not just windows never seen)
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    # group of 4 windows per traj -> split on trajectory index
    n_traj = n // 4
    tr_traj = rng.permutation(n_traj)
    cut = int(0.8 * n_traj)
    tr_mask = np.zeros(n, bool)
    for tt in tr_traj[:cut]:
        tr_mask[tt * 4:tt * 4 + 4] = True
    te_mask = ~tr_mask

    Xtr = torch.tensor(X[tr_mask]); ytr = torch.tensor(y[tr_mask])
    Xte = torch.tensor(X[te_mask]); yte = torch.tensor(y[te_mask])
    print(f"  train windows {Xtr.shape[0]}, held-out windows {Xte.shape[0]}")

    model = BiasProbe(args.k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    B = 512
    for it in range(args.iters):
        j = torch.randint(0, Xtr.shape[0], (B,))
        opt.zero_grad()
        out = model(Xtr[j])
        loss = lossf(out, ytr[j])
        loss.backward(); opt.step()
        if (it + 1) % 500 == 0:
            with torch.no_grad():
                pred = model(Xte)
                te = lossf(pred, yte).item()
            print(f"  iter {it+1:4d}  train {loss.item():.4e}  held-out {te:.4e}")

    # verdict: compare held-out error to the zero-guess baseline
    with torch.no_grad():
        pred = model(Xte).numpy()
    truth = yte.numpy()
    mse_model = np.mean((pred - truth) ** 2)
    mse_zero = np.mean(truth ** 2)                 # error of always guessing 0
    # per-axis correlation between predicted and true bias on held-out flights
    corrs = []
    for a in range(3):
        c = np.corrcoef(pred[:, a], truth[:, a])[0, 1]
        corrs.append(c)
    r2 = 1.0 - mse_model / mse_zero

    print("\n================= VERDICT =================")
    print(f"  held-out MSE (probe)      : {mse_model:.4e}")
    print(f"  held-out MSE (guess zero) : {mse_zero:.4e}")
    print(f"  R^2 vs zero-baseline      : {r2:+.3f}   (>0 means signal exists)")
    print(f"  per-axis corr(pred,true)  : "
          f"x {corrs[0]:+.2f}  y {corrs[1]:+.2f}  z {corrs[2]:+.2f}")
    print("  ------------------------------------------")
    if r2 > 0.15:
        print("  => BIAS IS RECOVERABLE from the window. The information is")
        print("     present. k=16's failure is a TRAINING problem, not physics.")
        print("     The auxiliary-bias-head fix is worth one focused attempt.")
    elif r2 > 0.03:
        print("  => WEAK signal. The bias is only marginally observable; memory")
        print("     could help a little at best. Explains the flat L4 floor.")
    else:
        print("  => BIAS IS ~UNRECOVERABLE from the window. Unobservable over")
        print("     this geometry. The L4 floor is a HARD LIMIT. No architecture")
        print("     clears it. Write the negative result with full confidence.")
    print("===========================================")


if __name__ == "__main__":
    main()