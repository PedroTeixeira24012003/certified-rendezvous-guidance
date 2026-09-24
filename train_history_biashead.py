"""
train_history_biashead.py
Auxiliary-bias-head variant of the privileged-relabelled history transformer.
Standalone, modifies no existing file. A second head predicts the per-flight
bias from the pooled window, graded against the true bias, a privileged label
used only in training, so the command loss is joined by an explicit
estimation objective. The saved checkpoint strips the bias_head weights and
loads as a plain models.HistoryPolicy through deploy_closedloop; flight only
ever calls model(x_hist, x_now) -> command.
"""
import argparse, json, os, time, uuid
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import bc_data
import bc_data_relabel as R
import models
import train_history_relabel as THR
from train_policy import CKPT_DIR


class HistoryPolicyBiasHead(models.HistoryPolicy):
    """HistoryPolicy + auxiliary bias head reading the same pooled window."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bias_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.ReLU(),
            nn.Linear(self.d_model, 3),
        )

    def forward(self, x_hist, x_now, scaled=False, want_bias=False):
        h = self.embed(x_hist) + self.pos
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        z = torch.cat([pooled, x_now], dim=-1)
        u = self.head(self.out(self.dec(z)), scaled=scaled)
        if want_bias:
            return u, self.bias_head(pooled)
        return u


def bias_for_traj(traj_id, epoch):
    """Replay corrupt_traj_raw's fixed draw order to recover bias_r exactly."""
    rng = np.random.default_rng([R.NOISE_SEED_BASE, int(traj_id), int(epoch)])
    lam = float(np.exp(rng.uniform(np.log(R.LAM_LO), np.log(R.LAM_HI))))
    _sigv = R.SIGV_PER_LAM * lam
    return rng.normal(0.0, R.BETA * lam * R.sigma_r_of_R(R.BIAS_REF_R),
                      size=3).astype(np.float32)


class BiasWindowWrapper(Dataset):
    """Wrap RelabelWindowDataset; append the true per-flight bias to each item.
    Uses the base dataset's real attributes: .traj (traj ids), .k, and the
    STEPS_PER_TRAJ row layout. Tracks epoch to refresh the bias table."""
    def __init__(self, base):
        self.base = base
        self.S = bc_data.STEPS_PER_TRAJ
        self.epoch = 0
        self._build_table()

    def _build_table(self):
        self.tbl = np.stack([bias_for_traj(int(t), self.epoch)
                             for t in self.base.traj])   # (n_traj, 3)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.base.set_epoch(epoch)
        self._build_table()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]                 # tuple of 7 tensors
        t, _ = divmod(i, self.S)            # same row layout as the base dataset
        b = torch.from_numpy(self.tbl[t])
        return item + (b,)


def biashead_loss(model, batch, cfg, device):
    xh_c, xn_c, xh_cl, xn_cl, y, dlt, dexp, bias = [
        b.to(device, non_blocking=True) for b in batch]
    B = xn_c.shape[0]
    xh_all = torch.cat([xh_c, xh_cl, xh_cl], dim=0)
    xn_all = torch.cat([xn_c, xn_cl, xn_cl + dlt], dim=0)
    out, bpred_all = model(xh_all, xn_all, scaled=True, want_bias=True)
    u_c, u_b, u_p = out[:B], out[B:2 * B], out[2 * B:]

    l_cmd = ((u_c - y) ** 2).mean()
    mask = torch.isfinite(dexp[:, 0])
    if mask.any():
        d_net = u_p[mask] - u_b[mask]
        l_ta = ((d_net - dexp[mask]) ** 2).mean()
    else:
        l_ta = torch.zeros((), device=device)
    l_bias_axes = ((bpred_all[:B] - bias) ** 2)      # (B,3) per-axis
    axmask = cfg.get('bias_axis_mask', None)
    if axmask is not None:
        am = torch.tensor(axmask, device=device, dtype=l_bias_axes.dtype)
        l_bias = (l_bias_axes * am).sum(dim=1).mean() / max(1.0, float(am.sum()))
    else:
        l_bias = l_bias_axes.mean()

    total = l_cmd + cfg['lambda_ta'] * l_ta + cfg['lambda_bias'] * l_bias
    return total, dict(cmd=l_cmd.item(), ta=l_ta.item(), bias=l_bias.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='true')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lambda_ta', type=float, default=5.0)
    ap.add_argument('--lambda_bias', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--tag', default='biashead-k16')
    ap.add_argument('--bias_axes', default='xyz',
                    help="which axes the bias loss uses, e.g. 'xyz' (all) or "
                         "'yz' (drop the unobservable along-track x channel)")
    args = ap.parse_args()

    device = 'cpu'
    axmask = [1.0 if c in args.bias_axes else 0.0 for c in 'xyz']
    cfg = dict(arch='history', k=args.k, d_model=64, n_layers=2, n_heads=4,
               width=256, activation='relu', head='tanh',
               lr=3e-4, batch=4096, subset=(args.subset == 'true'),
               seed=0, lambda_ta=args.lambda_ta, lambda_bias=args.lambda_bias,
               bias_axis_mask=axmask,
               plateau_factor=0.5, plateau_patience=5, tag=args.tag)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("=" * 62)
    print("TRAIN HISTORY TRANSFORMER + AUXILIARY BIAS HEAD (Chapter 21.7)")
    print("=" * 62)
    print("  config:", json.dumps(cfg))
    print(f"  noise: lam [{R.LAM_LO},{R.LAM_HI}]  lambda_bias={args.lambda_bias}")

    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])

    train_loader0, val_loader, _, train_ds = \
        R.make_relabel_window_loaders(batch_size=cfg['batch'],
                                      subset=cfg['subset'], k=cfg['k'])
    wrapped = BiasWindowWrapper(train_ds)
    train_loader = DataLoader(wrapped, batch_size=cfg['batch'], shuffle=True,
                              num_workers=0)

    model = HistoryPolicyBiasHead(k=cfg['k'], d_model=cfg['d_model'],
                                  n_heads=cfg['n_heads'], n_layers=cfg['n_layers'],
                                  dec_width=cfg['width']).to(device)
    print(f"  model: history+biashead k={cfg['k']} "
          f"({models.count_params(model):,} params)")

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg['plateau_factor'],
        patience=cfg['plateau_patience'])
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-hb-' + uuid.uuid4().hex[:6]
    best_val, best_state, t0 = float('inf'), None, time.time()

    for ep in range(1, args.epochs + 1):
        wrapped.set_epoch(ep)
        model.train(); run = 0.0; lastp = {}
        for batch in train_loader:
            opt.zero_grad()
            loss, parts = biashead_loss(model, batch, cfg, device)
            loss.backward(); opt.step(); run += loss.item(); lastp = parts
        run /= max(1, len(train_loader))

        model.eval(); vc = 0.0; nb = 0
        with torch.no_grad():
            for vb in val_loader:
                xh_c, xn_c, xh_cl, xn_cl, y, dlt, dexp = [b.to(device) for b in vb]
                u = model(xh_c, xn_c, scaled=True)
                vc += ((u - y) ** 2).mean().item(); nb += 1
        vc /= max(1, nb); sched.step(vc)
        star = ''
        if vc < best_val - 1e-9:
            best_val = vc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            star = ' *'
        print(f"    ep {ep:3d}  train {run:.6e}  val_cmd {vc:.6e}  "
              f"[bias {lastp.get('bias',0):.2e}]  lr "
              f"{opt.param_groups[0]['lr']:.1e}{star}")

    # strip bias_head.* so the checkpoint loads as a plain HistoryPolicy
    deploy_state = {k: v for k, v in best_state.items()
                    if not k.startswith('bias_head.')}
    ckpt = os.path.join(CKPT_DIR, f'{run_id}.pt')
    torch.save({'state_dict': deploy_state, 'config': cfg,
                'best_val': best_val, 'run_id': run_id}, ckpt)
    torch.save({'state_dict': deploy_state, 'config': cfg,
                'run_id': run_id + '-last'}, ckpt.replace('.pt', '-last.pt'))
    dt = (time.time() - t0) / 60.0
    print("  " + "-" * 56)
    print(f"  run {run_id}: best val_cmd {best_val:.6e}, {dt:.1f} min")
    print(f"  saved -> {ckpt}  (+ -last.pt)")
    print("  NOTE: bias_head stripped from checkpoint; deploy loads plain history.")


if __name__ == "__main__":
    main()