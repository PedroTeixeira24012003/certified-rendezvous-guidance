"""
train_history_relabel.py
Privileged-relabelling trainer for the history transformer. The network
reads a window of corrupted states and is trained to output the clean-state
expert command, plus a TaSIL correction-law term anchored on the clean
window, where the stored targets were measured, with the probe on x_now
only. One batched forward serves all three evaluation points. Training
protocol, optimizer, scheduler, logging and checkpoint format are the shared
machinery from train_policy and train_history, imported. Saves best-val and
'-last' checkpoints.
"""

import os
import json
import time
import uuid
import numpy as np
import torch

import bc_data_relabel
import models
import train_policy
from train_policy import (make_optimizer, pick_device, mps_sanity_check,
                          log_row, CKPT_DIR)
from train_history import build_history_model


ANCHOR_HR = dict(train_policy.ANCHOR)
ANCHOR_HR.update(arch='history', k=16, d_model=64, n_heads=4, n_layers=2,
                 lambda_ta=5.0, lambda_v=0.0, lambda_h=0.0,
                 cert_norm='unit', tag='relabel-hist')


def resolve_config_hr():
    import argparse
    p = argparse.ArgumentParser(
        description="Train the privileged-relabeled history transformer (21.7).")
    p.add_argument('--config', type=str, default=None)
    for k, v in ANCHOR_HR.items():
        if isinstance(v, bool) or v is None:
            p.add_argument(f'--{k}', type=str, default=None)
        elif isinstance(v, int) and not isinstance(v, bool):
            p.add_argument(f'--{k}', type=int, default=None)
        elif isinstance(v, float):
            p.add_argument(f'--{k}', type=float, default=None)
        else:
            p.add_argument(f'--{k}', type=str, default=None)
    args = p.parse_args()
    cfg = dict(ANCHOR_HR)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    for k in ANCHOR_HR:
        v = getattr(args, k, None)
        if v is not None:
            if k == 'subset' or isinstance(ANCHOR_HR[k], bool):
                cfg[k] = (str(v).lower() in ('true', '1', 'yes', 'subset'))
            else:
                cfg[k] = v
    if cfg['subset'] is None:
        p.error("--subset is required (true = sweep subset, false = full)")
    cfg['arch'] = 'history'
    return cfg


def relabel_loss_hist(model, batch, cfg, device):
    """Clean-anchored windowed relabeling loss. Returns (total, parts)."""
    xh_c, xn_c, xh_cl, xn_cl, y, dlt, dexp = [
        b.to(device, non_blocking=True) for b in batch]

    B = xn_c.shape[0]
    xh_all = torch.cat([xh_c, xh_cl, xh_cl], dim=0)
    xn_all = torch.cat([xn_c, xn_cl, xn_cl + dlt], dim=0)
    out = model(xh_all, xn_all, scaled=True)
    u_c, u_b, u_p = out[:B], out[B:2 * B], out[2 * B:]

    e_c = u_c - y
    l_cmd = (e_c ** 2).mean()

    mask = torch.isfinite(dexp[:, 0])
    if mask.any():
        d_net = u_p[mask] - u_b[mask]
        l_ta = ((d_net - dexp[mask]) ** 2).mean()
    else:
        l_ta = torch.zeros((), device=device)

    total = l_cmd + cfg['lambda_ta'] * l_ta
    parts = dict(cmd=l_cmd.item(), ta=l_ta.item())
    return total, parts


def run_epoch_hr(model, loader, cfg, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    grad_clip = cfg['grad_clip']
    tot, n = 0.0, 0
    acc = dict(cmd=0.0, ta=0.0)
    for batch in loader:
        with torch.set_grad_enabled(train):
            loss, parts = relabel_loss_hist(model, batch, cfg, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            finite = all(torch.isfinite(p.grad).all()
                         for p in model.parameters() if p.grad is not None)
            if not finite:
                optimizer.zero_grad(set_to_none=True)
                continue
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = batch[0].shape[0]
        tot += loss.item() * bs
        for k in acc:
            acc[k] += parts[k] * bs
        n += bs
    n = max(n, 1)
    return tot / n, {k: v / n for k, v in acc.items()}


def train(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-hr-' + uuid.uuid4().hex[:6]

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = pick_device(cfg)
    if device.type == 'mps':
        forced = mps_sanity_check(cfg)
        if forced is not None:
            device = forced
    print(f"  device: {device}")

    model = build_history_model(cfg).to(device)
    n_params = models.count_params(model)
    print(f"  model: history k={cfg['k']} d_model={cfg.get('d_model', 64)} "
          f"layers={cfg.get('n_layers', 2)} heads={cfg.get('n_heads', 4)} "
          f"dec_width={cfg['width']}  ({n_params:,} params)")
    print(f"  weights: lambda_ta={cfg['lambda_ta']}")
    print(f"  noise: lam log-uniform [{bc_data_relabel.LAM_LO}, "
          f"{bc_data_relabel.LAM_HI}] per trajectory per epoch, "
          f"sigv = {bc_data_relabel.SIGV_PER_LAM} * lam, bias fraction "
          f"{bc_data_relabel.BETA}")

    train_loader, val_loader, _, train_ds = \
        bc_data_relabel.make_relabel_window_loaders(
            batch_size=cfg['batch'], subset=cfg['subset'], k=cfg['k'])

    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg['plateau_factor'],
        patience=cfg['plateau_patience'])

    best_val, best_state, best_epoch, since = float('inf'), None, -1, 0
    t0 = time.time()
    for epoch in range(1, cfg['max_epochs'] + 1):
        train_ds.set_epoch(epoch)
        tr, trp = run_epoch_hr(model, train_loader, cfg, device, optimizer)
        va, vap = run_epoch_hr(model, val_loader, cfg, device)
        scheduler.step(va)
        improved = va < best_val - 1e-9
        if improved:
            best_val, best_epoch, since = va, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
        print(f"    ep {epoch:3d}  train {tr:.6e}  val {va:.6e}  "
              f"[cmd {vap['cmd']:.2e} ta {vap['ta']:.2e}]  "
              f"lr {optimizer.param_groups[0]['lr']:.1e} "
              f"{'*' if improved else ''}")
        if since >= cfg['early_stop_patience']:
            print(f"    early stop at epoch {epoch}")
            break
    wall = time.time() - t0

    last_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ckpt = os.path.join(CKPT_DIR, f'{run_id}.pt')
    torch.save({'state_dict': best_state, 'config': cfg, 'best_val': best_val,
                'best_epoch': best_epoch, 'n_params': n_params,
                'run_id': run_id}, ckpt)
    torch.save({'state_dict': last_state, 'config': cfg,
                'run_id': run_id + '-last'}, ckpt.replace('.pt', '-last.pt'))
    log_row(run_id, cfg, best_val, best_epoch, wall, ckpt, n_params,
            str(device))

    print("  " + "-" * 56)
    print(f"  run {run_id}: best val {best_val:.6e} at ep {best_epoch}, "
          f"{wall/60:.1f} min")
    print(f"  saved -> {ckpt}  (+ -last.pt)")
    print("  NEXT: fly on the fixed-tape ladder (the ladder already knows how")
    print("  to fly a history model with its rolling window of noisy reads).")
    return run_id, best_val, ckpt


if __name__ == "__main__":
    cfg = resolve_config_hr()
    print("=" * 62)
    print("TRAIN THE PRIVILEGED-RELABELED HISTORY TRANSFORMER (Chapter 21.7)")
    print("=" * 62)
    print("  config:", json.dumps({k: cfg[k] for k in
          ('arch', 'k', 'd_model', 'n_layers', 'n_heads', 'width', 'lr',
           'batch', 'subset', 'seed', 'lambda_ta', 'tag')}))
    train(cfg)