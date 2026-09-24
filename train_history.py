"""
train_history.py
Trainer for the history transformer. Sibling of train_policy.py serving
windows instead of single states, batches carry (x_hist, x_now, y) and the
forward call takes two arguments. The config resolution, loss, optimizer,
stopping rules and per-run logging are imported from train_policy so the
two trainers cannot drift apart.
"""

import os
import sys
import json
import time
import uuid
import argparse
import numpy as np
import torch

import bc_data
import models
import train_policy          # shared protocol: config, loss, optimiser, logging

CKPT_DIR = train_policy.CKPT_DIR


def build_history_model(cfg):
    return models.HistoryPolicy(
        k=cfg['k'],
        d_model=cfg.get('d_model', 64),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 2),
        dec_width=cfg['width'],
        activation=cfg['activation'],
        head=cfg['head'],
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=0.0):
    """One pass. Same structure as train_policy.run_epoch, but unpacks
    (x_hist, x_now, y) and calls the two-argument forward."""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for xh, xn, yb in loader:
        xh = xh.to(device, non_blocking=True)
        xn = xn.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(xh, xn, scaled=True)
            loss = loss_fn(pred, yb)
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
        bs = xh.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def train(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = train_policy.pick_device(cfg)
    print(f"  device: {device}")

    model = build_history_model(cfg).to(device)
    n_params = models.count_params(model)
    print(f"  model: history k={cfg['k']} d_model={cfg.get('d_model',64)} "
          f"layers={cfg.get('n_layers',2)} heads={cfg.get('n_heads',4)} "
          f"dec_width={cfg['width']}  ({n_params:,} params)")

    train_loader, val_loader, _ = bc_data.make_window_loaders(
        batch_size=cfg['batch'], subset=cfg['subset'], k=cfg['k'])
    print(f"  data: {'25% subset' if cfg['subset'] else 'FULL'} train, "
          f"{len(train_loader.dataset):,} windows, batch {cfg['batch']}")

    loss_fn = train_policy.make_loss(cfg)
    optimizer = train_policy.make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg['plateau_factor'],
        patience=cfg['plateau_patience'])

    best_val, best_state, best_epoch, since = float('inf'), None, -1, 0
    t0 = time.time()
    print(f"  training (max {cfg['max_epochs']} epochs, early-stop patience "
          f"{cfg['early_stop_patience']})...")

    for epoch in range(1, cfg['max_epochs'] + 1):
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer,
                       grad_clip=cfg['grad_clip'])
        va = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step(va)
        lr_now = optimizer.param_groups[0]['lr']

        improved = va < best_val - 1e-12
        if improved:
            best_val, best_epoch, since = va, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1

        print(f"    epoch {epoch:3d}  train {tr:.6e}  val {va:.6e}  "
              f"lr {lr_now:.1e}  {'*' if improved else ''}")

        if since >= cfg['early_stop_patience']:
            print(f"    early stop at epoch {epoch}")
            break

    wall = time.time() - t0
    model.load_state_dict(best_state)
    ckpt_path = os.path.join(CKPT_DIR, f'{run_id}.pt')
    torch.save({'state_dict': best_state, 'config': cfg, 'best_val': best_val,
                'best_epoch': best_epoch, 'n_params': n_params,
                'run_id': run_id}, ckpt_path)

    train_policy.log_row(run_id, cfg, best_val, best_epoch, wall, ckpt_path,
                         n_params, str(device))

    print("  " + "-" * 56)
    print(f"  run {run_id}: best val {best_val:.6e} at epoch {best_epoch}, "
          f"{wall/60:.1f} min")
    print(f"  saved -> {ckpt_path}")
    return run_id, best_val, ckpt_path


# ============================================================================
if __name__ == "__main__":
    # start from the shared defaults, add the history-specific fields
    base = dict(train_policy.ANCHOR)
    base.update(arch='history', k=16, d_model=64, n_heads=4, n_layers=2)

    p = argparse.ArgumentParser(description="Train the history transformer (7.3.2).")
    p.add_argument('--config', type=str, default=None)
    for key, val in base.items():
        if isinstance(val, bool) or val is None:
            p.add_argument(f'--{key}', type=str, default=None)
        elif isinstance(val, int) and not isinstance(val, bool):
            p.add_argument(f'--{key}', type=int, default=None)
        elif isinstance(val, float):
            p.add_argument(f'--{key}', type=float, default=None)
        else:
            p.add_argument(f'--{key}', type=str, default=None)
    args = p.parse_args()

    cfg = dict(base)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    for key in base:
        v = getattr(args, key, None)
        if v is not None:
            if key == 'subset':
                cfg[key] = (str(v).lower() in ('true', '1', 'yes'))
            elif isinstance(base[key], bool):
                cfg[key] = (str(v).lower() in ('true', '1', 'yes'))
            else:
                cfg[key] = v
    if cfg['subset'] is None:
        p.error("--subset is required: pass --subset true or --subset false")
    cfg['arch'] = 'history'

    print("=" * 62)
    print("TRAIN THE HISTORY TRANSFORMER (Section 7.3.2)")
    print("=" * 62)
    print("  config:", json.dumps({k: cfg[k] for k in
          ('arch', 'k', 'd_model', 'n_layers', 'n_heads', 'width', 'lr',
           'batch', 'subset', 'seed', 'tag')}))
    train(cfg)