"""
train_policy.py
Trains one behavioural-cloning policy per run, MLP or encoder-decoder,
from a configuration resolved as defaults, then an optional JSON file,
then CLI flags. Trains in scaled command space with early stopping and
a plateau scheduler, saves the best-epoch checkpoint and appends one
row per run to experiments_log.csv. The history policy trains with
train_history.py, which reuses this loop on the windowed loader.
"""

import os
import sys
import csv
import json
import time
import uuid
import argparse
import numpy as np
import torch
import torch.nn as nn

import bc_data
import models

LOG_FILE = 'experiments_log.csv'
CKPT_DIR = 'checkpoints'


# ============================================================================
# Default configuration. A sweep run is this dict with one field changed.
# ============================================================================
ANCHOR = dict(
    arch='mlp',            # 'mlp' | 'encdec'  (history via sibling script)
    depth=3,
    width=256,
    activation='relu',
    head='tanh',           # 'tanh' | 'linear'
    conditioning='concat', # encdec only: 'concat' | 'film'
    loss='mse',            # 'mse' | 'huber' | 'l1'
    huber_delta=0.1,       # in u/U_MAX units, used only if loss='huber'
    optimizer='adam',      # 'adam' | 'adamw' | 'sgd'
    weight_decay=0.0,
    lr=3e-4,               # stable on this data, 1e-3 diverged
    batch=4096,
    max_epochs=100,
    early_stop_patience=10,
    plateau_factor=0.5,
    plateau_patience=5,
    grad_clip=1.0,         # max gradient norm per step (0 disables). Stops rare
                           # catastrophic batches from scrambling the weights.
    seed=0,
    device='cpu',          # cpu is the reliable default, MPS produced bad
                           # gradients on this workload; override with
                           # --device mps | 'auto' | 'cpu'
    subset=None,           # REQUIRED: True (25% sweep subset) or False (full)
    tag='',                # free-text label for the run
)


# ============================================================================
# Config resolution, defaults -> JSON -> CLI overrides
# ============================================================================
def resolve_config():
    p = argparse.ArgumentParser(description="Train one BC policy (Chapter 7).")
    p.add_argument('--config', type=str, default=None,
                   help='JSON file with any subset of config fields')
    # every default field is also a CLI flag; None means "not set on CLI"
    for k, v in ANCHOR.items():
        if isinstance(v, bool) or v is None:
            # booleans and the required subset flag, parsed as strings
            p.add_argument(f'--{k}', type=str, default=None)
        elif isinstance(v, int) and not isinstance(v, bool):
            p.add_argument(f'--{k}', type=int, default=None)
        elif isinstance(v, float):
            p.add_argument(f'--{k}', type=float, default=None)
        else:
            p.add_argument(f'--{k}', type=str, default=None)
    args = p.parse_args()

    cfg = dict(ANCHOR)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    # CLI overrides, only fields actually provided
    for k in ANCHOR:
        v = getattr(args, k)
        if v is not None:
            # coerce the string-typed bool/subset flags
            if k == 'subset':
                cfg[k] = (str(v).lower() in ('true', '1', 'yes', 'subset'))
            elif isinstance(ANCHOR[k], bool):
                cfg[k] = (str(v).lower() in ('true', '1', 'yes'))
            else:
                cfg[k] = v

    # subset is required, force an explicit choice
    if cfg['subset'] is None:
        p.error("--subset is required: pass --subset true (25% sweep subset) "
                "or --subset false (full train set).")
    return cfg


# ============================================================================
# Loss and optimizer factories
# ============================================================================
def make_loss(cfg):
    name = cfg['loss'].lower()
    if name == 'mse':
        return nn.MSELoss()
    if name == 'l1':
        return nn.L1Loss()
    if name == 'huber':
        return nn.HuberLoss(delta=cfg['huber_delta'])
    raise ValueError(f"unknown loss '{name}'")


def make_optimizer(model, cfg):
    name = cfg['optimizer'].lower()
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                weight_decay=cfg['weight_decay'])
    if name == 'adamw':
        wd = cfg['weight_decay'] if cfg['weight_decay'] > 0 else 1e-4
        return torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=wd)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9,
                               weight_decay=cfg['weight_decay'])
    raise ValueError(f"unknown optimizer '{name}'")


def build_model(cfg):
    if cfg['arch'] == 'mlp':
        return models.MLPPolicy(depth=cfg['depth'], width=cfg['width'],
                                activation=cfg['activation'], head=cfg['head'])
    if cfg['arch'] == 'encdec':
        return models.EncDecPolicy(trunk_width=cfg['width'],
                                   trunk_depth=cfg['depth'],
                                   activation=cfg['activation'],
                                   head=cfg['head'],
                                   conditioning=cfg['conditioning'])
    if cfg['arch'] == 'history':
        raise SystemExit("arch='history' needs the windowed loader. Train it "
                         "with train_history.py (Section 7.3.2), not this script.")
    raise ValueError(f"unknown arch '{cfg['arch']}'")


# ============================================================================
# Device selection and the MPS-vs-CPU numerical check
# ============================================================================
def pick_device(cfg=None):
    choice = (cfg or {}).get('device', 'auto')
    if choice == 'cpu':
        return torch.device('cpu')
    if choice == 'mps':
        return torch.device('mps')
    # auto
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def mps_sanity_check(cfg):
    """Run one batch of the configured model on MPS and on CPU and check the
    outputs agree to 1e-6. Falls back to CPU on disagreement."""
    if not torch.backends.mps.is_available():
        print("  MPS sanity check: skipped (MPS unavailable, using CPU)")
        return
    torch.manual_seed(cfg['seed'])
    m_cpu = build_model(cfg)
    m_mps = build_model(cfg)
    m_mps.load_state_dict(m_cpu.state_dict())     # identical weights
    x = torch.randn(64, models.N_IN)
    with torch.no_grad():
        y_cpu = m_cpu(x)
        y_mps = m_mps.to('mps')(x.to('mps')).cpu()
    max_diff = (y_cpu - y_mps).abs().max().item()
    status = "OK" if max_diff < 1e-6 else "WARNING"
    print(f"  MPS sanity check: {status}  (max |cpu-mps| = {max_diff:.2e})")
    if max_diff >= 1e-6:
        print("  WARNING: MPS and CPU disagree beyond 1e-6. Falling back to CPU "
              "for correctness. Report this.")
        return torch.device('cpu')
    return None   # None => keep the chosen device


# ============================================================================
# One epoch
# ============================================================================
def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=0.0):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(xb, scaled=True)          # train in scaled space
            loss = loss_fn(pred, yb)
        if train:
            optimizer.zero_grad(set_to_none=True)
            # skip a batch with a non-finite loss rather than let one bad
            # gradient turn every weight into inf/nan
            if not torch.isfinite(loss):
                continue
            loss.backward()
            # skip the step too if any gradient came back non-finite
            finite_grads = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters() if p.grad is not None)
            if not finite_grads:
                optimizer.zero_grad(set_to_none=True)
                continue
            # clip the gradient norm so no single batch takes a catastrophic step
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = xb.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


# ============================================================================
# Train one configuration end to end
# ============================================================================
def train(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = pick_device(cfg)
    if device.type == 'mps':
        forced = mps_sanity_check(cfg)
        if forced is not None:
            device = forced
    else:
        print(f"  device forced to {device.type} (skipping MPS sanity check)")
    print(f"  device: {device}")

    model = build_model(cfg).to(device)
    n_params = models.count_params(model)
    print(f"  model: {cfg['arch']} depth={cfg['depth']} width={cfg['width']} "
          f"act={cfg['activation']} head={cfg['head']}  ({n_params:,} params)")

    train_loader, val_loader, _ = bc_data.make_loaders(
        batch_size=cfg['batch'], subset=cfg['subset'])
    print(f"  data: {'25% subset' if cfg['subset'] else 'FULL'} train, "
          f"{len(train_loader.dataset):,} pairs, batch {cfg['batch']}")

    loss_fn = make_loss(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg['plateau_factor'],
        patience=cfg['plateau_patience'])

    best_val = float('inf')
    best_state = None
    best_epoch = -1
    since_improve = 0
    t0 = time.time()

    print(f"  training (max {cfg['max_epochs']} epochs, early-stop patience "
          f"{cfg['early_stop_patience']})...")
    for epoch in range(1, cfg['max_epochs'] + 1):
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer,
                       grad_clip=cfg['grad_clip'])
        va = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step(va)
        lr_now = optimizer.param_groups[0]['lr']

        improved = va < best_val - 1e-9
        if improved:
            best_val = va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            since_improve = 0
        else:
            since_improve += 1

        print(f"    epoch {epoch:3d}  train {tr:.6e}  val {va:.6e}  "
              f"lr {lr_now:.1e}  {'*' if improved else ''}")

        if since_improve >= cfg['early_stop_patience']:
            print(f"    early stop at epoch {epoch} "
                  f"(no val improvement for {cfg['early_stop_patience']})")
            break

    wall = time.time() - t0

    # restore and save the best epoch
    model.load_state_dict(best_state)
    ckpt_path = os.path.join(CKPT_DIR, f'{run_id}.pt')
    torch.save({'state_dict': best_state, 'config': cfg,
                'best_val': best_val, 'best_epoch': best_epoch,
                'n_params': n_params, 'run_id': run_id}, ckpt_path)

    log_row(run_id, cfg, best_val, best_epoch, wall, ckpt_path, n_params, str(device))

    print("  " + "-" * 56)
    print(f"  run {run_id}: best val {best_val:.6e} at epoch {best_epoch}, "
          f"{wall/60:.1f} min")
    print(f"  saved -> {ckpt_path}")
    return run_id, best_val, ckpt_path


# ============================================================================
# One CSV row per run
# ============================================================================
def log_row(run_id, cfg, best_val, best_epoch, wall, ckpt_path, n_params, device):
    header = ['run_id', 'best_val', 'best_epoch', 'n_params', 'wall_s',
              'device', 'seed', 'subset', 'config_json', 'ckpt']
    row = [run_id, f'{best_val:.8e}', best_epoch, n_params, f'{wall:.1f}',
           device, cfg['seed'], cfg['subset'], json.dumps(cfg, sort_keys=True),
           ckpt_path]
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


# ============================================================================
if __name__ == "__main__":
    cfg = resolve_config()
    print("=" * 62)
    print("TRAIN ONE POLICY (Chapter 7)")
    print("=" * 62)
    print("  config:", json.dumps({k: cfg[k] for k in
          ('arch','depth','width','activation','head','loss','optimizer',
           'lr','batch','subset','seed','tag')}, indent=None))
    train(cfg)