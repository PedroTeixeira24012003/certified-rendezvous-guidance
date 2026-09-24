"""
train_policy_4term.py
Four-term loss training engine. Same discipline as train_policy.py, one
script, one config, one CSV row per run, and it reuses train_policy's model
builder, optimizer factory, device logic and logging so the two engines
cannot drift apart. Only the loss and the loader change, the command-match
term is joined by the TaSIL correction-law term (probed states ride along in
the same batch, rows without a stored target are masked out of this term
only) and by two certificate terms, squared projections of the command error
onto the per-row certificate directions, unit-normalized by default. Saves
the best-val checkpoint and a '-last' checkpoint so both can be flown.
"""

import os
import json
import time
import uuid
import numpy as np
import torch

import bc_data_aug
from train_policy import (ANCHOR, make_optimizer, build_model, pick_device,
                          mps_sanity_check, log_row, CKPT_DIR)

# the four-term additions to the anchor config
ANCHOR_4T = dict(ANCHOR)
ANCHOR_4T.update(
    lambda_ta=0.01,      # TaSIL correction-law weight, the paper's
                         # lambda_1=0.01 as the starting prior
    lambda_v=0.1,        # arrival-rate (V-dot) weight
    lambda_h=0.1,        # braking-law (H-dot) weight
    cert_norm='unit',    # 'unit' (projection, scale-free) | 'raw' (physical)
    tag='4term',
)


def resolve_config_4t():
    import argparse
    p = argparse.ArgumentParser(description="Train one 4-term policy (Ch 21).")
    p.add_argument('--config', type=str, default=None)
    for k, v in ANCHOR_4T.items():
        if isinstance(v, bool) or v is None:
            p.add_argument(f'--{k}', type=str, default=None)
        elif isinstance(v, int) and not isinstance(v, bool):
            p.add_argument(f'--{k}', type=int, default=None)
        elif isinstance(v, float):
            p.add_argument(f'--{k}', type=float, default=None)
        else:
            p.add_argument(f'--{k}', type=str, default=None)
    args = p.parse_args()
    cfg = dict(ANCHOR_4T)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    for k in ANCHOR_4T:
        v = getattr(args, k)
        if v is not None:
            if k == 'subset':
                cfg[k] = (str(v).lower() in ('true', '1', 'yes', 'subset'))
            elif isinstance(ANCHOR_4T[k], bool):
                cfg[k] = (str(v).lower() in ('true', '1', 'yes'))
            else:
                cfg[k] = v
    if cfg['subset'] is None:
        p.error("--subset is required (true = 25% sweep subset, false = full)")
    if cfg['arch'] == 'history':
        raise SystemExit("history arch: use the windowed sibling (later step).")
    return cfg


# ============================================================================
# The four-term loss on one batch. Returns (total, parts dict).
# ============================================================================
def four_term_loss(model, batch, cfg, device):
    x, y, gV, gH, dlt, dexp = [b.to(device, non_blocking=True) for b in batch]

    # one forward for base + probed states together (single graph, cheap)
    xp = x + dlt
    out = model(torch.cat([x, xp], dim=0), scaled=True)
    u_hat, u_hat_p = out[:x.shape[0]], out[x.shape[0]:]

    e = u_hat - y                                          # scaled error
    l_cmd = (e ** 2).mean()

    # TaSIL correction-law term, masked to rows that have a stored target
    mask = torch.isfinite(dexp[:, 0])
    if mask.any():
        d_net = u_hat_p[mask] - u_hat[mask]
        l_ta = ((d_net - dexp[mask]) ** 2).mean()
    else:
        l_ta = torch.zeros((), device=device)

    # certificate terms: squared projection of the command error onto the
    # certificate direction (unit-normalized by default; drift cancels, see
    # make_cert_targets.py)
    if cfg['cert_norm'] == 'unit':
        gVn = gV / gV.norm(dim=1, keepdim=True).clamp_min(1e-12)
        gHn = gH / gH.norm(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        gVn, gHn = gV, gH
    l_v = ((gVn * e).sum(dim=1) ** 2).mean()
    l_h = ((gHn * e).sum(dim=1) ** 2).mean()

    total = (l_cmd + cfg['lambda_ta'] * l_ta
             + cfg['lambda_v'] * l_v + cfg['lambda_h'] * l_h)
    parts = dict(cmd=l_cmd.item(), ta=l_ta.item(),
                 v=l_v.item(), h=l_h.item())
    return total, parts


def run_epoch_4t(model, loader, cfg, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    grad_clip = cfg['grad_clip']
    tot, n = 0.0, 0
    acc = dict(cmd=0.0, ta=0.0, v=0.0, h=0.0)
    for batch in loader:
        with torch.set_grad_enabled(train):
            loss, parts = four_term_loss(model, batch, cfg, device)
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


# ============================================================================
def train(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-4t-' + uuid.uuid4().hex[:6]

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    device = pick_device(cfg)
    if device.type == 'mps':
        forced = mps_sanity_check(cfg)
        if forced is not None:
            device = forced
    print(f"  device: {device}")

    model = build_model(cfg).to(device)
    import models
    n_params = models.count_params(model)
    print(f"  model: {cfg['arch']} d{cfg['depth']} w{cfg['width']} "
          f"({n_params:,} params)")
    print(f"  weights: lambda_ta={cfg['lambda_ta']}  lambda_v={cfg['lambda_v']}"
          f"  lambda_h={cfg['lambda_h']}  cert_norm={cfg['cert_norm']}")

    train_loader, val_loader, _ = bc_data_aug.make_loaders_aug(
        batch_size=cfg['batch'], subset=cfg['subset'])

    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg['plateau_factor'],
        patience=cfg['plateau_patience'])

    best_val, best_state, best_epoch, since = float('inf'), None, -1, 0
    t0 = time.time()
    for epoch in range(1, cfg['max_epochs'] + 1):
        tr, trp = run_epoch_4t(model, train_loader, cfg, device, optimizer)
        va, vap = run_epoch_4t(model, val_loader, cfg, device)
        scheduler.step(va)
        improved = va < best_val - 1e-9
        if improved:
            best_val, best_epoch, since = va, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
        print(f"    ep {epoch:3d}  train {tr:.6e}  val {va:.6e}  "
              f"[cmd {vap['cmd']:.2e} ta {vap['ta']:.2e} "
              f"v {vap['v']:.2e} h {vap['h']:.2e}]  "
              f"lr {optimizer.param_groups[0]['lr']:.1e} "
              f"{'*' if improved else ''}")
        if since >= cfg['early_stop_patience']:
            print(f"    early stop at epoch {epoch}")
            break
    wall = time.time() - t0

    # save best-val AND last, so flight selection can compare both
    last_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    ckpt = os.path.join(CKPT_DIR, f'{run_id}.pt')
    torch.save({'state_dict': best_state, 'config': cfg, 'best_val': best_val,
                'best_epoch': best_epoch, 'n_params': n_params,
                'run_id': run_id}, ckpt)
    torch.save({'state_dict': last_state, 'config': cfg,
                'run_id': run_id + '-last'}, ckpt.replace('.pt', '-last.pt'))
    log_row(run_id, cfg, best_val, best_epoch, wall, ckpt, n_params, str(device))

    print("  " + "-" * 56)
    print(f"  run {run_id}: best val {best_val:.6e} at ep {best_epoch}, "
          f"{wall/60:.1f} min")
    print(f"  saved -> {ckpt}  (+ -last.pt)")
    print("  NEXT: fly both checkpoints on the paired thousand; the flight,")
    print("  not this val loss, selects (Section 21.2).")
    return run_id, best_val, ckpt


if __name__ == "__main__":
    cfg = resolve_config_4t()
    print("=" * 62)
    print("TRAIN ONE FOUR-TERM POLICY (Chapter 21)")
    print("=" * 62)
    print("  config:", json.dumps({k: cfg[k] for k in
          ('arch', 'depth', 'width', 'lr', 'batch', 'subset', 'seed',
           'lambda_ta', 'lambda_v', 'lambda_h', 'cert_norm', 'tag')}))
    train(cfg)