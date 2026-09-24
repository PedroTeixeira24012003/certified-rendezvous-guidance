"""
plot_training_curve.py
Parse a training stdout log (captured with `| tee train_full.log`) and plot the
train and validation loss versus epoch, to confirm the run plateaued. Works with
the per-epoch lines printed by train_policy_4term.py / train_history*.py, of the
form:
    ep  NN  train X.XXe-XX  val X.XXe-XX  [ ... ]  lr X.Xe-XX [*]
Robust to extra columns (per-term components, star markers).

Usage:
  python3 plot_training_curve.py --log train_full.log [--out curve.png] [--logy]
"""
import argparse, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--log', required=True)
ap.add_argument('--out', default='training_curve.png')
ap.add_argument('--logy', action='store_true', help='log scale on the loss axis')
a = ap.parse_args()

ep_re    = re.compile(r'\bep\s+(\d+)\b')
train_re = re.compile(r'\btrain\s+([0-9.eE+-]+)')
val_re   = re.compile(r'\bval(?:_cmd)?\s+([0-9.eE+-]+)')

eps, tr, vl = [], [], []
with open(a.log) as f:
    for line in f:
        m = ep_re.search(line)
        if not m:
            continue
        mt, mv = train_re.search(line), val_re.search(line)
        if not (mt and mv):
            continue
        try:
            e = int(m.group(1)); t = float(mt.group(1)); v = float(mv.group(1))
        except ValueError:
            continue
        eps.append(e); tr.append(t); vl.append(v)

if not eps:
    raise SystemExit("no epoch lines parsed -- check the log format")

eps = np.array(eps); tr = np.array(tr); vl = np.array(vl)
best_i = int(np.argmin(vl))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(eps, tr, label='Train', lw=1.5)
ax.plot(eps, vl, label='Validation', lw=1.5)
ax.scatter([eps[best_i]], [vl[best_i]], c='crimson', zorder=5,
           label='Best validation loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
if a.logy:
    ax.set_yscale('log')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(a.out, dpi=140)
print(f"parsed {len(eps)} epochs")
print(f"best val {vl[best_i]:.6e} at epoch {eps[best_i]}")
print(f"final val {vl[-1]:.6e} at epoch {eps[-1]}")
if len(vl) >= 15:
    tail = vl[-15:]
    rel = (tail.max() - tail.min()) / tail.min()
    print(f"last 15 epochs val spread: {rel*100:.2f}%  "
          f"({'PLATEAUED' if rel < 0.1 else 'still moving'})")
print(f"saved -> {a.out}")