"""
make_split.py
One-time trajectory-level split of the dataset into train, validation and
test sets, 8000/1000/1000 with a fixed seed. Saves the trajectory indices
to split_indices.npz, which every later script reads, and refuses to
overwrite an existing split.
"""

import os
import sys
import numpy as np

# split constants
N_TRAJ   = 10000
N_TRAIN  = 8000
N_VAL    = 1000
N_TEST   = 1000
SEED     = 42
OUTFILE  = 'split_indices.npz'

# refuse to overwrite an existing split
if os.path.exists(OUTFILE):
    print(f"ERROR: {OUTFILE} already exists.")
    print("The split is made once and never regenerated. Every experiment so far")
    print("depends on it. If you truly intend to rebuild it (invalidating all")
    print("previous results), delete the file manually and rerun.")
    sys.exit(1)

assert N_TRAIN + N_VAL + N_TEST == N_TRAJ, "split sizes must sum to the trajectory count"

# shuffle trajectory indices with the fixed seed and assign whole trajectories
rng = np.random.default_rng(SEED)
perm = rng.permutation(N_TRAJ)

train_traj = np.sort(perm[:N_TRAIN])
val_traj   = np.sort(perm[N_TRAIN:N_TRAIN + N_VAL])
test_traj  = np.sort(perm[N_TRAIN + N_VAL:])

# sanity checks before writing anything
all_back = np.concatenate([train_traj, val_traj, test_traj])
assert all_back.shape[0] == N_TRAJ
assert np.array_equal(np.sort(all_back), np.arange(N_TRAJ)), \
    "sets must partition 0..9999 exactly (no overlap, no gap)"
assert len(np.intersect1d(train_traj, val_traj)) == 0
assert len(np.intersect1d(train_traj, test_traj)) == 0
assert len(np.intersect1d(val_traj, test_traj)) == 0

np.savez(OUTFILE,
         train_traj=train_traj.astype(np.int64),
         val_traj=val_traj.astype(np.int64),
         test_traj=test_traj.astype(np.int64),
         seed=SEED,
         n_traj=N_TRAJ)

print("=" * 62)
print("TRAJECTORY-LEVEL SPLIT WRITTEN (Stage 0.1)")
print("=" * 62)
print(f"  seed                : {SEED}")
print(f"  train trajectories  : {train_traj.shape[0]}  "
      f"({train_traj.shape[0] * 400:,} pairs)")
print(f"  val trajectories    : {val_traj.shape[0]}  "
      f"({val_traj.shape[0] * 400:,} pairs)")
print(f"  test trajectories   : {test_traj.shape[0]}  "
      f"({test_traj.shape[0] * 400:,} pairs)")
print(f"  file                : {OUTFILE}")
print(f"  first 5 of each     : train {train_traj[:5].tolist()}")
print(f"                        val   {val_traj[:5].tolist()}")
print(f"                        test  {test_traj[:5].tolist()}")
print("=" * 62)
print("  This file is now the single source of truth for the split.")
print("  Do not regenerate it.")