
# Machine Learning for Surrogate Optimal Rendezvous Guidance

Code for my MSc thesis at Cranfield University, 2026. An NMPC expert flies
a near-GEO proximity approach, a neural network is trained to clone it, a
CBF-CLF safety filter certifies its commands at every step, and the whole
stack is stress tested on a six rung disturbance ladder against the expert.

## Layout

All scripts live in one flat directory because they import each other by
name. The thesis appendix on code availability describes every file. The
pipeline runs in five stages, each importing the earlier ones unchanged.

1. Expert. th_mpc_expert_behind*.py and the 1000 flight validation run.
2. Dataset. generate_dataset_behind.py, make_split.py, bc_data.py.
3. Policies. models.py, train_policy.py, train_history.py, the sweep and
   evaluation scripts.
4. Safety filter. cbf_filter.py and its verification suite.
5. Robustness. robustness_ladder.py, the four term training scripts, the
   memory study, fly_finale.py.

## Requirements

Python 3.10 or newer. numpy, scipy, matplotlib, torch, osqp and casadi
install with pip. acados is built from source, follow the install guide at
docs.acados.org, the expert scripts need it.

## Data

The dataset (4 million rows), the trained checkpoints and the campaign
outputs are too large for the repository. Every random draw in the code is
seeded, so all of them regenerate from these scripts, and they are
available on request.
