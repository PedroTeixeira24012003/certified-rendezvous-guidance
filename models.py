"""
models.py
The three policy architectures, an MLP, an encoder-decoder over orbit context,
and a history transformer, all sharing one output head. Every model takes the
10 normalized features and returns a command inside the thrust box, in m/s^2
by default or in the scaled [-1, 1] training space with scaled=True. Run
directly for a self-test.
"""

import torch
import torch.nn as nn

U_MAX = 0.082          # per-axis thrust limit [m/s^2], must match bc_data.py
N_IN  = 10             # input features
N_OUT = 3              # command components


# ============================================================================
# Activation factory, a string in a config selects the nonlinearity
# ============================================================================
def make_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'silu':
        return nn.SiLU()
    raise ValueError(f"unknown activation '{name}' (use relu, tanh, or silu)")


# ============================================================================
# Shared output head. Turns a raw 3-vector into a command, in scaled [-1, 1]
# or physical m/s^2, so training and deployment go through the same maths.
# ============================================================================
class Head(nn.Module):
    def __init__(self, kind='tanh'):
        super().__init__()
        kind = kind.lower()
        if kind not in ('tanh', 'linear'):
            raise ValueError("head must be 'tanh' or 'linear'")
        self.kind = kind

    def forward(self, z, scaled=False):
        """z : (..., 3) raw pre-head values.
        scaled=True  -> value in [-1,1] (training space, = u / U_MAX)
        scaled=False -> command in m/s^2 (deployment space)"""
        if self.kind == 'tanh':
            s = torch.tanh(z)                      # in (-1, 1)
        else:  # linear head, clip to the box in scaled units
            s = torch.clamp(z, -1.0, 1.0)          # in [-1, 1]
        return s if scaled else s * U_MAX


# ============================================================================
# MLP, configurable depth, width, activation and head
# ============================================================================
class MLPPolicy(nn.Module):
    def __init__(self, depth=3, width=256, activation='relu', head='tanh',
                 n_in=N_IN, n_out=N_OUT):
        super().__init__()
        assert depth >= 1, "depth is the number of hidden layers, >= 1"
        layers = []
        d_prev = n_in
        for _ in range(depth):
            layers.append(nn.Linear(d_prev, width))
            layers.append(make_activation(activation))
            d_prev = width
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(d_prev, n_out)
        self.head = Head(head)

    def forward(self, x, scaled=False):
        z = self.out(self.trunk(x))
        return self.head(z, scaled=scaled)


# ============================================================================
# Encoder-decoder. A small context encoder compresses the orbit descriptors
# (sin th, cos th, a, e) into a latent that conditions the state trunk.
#
# conditioning='concat' : latent concatenated with the state (default)
# conditioning='film'   : latent produces per-feature scale and shift applied
#                         to the first trunk layer's activation
#
# Feature index convention, matching bc_data.py FEATURE_NAMES:
#   0..2 pos, 3..5 vel, 6 sin th, 7 cos th, 8 a, 9 e
# ============================================================================
class EncDecPolicy(nn.Module):
    def __init__(self, latent=16, enc_width=64, trunk_width=256, trunk_depth=3,
                 activation='relu', head='tanh', conditioning='concat'):
        super().__init__()
        self.conditioning = conditioning.lower()
        self.state_idx = [0, 1, 2, 3, 4, 5]
        self.ctx_idx   = [6, 7, 8, 9]
        n_state = len(self.state_idx)
        n_ctx   = len(self.ctx_idx)

        # context encoder, (4) -> latent
        self.encoder = nn.Sequential(
            nn.Linear(n_ctx, enc_width), make_activation(activation),
            nn.Linear(enc_width, latent),
        )

        if self.conditioning == 'concat':
            trunk_in = n_state + latent
            self.film = None
        elif self.conditioning == 'film':
            trunk_in = n_state
            # latent -> (scale, shift) for the first trunk-hidden activation
            self.film = nn.Linear(latent, 2 * trunk_width)
        else:
            raise ValueError("conditioning must be 'concat' or 'film'")

        # trunk, conditioned state -> command
        assert trunk_depth >= 1
        self.first = nn.Linear(trunk_in, trunk_width)
        self.first_act = make_activation(activation)
        rest = []
        for _ in range(trunk_depth - 1):
            rest.append(nn.Linear(trunk_width, trunk_width))
            rest.append(make_activation(activation))
        self.rest = nn.Sequential(*rest)
        self.out = nn.Linear(trunk_width, N_OUT)
        self.head = Head(head)

    def forward(self, x, scaled=False):
        state = x[..., self.state_idx]
        ctx   = x[..., self.ctx_idx]
        z = self.encoder(ctx)                       # latent

        if self.conditioning == 'concat':
            h = torch.cat([state, z], dim=-1)
            h = self.first_act(self.first(h))
        else:  # film
            h = self.first(state)                   # pre-activation
            gamma_beta = self.film(z)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = self.first_act(gamma * h + beta)    # feature-wise modulation

        h = self.rest(h)
        return self.head(self.out(h), scaled=scaled)


# ============================================================================
# History transformer. Embeds the last k steps of (10 features + 3 actions),
# runs a small transformer encoder, mean-pools over the window, concatenates
# the pooled summary with the current features and decodes a command.
#
# forward() takes
#     x_hist : (batch, k, 13)   [10 features ++ 3 actions] per past step
#     x_now  : (batch, 10)      the current normalized features
# bc_data.py's windowed loader builds x_hist with the boundary and padding
# conventions.
# ============================================================================
class HistoryPolicy(nn.Module):
    def __init__(self, k=16, d_model=64, n_heads=4, n_layers=2,
                 dec_width=256, activation='relu', head='tanh',
                 n_feat=N_IN, n_act=N_OUT):
        super().__init__()
        self.k = k
        self.d_model = d_model
        step_dim = n_feat + n_act                   # 13

        self.embed = nn.Linear(step_dim, d_model)
        # learned positional encoding over the k window positions
        self.pos = nn.Parameter(torch.zeros(1, k, d_model))
        nn.init.normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, batch_first=True,
            activation='gelu', dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # decoder, [pooled window summary ++ current features] -> command
        self.dec = nn.Sequential(
            nn.Linear(d_model + n_feat, dec_width), make_activation(activation),
            nn.Linear(dec_width, dec_width), make_activation(activation),
        )
        self.out = nn.Linear(dec_width, N_OUT)
        self.head = Head(head)

    def forward(self, x_hist, x_now, scaled=False):
        # x_hist: (B, k, 13)   x_now: (B, 10)
        h = self.embed(x_hist) + self.pos           # (B, k, d_model)
        h = self.encoder(h)                         # (B, k, d_model)
        pooled = h.mean(dim=1)                      # (B, d_model)
        z = torch.cat([pooled, x_now], dim=-1)      # (B, d_model + 10)
        return self.head(self.out(self.dec(z)), scaled=scaled)


# ============================================================================
# Parameter count utility
# ============================================================================
def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# Self-test, shapes, head behaviour, box guarantee, gradients
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 32
    x  = torch.randn(B, N_IN)
    print("=" * 62)
    print("models.py SELF-TEST")
    print("=" * 62)

    # MLP, shapes plus both output spaces
    m = MLPPolicy(depth=3, width=256, activation='relu', head='tanh')
    u_phys = m(x)                       # m/s^2
    u_scal = m(x, scaled=True)          # [-1,1]
    assert u_phys.shape == (B, 3) and u_scal.shape == (B, 3)
    assert torch.allclose(u_phys, u_scal * U_MAX, atol=1e-6)
    print(f"  MLP shapes + scaled/physical consistency : OK  ({count_params(m):,} params)")

    # box guarantee, the tanh head can never exceed U_MAX
    x_big = torch.randn(B, N_IN) * 1e3
    u_big = m(x_big)
    assert u_big.abs().max() <= U_MAX + 1e-6, "tanh head must respect the box"
    print(f"  tanh head box guarantee (|u|<=U_MAX)     : OK  (max |u| = {u_big.abs().max():.5f})")

    # linear head clips to the box
    m_lin = MLPPolicy(head='linear')
    u_lin = m_lin(x_big)
    assert u_lin.abs().max() <= U_MAX + 1e-6
    print(f"  linear head clips to the box             : OK  (max |u| = {u_lin.abs().max():.5f})")

    # activation and depth/width variations build and run
    for act in ('relu', 'tanh', 'silu'):
        for d, w in ((2, 128), (3, 256), (4, 512)):
            mm = MLPPolicy(depth=d, width=w, activation=act)
            _ = mm(x)
    print("  all depth/width/activation combos build  : OK")

    # encoder-decoder, concat and film
    for cond in ('concat', 'film'):
        ed = EncDecPolicy(conditioning=cond)
        u = ed(x)
        assert u.shape == (B, 3) and u.abs().max() <= U_MAX + 1e-6
        print(f"  EncDec ({cond}) shape + box             : OK  ({count_params(ed):,} params)")

    # history transformer, window forward and box, including k=1
    for k in (16, 1):
        hp = HistoryPolicy(k=k)
        x_hist = torch.randn(B, k, N_IN + N_OUT)
        x_now  = torch.randn(B, N_IN)
        u = hp(x_hist, x_now)
        assert u.shape == (B, 3) and u.abs().max() <= U_MAX + 1e-6
        print(f"  HistoryPolicy k={k:<2d} shape + box          : OK  ({count_params(hp):,} params)")

    # gradients flow at a near-saturation target via the scaled path
    m.zero_grad()
    target = torch.full((B, 3), 0.999)      # near-saturation scaled target
    pred_scaled = m(x, scaled=True)
    loss = ((pred_scaled - target) ** 2).mean()
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    assert torch.isfinite(g).all() and g.abs().sum() > 0, "gradients must flow"
    print(f"  gradients flow at near-saturation        : OK  (grad norm {g.norm():.3e})")

    print("=" * 62)
    print("  ALL MODEL TESTS PASS")
    print("=" * 62)