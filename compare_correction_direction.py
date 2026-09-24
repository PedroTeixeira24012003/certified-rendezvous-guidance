"""
compare_correction_direction.py
Fly the SAME ICs through two checkpoints on a rung and compare each flight's
final distance-from-hold. Distinguishes two hypotheses for the bias head's
effect, the outward-push hypothesis (the correction pushes finals
systematically away from center) and the variance-inflation hypothesis (it
adds symmetric scatter, some out, some in).

Reports, over all flights:
  frac_farther : fraction of flights where ckptB final is FARTHER from hold than ckptA
  mean d(dist) : mean change in distance-from-hold (B - A); >0 = net outward
  and per-axis mean |offset| for each, to see if any axis gained a SHIFT.
If frac_farther ~0.5 and mean d(dist)>0  -> variance-inflation hypothesis.
If frac_farther ~1.0                     -> outward-push hypothesis.
"""
import argparse, numpy as np
import robustness_ladder as RL, deploy_closedloop as DCL
import th_mpc_behind_1000run as EXP, bc_data, models

def fly(ckpt, rung, n):
    model, cfg, _ = DCL.load_checkpoint(ckpt)
    xm, xs = bc_data.load_norm_stats()
    refs = np.load(DCL.REFS_FILE); lc = RL.LADDER[rung]
    n = min(n, int(refs['n_mc']))
    x0, th, A, E = refs['mc_x0'], refs['mc_theta0'], refs['mc_a'], refs['mc_e']
    offs = np.zeros((n,3))
    for i in range(n):
        tp = RL.DisturbanceTape(rung, i, lc)
        st = np.concatenate([x0[i], [float(th[i])]]); Xh=[st.copy()]
        ih = isinstance(model, models.HistoryPolicy)
        if ih: fl,al=[],[]; kw=model.k
        div=False
        for _ in range(EXP.SIM_STEPS):
            Rr=np.linalg.norm(st[:3]); sm=st.copy(); sm[:6]=st[:6]+tp.nav_noise(Rr)
            u = DCL.policy_command_history(model,sm,float(A[i]),float(E[i]),xm,xs,fl,al,kw) if ih \
                else DCL.policy_command(model,sm,float(A[i]),float(E[i]),xm,xs)
            ua=tp.thrust_apply(u)
            st=RL.rk4_step_disturbed(st,ua,EXP.DT,float(A[i]),float(E[i]),tp.env_fn(float(A[i]),float(E[i])))
            Xh.append(st.copy())
            if not np.all(np.isfinite(st)) or np.linalg.norm(st[:3]-EXP.HOLD)>DCL.DIVERGE_DIST:
                div=True;break
        offs[i]=np.array(Xh)[-1,:3]-EXP.HOLD
    return offs

ap=argparse.ArgumentParser()
ap.add_argument('--ckptA',required=True); ap.add_argument('--ckptB',required=True)
ap.add_argument('--rung',default='L3'); ap.add_argument('--n',type=int,default=1000)
a=ap.parse_args()
print(f"A (baseline) = {a.ckptA}\nB (biashead) = {a.ckptB}\nrung {a.rung}, n {a.n}")
oA=fly(a.ckptA,a.rung,a.n); oB=fly(a.ckptB,a.rung,a.n)
dA=np.linalg.norm(oA,axis=1); dB=np.linalg.norm(oB,axis=1)
frac_farther=float((dB>dA).mean())
print(f"\n  mean dist-from-hold: A {dA.mean():.3f}  B {dB.mean():.3f}")
print(f"  frac flights B FARTHER than A : {frac_farther:.3f}")
print(f"  mean d(dist) (B-A)            : {(dB-dA).mean():+.3f}  (>0 net outward)")
print(f"  --- per-axis mean offset (shift check) ---")
for j,nm in enumerate('xyz'):
    print(f"    {nm}: A mean {oA[:,j].mean():+.3f}  B mean {oB[:,j].mean():+.3f}  "
          f"| A std {oA[:,j].std():.3f}  B std {oB[:,j].std():.3f}")
print("\n  VERDICT:")
if frac_farther>0.85:
    print("   ~all flights farther -> SYSTEMATIC OUTWARD push (outward-push hypothesis).")
elif 0.4<frac_farther<0.65:
    print("   ~half farther half closer -> symmetric VARIANCE inflation (not a")
    print("   directional push). Fix = lower weight / gating, not a sign flip.")
else:
    print(f"   frac_farther={frac_farther:.2f} -> intermediate; inspect per-axis above.")