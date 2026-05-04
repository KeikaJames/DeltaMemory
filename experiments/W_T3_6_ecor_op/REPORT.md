# W-T3.6 — ECOR operator-level ablation

**Model**: `Qwen/Qwen2.5-0.5B-Instruct`  
**Device / dtype**: mps / torch.bfloat16  
**Git rev**: `57aeeac1094440b9d7f9773c96528ba1d0a250b4`  
**Layers profiled**: 24  
**Neutral prompts**: 8  
**Total cells**: 32  (ortho × soft_blend × α = 2×4×4)  
**Elapsed**: 7.0s  
**Redline `soft_blend=0` ≡ additive**: PASS  

## What this answers

Why is orthogonal projection (`LOPIConfig.orthogonal`) still default `False`,
and what does the ECOR `soft_blend` knob actually do at the operator level?

* **`ortho=False`** ⇒ M_perp := M_V (raw bank V-readout, no projection).
* **`ortho=True`** ⇒ M_perp := M_V − proj_V_ctx(M_V) (legacy LOPI step 1).
* **`soft_blend=0`** ⇒ ECOR disabled, output is `V_ctx + α·M_perp` (additive).
* **`soft_blend=1`** ⇒ pure ECOR rotation (norm-preserving by construction).
* **`soft_blend ∈ (0,1)`** ⇒ linear blend of additive and rotated outputs.

Metrics are mean over (prompts × layers):

* `rel_perturb` = ‖V_out − V_ctx‖ / ‖V_ctx‖  — perturbation magnitude (smaller = gentler).
* `cos_v_out_v_ctx` — direction preservation (1.0 = unchanged direction).
* `norm_ratio` = ‖V_out‖ / ‖V_ctx‖ — energy preservation (ECOR target = 1.0).
* `m_perp_ratio` = ‖M_perp‖ / ‖M_V‖ — how aggressive the ortho projection is (ortho=False ⇒ 1.0).

## α = 0.5

| ortho | soft_blend | rel_perturb | cos(V_out,V_ctx) | ‖V_out‖/‖V_ctx‖ | ‖M_⊥‖/‖M_V‖ |
|:-----:|:----------:|------------:|-----------------:|----------------:|-------------:|
| False | 0.00 | 2.2126 | 0.5775 | 2.6590 | 1.0000 |
| False | 0.25 | 1.7687 | 0.6158 | 2.2290 | 1.0000 |
| False | 0.50 | 1.3270 | 0.6733 | 1.8110 | 1.0000 |
| False | 1.00 | 0.4494 | 0.9125 | 1.0927 | 1.0000 |
|  True | 0.00 | 2.1842 | 0.5233 | 2.5384 | 0.9081 |
|  True | 0.25 | 1.7550 | 0.5642 | 2.1093 | 0.9081 |
|  True | 0.50 | 1.3265 | 0.6253 | 1.6949 | 0.9081 |
|  True | 1.00 | 0.4785 | 0.8865 | 1.0000 | 0.9081 |

## α = 1.0

| ortho | soft_blend | rel_perturb | cos(V_out,V_ctx) | ‖V_out‖/‖V_ctx‖ | ‖M_⊥‖/‖M_V‖ |
|:-----:|:----------:|------------:|-----------------:|----------------:|-------------:|
| False | 0.00 | 4.4260 | 0.4587 | 4.7954 | 1.0000 |
| False | 0.25 | 3.4797 | 0.4775 | 3.8459 | 1.0000 |
| False | 0.50 | 2.5398 | 0.5091 | 2.9047 | 1.0000 |
| False | 1.00 | 0.6983 | 0.7788 | 1.1093 | 1.0000 |
|  True | 0.00 | 4.3684 | 0.3813 | 4.6188 | 0.9081 |
|  True | 0.25 | 3.4587 | 0.3962 | 3.6814 | 0.9081 |
|  True | 0.50 | 2.5505 | 0.4224 | 2.7515 | 0.9081 |
|  True | 1.00 | 0.7752 | 0.6994 | 1.0000 | 0.9081 |

## α = 2.0

| ortho | soft_blend | rel_perturb | cos(V_out,V_ctx) | ‖V_out‖/‖V_ctx‖ | ‖M_⊥‖/‖M_V‖ |
|:-----:|:----------:|------------:|-----------------:|----------------:|-------------:|
| False | 0.00 | 8.8521 | 0.3764 | 9.1718 | 1.0000 |
| False | 0.25 | 6.8163 | 0.3850 | 7.1316 | 1.0000 |
| False | 0.50 | 4.7982 | 0.4008 | 5.1001 | 1.0000 |
| False | 1.00 | 0.8453 | 0.6618 | 1.0992 | 1.0000 |
|  True | 0.00 | 8.7367 | 0.2678 | 8.9069 | 0.9081 |
|  True | 0.25 | 6.7715 | 0.2716 | 6.9026 | 0.9081 |
|  True | 0.50 | 4.8021 | 0.2789 | 4.9094 | 0.9081 |
|  True | 1.00 | 0.9633 | 0.5361 | 0.9998 | 0.9081 |

## α = 4.0

| ortho | soft_blend | rel_perturb | cos(V_out,V_ctx) | ‖V_out‖/‖V_ctx‖ | ‖M_⊥‖/‖M_V‖ |
|:-----:|:----------:|------------:|-----------------:|----------------:|-------------:|
| False | 0.00 | 17.7043 | 0.3208 | 17.9910 | 1.0000 |
| False | 0.25 | 13.4647 | 0.3272 | 13.7488 | 1.0000 |
| False | 0.50 | 9.2242 | 0.3387 | 9.5066 | 1.0000 |
| False | 1.00 | 0.8744 | 0.6358 | 1.0974 | 1.0000 |
|  True | 0.00 | 17.4730 | 0.1775 | 17.5870 | 0.9081 |
|  True | 0.25 | 13.3395 | 0.1829 | 13.4136 | 0.9081 |
|  True | 0.50 | 9.1792 | 0.1907 | 9.2435 | 0.9081 |
|  True | 1.00 | 1.0002 | 0.4997 | 1.0006 | 0.9081 |

## Takeaways

* **Norm preservation**: pure ECOR (soft_blend=1) mean ‖V_out‖/‖V_ctx‖ = 1.0499 vs additive (soft_blend=0) 8.5335. ECOR designed target is 1.0; deviation reveals scale of M_perp/V_ctx mismatch.
* **Ortho aggression**: ortho=True keeps 0.9081 of ‖M_V‖ as M_⊥ (vs 1.0 when ortho=False). Smaller value ⇒ more of M_V was already aligned with V_ctx (i.e. the projection threw away that fraction of the bank signal).
* **rel_perturb vs α** (ortho=False, blend=0.0): α=0.5: 2.213, α=1.0: 4.426, α=2.0: 8.852, α=4.0: 17.704
* **rel_perturb vs α** (ortho=False, blend=1.0): α=0.5: 0.449, α=1.0: 0.698, α=2.0: 0.845, α=4.0: 0.874
* **rel_perturb vs α** (ortho=True, blend=0.0): α=0.5: 2.184, α=1.0: 4.368, α=2.0: 8.737, α=4.0: 17.473
* **rel_perturb vs α** (ortho=True, blend=1.0): α=0.5: 0.478, α=1.0: 0.775, α=2.0: 0.963, α=4.0: 1.000

**Caveat**: this is operator-level (single forward, frozen V_ctx tensors); downstream NLL/drift impact requires wiring ECOR through `attn_native_bank.py` (deferred — see plan N5 / W-T3 round 1).
