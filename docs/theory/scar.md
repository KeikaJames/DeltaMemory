# SCAR: Steering with Contrastive Activation Rotation

SCAR is the proposed M.3 replacement candidate after the W.3/W.4 failure diagnosis. It keeps the useful part of CAA--zero-training contrastive steering in activation space--but removes two brittle assumptions: one mean-difference vector is enough, and one heuristic layer is correct. The design goal is still frozen-weight, hook-based inference with exact alpha-zero identity.

## Method

For each selected layer l, collect N positive and N negative attention-output or residual activations on matched calibration prompts. Let

`X_l = A_l^+ - A_l^- in R^{N x d}`

where each row is a paired positive-minus-negative activation contrast. Compute a thin SVD:

`X_l = U Sigma V^T`.

The top-k right singular vectors form an orthonormal steering basis. In implementation terms, store `B_l in R^{k x d}` with rows `v_1, ..., v_k`, so `B_l B_l^T = I_k` and the residual-space projector is

`P_l = B_l^T B_l`.

At inference, for activation `a` and target activation `t` (for example the positive calibration mean, a per-layer fact target, or the current bank-derived target), inject

`a' = a + alpha * P_l (t - a)`.

Equivalently, SCAR moves only the component of `a` that lies in the contrastive subspace. Components orthogonal to the learned basis are unchanged. For `0 <= alpha <= 1`, this is a partial rotation/relaxation toward the target inside the contrastive plane; for small alpha it reduces to a controlled activation edit, but unlike raw CAA it does not add unconstrained off-subspace energy.

## Why SCAR should beat CAA

CAA uses one vector, usually `mean_pos - mean_neg`, then broadcasts `a' = a + alpha s` at a chosen layer. If the mean contrast is dominated by prompt noise, dataset imbalance, or a layer where the concept is linearly visible but causally harmful, the whole edit is sign-fragile. W.4's 5,041-cell aggregate is the warning: CAA was Holm-significant at alpha >= 1 in all 12 CAA-vs-none tests, but 11/12 significant effects increased drift.

SCAR changes the geometry. The orthogonal projector `P = B^T B` is symmetric and idempotent. Its operator norm is 1, so

`||P x||_2 <= ||x||_2`.

Thus the injected displacement is bounded by `|alpha| * ||t - a||` in the selected subspace and cannot expand components outside that subspace. CAA's single additive vector can increase residual norm in arbitrary directions at every token; SCAR can only move along the top-k principal axes supported by paired activation contrasts. Using `k = 2..4` also makes the method less sensitive to one noisy mean direction while staying low-rank enough for interpretability.

The method is closer to RepE's PCA view of representation directions than to raw ActAdd: it models a contrastive subspace, not a single arrow. It also inherits ITI's lesson that the intervention site should be selected empirically, not by a universal middle-layer prior.

## Layer and hyperparameter plan

SCAR should use LOPI-derived profiling infrastructure only as a candidate generator. The actual layer subset should be chosen by measured per-layer Delta-NLL ranks on a small validation panel: evaluate candidate layers, rank by neutral drift reduction and counter-prior lift, and select the top layer or top few layers on a Pareto rule. This directly targets the sign error identified in W.4.

Initial hyperparameters:

| Hyperparameter | Initial range | Rationale |
|---|---:|---|
| N calibration pairs | 16, 32, 64 | 16 matches W.4 CAA; larger N stabilizes SVD. |
| Rank k | 1, 2, 4 | k=1 recovers a normalized CAA-like subspace; k=2..4 tests noise robustness. |
| Alpha | 0, 0.25, 0.5, 1, 2 | Avoid alpha=4/8 until the non-expansive claim is validated empirically. |
| Layers | top 1, top 3 by Delta-NLL rank | Replaces `mu_arch`/`L//2` priors with measured effect. |
| Target t | positive mean, bank target, blended target | Ablate whether SCAR is pure steering or bank-conditioned steering. |

Alpha-zero bit equality is mathematically immediate: the edit term is multiplied by alpha, so at `alpha = 0`, `a' = a` exactly. The implementation should still short-circuit at alpha zero to avoid allocation or dtype/device casts, matching the existing redline discipline.

## Prediction on the W.4 grid

SCAR should win most clearly where CAA was significantly harmful because of off-axis or wrong-layer steering: Qwen2.5-1.5B at alpha 1-8 and GPT-2-medium at alpha 1-8. On Qwen2.5-0.5B alpha=1, where CAA already produced an approximately 19% reduction, SCAR should preserve or modestly improve the win if the selected subspace captures the same helpful axis. At alpha >= 2 on Qwen2.5-0.5B, the projection constraint should reduce overshoot relative to CAA's large positive median differences. SCAR may not beat raw `none` when the calibration target is semantically unrelated to the neutral drift panel; this is why per-layer Delta-NLL ranking is part of the method, not a later tuning luxury.

## M.3 unit-test plan

| Test | Expected result |
|---|---|
| alpha=0 bit-equal over at least 256 tokens | max-abs-diff = 0 |
| SVD basis shape | `B.shape == (k, d)` and k is clipped to rank/data size |
| Projection idempotence | `P(P(x)) == P(x)` within dtype tolerance |
| CPU/CUDA equivalence | same basis/projection within tolerance; skip if CUDA unavailable |
| Empty bank/calibration basis | no-op, no hook mutation |
| Layer hook path, Gemma-style | attaches to the intended decoder module path |
