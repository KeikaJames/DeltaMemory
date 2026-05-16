# e04: ACT-style Halt-Head Training + K_max Sweep

## Goal
Train a per-position halt head that decides whether to "pause" (write to bank) at each position during round 1. Each pause adds K tokens to the bank, increasing quadratic attention cost. We apply an ACT-style sparsity penalty to learn sparse, useful pauses.

## Architecture
- **Halt Head**: `Linear(hidden_size, 1)` taking round-1 hidden state at bank_layer → halt_logit
- **Halt Decision**: `sigmoid(halt_logit) > 0.5` OR sampled Bernoulli during training
- **K_max Budget**: If more than K_max positions fire halt, keep only top-K by halt_prob

## Training Objective
```
L_total = L_NLL(round_2) + λ_act * mean(halt_prob)
```
- Higher λ_act → fewer halts → smaller effective K
- Sweep λ_act ∈ {0.0, 0.01, 0.1, 0.5, 1.0}
- Sweep K_max ∈ {1, 4, 16, 64, 256, -1} (-1 = unlimited)

## Grid
30 cells = 5 λ_act × 6 K_max

Each cell:
- Train 200 steps, lr=2e-4
- Canonical Exp35b data (n_train=120, n_preload=512, layer=9)
- Reports: Δ NLL, mean halts/sample, effective K

## Pass Criterion
At least one (λ_act, K_max) cell achieves:
- mean halts ≤ 4 per sample
- Δ NLL ≤ -2.0

This proves the model can learn to be sparse without losing signal.

## Usage
```bash
python v2/experiments/e04_act_halt/run.py \
    --device mps \
    --seed 0 \
    --steps 200 \
    --lr 2e-4 \
    --rank 64 \
    --bank_layer 9 \
    --lam_grid "0.0,0.01,0.1,0.5,1.0" \
    --kmax_grid "1,4,16,64,256,-1"
```

## Output
- Per-cell: `cells/lam{λ}_kmax{K}_seed{S}.json`
- Summary: `e04_summary_seed{S}.json`

## Implementation Notes
The current LPL patch uses `pause_heads` that operate at the layer level. This experiment monkey-patches the bank.write logic via `force_pause_mask` callable that:
1. Computes `halt_prob = sigmoid(halt_head(h_in))` at bank_layer during round 1
2. Applies K_max budget by keeping only top-K positions
3. Returns boolean pause mask for the layer forward
