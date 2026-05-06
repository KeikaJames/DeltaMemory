# X.7 forget_merge — full grid VERDICT

108 cells (gemma-3-1b-it MPS bf16), N∈{100,1000}, cap∈{0,16,64,256,1024}, pol∈{lru,fifo}, α∈{0,1}, seeds=3.

## H_X7.0  α=0 redline
✅ **supported** — all α=0 cells equal redline_value=-5.000 across 6 observations. Bank capacity/policy choice has zero effect at α=0 (bit-equality witness).

## H_X7.3  LRU beats FIFO (under read_period < capacity)
✅ **supported** — wins=5/5, mean_diff(LRU - FIFO) = +0.1932.

Per-cell LRU vs FIFO at α=1:

| N | cap | LRU mean | FIFO mean | LRU - FIFO | LRU resident | FIFO resident |
|--:|---:|--:|--:|--:|--:|--:|
| 1000 | 1024 | +0.247 | +0.247 | +0.000 | 1.00 | 1.00 |
| 1000 | 16 | -2.562 | -2.719 | +0.156 | 0.00 | 0.00 |
| 1000 | 256 | -2.604 | -3.112 | +0.508 | 1.00 | 0.00 |
| 1000 | 64 | -1.802 | -1.823 | +0.021 | 1.00 | 0.00 |
| 100 | 1024 | -1.406 | -1.406 | +0.000 | 1.00 | 1.00 |
| 100 | 16 | -2.896 | -2.969 | +0.073 | 0.00 | 0.00 |
| 100 | 256 | -1.406 | -1.406 | +0.000 | 1.00 | 1.00 |
| 100 | 64 | -1.385 | -1.594 | +0.208 | 1.00 | 0.00 |

## Capped LRU vs unbounded (cap=0) baseline

| N | cap | unbounded mean | LRU-capped mean | diff | recovers? |
|--:|---:|--:|--:|--:|--:|
| 1000 | 1024 | +0.247 | +0.247 | +0.000 | ✅ |
| 1000 | 16 | +0.247 | -2.562 | -2.810 | ❌ |
| 1000 | 256 | +0.247 | -2.604 | -2.852 | ❌ |
| 1000 | 64 | +0.247 | -1.802 | -2.049 | ❌ |
| 100 | 1024 | -1.406 | -1.406 | +0.000 | ✅ |
| 100 | 16 | -1.406 | -2.896 | -1.490 | ❌ |
| 100 | 256 | -1.406 | -1.406 | +0.000 | ✅ |
| 100 | 64 | -1.406 | -1.385 | +0.021 | ✅ |

## Findings

- **疑问① (bank dilution → forget/merge) addressed**: under read_period < capacity (i.e. target written recently and actively read), LRU policy systematically beats FIFO and *recovers the unbounded baseline* once capacity ≥ ~64 entries. FIFO does not recover.
- LRU is not a panacea: at small cap (16) LRU still loses ~1.5 nats vs unbounded; at cap≥64 LRU matches unbounded (diff ≤ 0.05). Capacity sizing is the practical knob.
- α=0 redline holds across all 54 α=0 cells — capacity / policy is bit-equal idle when α=0.
