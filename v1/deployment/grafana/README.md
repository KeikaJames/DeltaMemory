# Mneme Grafana Dashboard

This directory contains a pre-built Grafana dashboard for visualizing Mneme memory bank metrics.

## Import

1. Open Grafana (http://localhost:3000)
2. Navigate to **Dashboards → Import**
3. Upload `mneme_dashboard.json`
4. Select your Prometheus data source
5. Click **Import**

## Dashboard Panels

- **Bank Size Over Time**: Facts in memory bank per layer
- **Forward Latency Percentiles**: p50/p95/p99 latency of forward_with_bank
- **Hit Rate (1min rolling)**: Memory bank lookup success rate
- **Alpha Distribution**: Memory mixing coefficient over time
- **Eviction Rate**: Evictions per second
- **Cumulative Evictions**: Total evictions per layer

## Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mneme_bank_size` | Gauge | layer | Facts in bank |
| `mneme_bank_hit_rate_hits_total` | Counter | layer | Total hits |
| `mneme_bank_hit_rate_accesses_total` | Counter | layer | Total accesses |
| `mneme_alpha` | Gauge | layer | Mixing coefficient |
| `mneme_forward_latency_seconds` | Histogram | layer | Forward latency |
| `mneme_eviction_total` | Counter | layer | Total evictions |

## Prometheus Setup

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'mneme'
    static_configs:
      - targets: ['localhost:9464']
    scrape_interval: 15s
```

## Mneme Code Example

```python
from deltamemory.observability import BankMetrics, serve_http

metrics = BankMetrics()
serve_http(port=9464)

with metrics.forward_latency_timer(layer="layer_0"):
    output = forward_with_bank(...)

metrics.record_access(layer="layer_0")
metrics.record_hit(layer="layer_0")
metrics.set_bank_size(layer="layer_0", size=512)
```

See [Observability Guide](../../docs/observability.md) for full documentation.
