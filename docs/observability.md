# Mneme Observability

Comprehensive observability for Mneme memory bank operations using Prometheus metrics and OpenTelemetry tracing.

## Installation

```bash
pip install "deltamemory[obs]"
```

## Prometheus Metrics

### Quick Start

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

### Metrics

- **`mneme_bank_size`** (Gauge): Number of facts in memory bank
- **`mneme_bank_hit_rate_hits_total`** (Counter): Total memory hits
- **`mneme_bank_hit_rate_accesses_total`** (Counter): Total memory accesses
- **`mneme_alpha`** (Gauge): Memory mixing coefficient (α)
- **`mneme_forward_latency_seconds`** (Histogram): Forward call latency (buckets: 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s)
- **`mneme_eviction_total`** (Counter): Total evictions

### Using InstrumentedBank

```python
from deltamemory.observability import InstrumentedBank

metrics = BankMetrics()
instrumented_bank = InstrumentedBank(bank, metrics, layer="layer_0")

with instrumented_bank.forward_with_bank_instrumented():
    output = forward_with_bank(...)

instrumented_bank.record_hit()
instrumented_bank.update_bank_size(512)
```

## OpenTelemetry Tracing

### Setup

```python
from deltamemory.observability import setup_otel, BankTracer

provider = setup_otel(service_name="mneme")
tracer = BankTracer(layer="layer_0")

with tracer.forward_with_bank_span(context="inference") as span:
    output = forward_with_bank(...)
    tracer.record_hit_event(span, fact_id="fact_id")
    tracer.record_eviction_event(span, count=1, reason="capacity")
```

### Spans

- `mneme.forward_with_bank`: Wraps forward_with_bank calls
- `mneme.write_fact`: Wraps write_fact calls

### Events

- `bank.hit`: Memory lookup successful
- `bank.miss`: Memory lookup failed
- `bank.eviction`: Facts evicted (count, reason)

## Prometheus Integration

### Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'mneme'
    static_configs:
      - targets: ['localhost:9464']
    scrape_interval: 15s
```

### Example Queries

```promql
# Hit rate over 5 minutes
rate(mneme_bank_hit_rate_hits_total[5m]) / rate(mneme_bank_hit_rate_accesses_total[5m])

# Latency p99
histogram_quantile(0.99, rate(mneme_forward_latency_seconds_bucket[1m]))

# Eviction rate
rate(mneme_eviction_total[1m])
```

## Grafana Dashboard

Pre-configured dashboard available at `deployment/grafana/mneme_dashboard.json`.

See [Grafana Setup](../deployment/grafana/README.md) for import instructions.

## Combining Metrics and Traces

```python
from deltamemory.observability import (
    BankMetrics, BankTracer, serve_http, setup_otel
)

metrics = BankMetrics()
serve_http(port=9464)

provider = setup_otel(service_name="mneme")
tracer = BankTracer(layer="layer_0")

with metrics.forward_latency_timer(layer="layer_0"):
    with tracer.forward_with_bank_span() as span:
        output = forward_with_bank(...)
        metrics.record_access(layer="layer_0")
        metrics.record_hit(layer="layer_0")
        tracer.record_hit_event(span)
```

## Advanced

### Custom Meters

```python
from opentelemetry import metrics
meter = metrics.get_meter("my_app")
custom_histogram = meter.create_histogram("custom_latency")
```

### Sampling (Tracing)

For high-throughput systems, use sampling to reduce overhead:

```python
from opentelemetry.sdk.trace.sampler import TraceIdRatioBased

sampler = TraceIdRatioBased(0.1)  # 10% of traces
```

## Performance

- **Metrics overhead**: Negligible (nanosecond-level counters)
- **Histogram recording**: ~microsecond per call
- **Tracing overhead**: ~microsecond per span
- **HTTP server**: Non-blocking (separate thread)

## See Also

- [Prometheus Documentation](https://prometheus.io/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/instrumentation/python/)
- [Grafana Dashboard Guide](../deployment/grafana/README.md)
