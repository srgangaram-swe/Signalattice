"""Decision-readiness and implementation-feasibility diagnostics.

The public functions in this package turn predictive and portfolio outputs into
report-friendly evidence about transaction costs, execution delay, capacity,
inference latency, and deployment readiness.
"""

from __future__ import annotations

from quant_platform.evaluation.decision import (
    break_even_cost_bps,
    cost_sensitivity,
    execution_delay_sensitivity,
    liquidity_capacity_table,
    readiness_gate,
    warm_inference_benchmark,
)

__all__ = [
    "break_even_cost_bps",
    "cost_sensitivity",
    "execution_delay_sensitivity",
    "liquidity_capacity_table",
    "readiness_gate",
    "warm_inference_benchmark",
]
