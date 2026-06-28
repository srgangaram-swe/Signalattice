"""Print a compact table of recent tracked experiments."""

from __future__ import annotations

import argparse

from quant_platform.config import AppConfig
from quant_platform.tracking import get_tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    tracker = get_tracker(cfg.tracking)
    runs = tracker.list_runs()[: args.limit]
    if not runs:
        print("No tracked runs found.")
        return

    print(f"{'started_at':<20} {'name':<18} {'status':<10} {'bt_sharpe':>10} {'data_hash':<12}")
    for run in runs:
        metrics = run.get("metrics", {}) or {}
        sharpe = metrics.get("bt_sharpe")
        sharpe_str = "n/a" if sharpe is None else f"{float(sharpe):.3f}"
        print(
            f"{str(run.get('started_at', ''))[:19]:<20} "
            f"{str(run.get('name', '')):<18} "
            f"{str(run.get('status', '')):<10} "
            f"{sharpe_str:>10} "
            f"{str(run.get('data_hash', '')):<12}"
        )


if __name__ == "__main__":
    main()
