"""
main.py — P74: Entry point for the full-chain electricity price prediction system.

Usage::

    python main.py \\
        --raw-data ../electricity_forecast_model2.1/data/shandong_pmos_hourly.csv \\
        --dayahead-source-repo .local_artifacts/source_repos/epf-sota-experiment \\
        --realtime-source-repo .local_artifacts/source_repos/electricity_forecast_deep_sgdf_delta \\
        --target-start 2026-06-01 \\
        --target-end 2026-06-30 \\
        --profile trusted_delivery \\
        --fusion-engine period_bgew \\
        --work-dir .local_artifacts/full_chain_run \\
        --strict --strict-no-leakage \\
        --train-realtime-if-missing \\
        --reuse-artifacts \\
        --json
"""

from __future__ import annotations

import sys


def main() -> int:
    """Entry point: delegates to scripts.run_full_chain.main()."""
    from scripts.run_full_chain import main as run_full_chain_main
    return run_full_chain_main()


if __name__ == "__main__":
    sys.exit(main())
