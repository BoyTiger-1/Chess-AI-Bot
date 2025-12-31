from __future__ import annotations

import argparse
import sys

from ai_business_assistant.config import Settings
from ai_business_assistant.orchestration.prefect_flows import build_flows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Business Assistant ETL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-warehouse", help="Create DuckDB warehouse schema")

    run_flow = sub.add_parser("run-flow", help="Run a Prefect flow (local)")
    run_flow.add_argument(
        "name",
        choices=[
            "historical_market_data",
            "realtime_market_polling",
            "customer_load",
            "competitive_intel",
        ],
    )

    migrate = sub.add_parser("migrate", help="Database migration commands")
    migrate_sub = migrate.add_subparsers(dest="subcmd", required=True)
    migrate_sub.add_parser("upgrade", help="Upgrade to latest revision")
    migrate_sub.add_parser("downgrade", help="Downgrade one revision")
    migrate_sub.add_parser("current", help="Show current revision")
    migrate_sub.add_parser("status", help="Show migration status")

    args = parser.parse_args(argv)
    settings = Settings()
    settings.ensure_dirs()

    flows = build_flows(settings)

    if args.cmd == "init-warehouse":
        flows["init_warehouse"]()
        return 0

    if args.cmd == "run-flow":
        flows[args.name]()
        return 0

    if args.cmd == "migrate":
        import alembic.config
        alembic_args = []
        if args.subcmd == "upgrade":
            alembic_args.extend(['upgrade', 'head'])
        elif args.subcmd == "downgrade":
            alembic_args.extend(['downgrade', '-1'])
        elif args.subcmd == "current":
            alembic_args.extend(['current'])
        elif args.subcmd == "status":
            alembic_args.extend(['history'])
        
        alembic.config.main(argv=alembic_args)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
