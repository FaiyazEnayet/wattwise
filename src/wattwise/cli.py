"""Command-line interface for wattwise.

Usage:
    python -m wattwise.cli insights
    wattwise insights   (after pip install)
"""

import argparse

from .insights import report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="wattwise",
        description="Home energy analytics on measured smart-meter data.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "insights",
        help="Print the six-section insights report (forecast, peak "
             "drift, what-if, seasonal outlook, cost drivers, anomalies) "
             "for the latest measured period.")
    args = parser.parse_args(argv)
    if args.command == "insights":
        print(report())
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
