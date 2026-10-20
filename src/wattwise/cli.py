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
    analysis = sub.add_parser('analyze', help='Build the audited full-history decision analysis JSON')
    analysis.add_argument('--root', default='.', help='Repository root containing demo input files')
    analysis.add_argument('--out', default='demo/analysis.json')
    args = parser.parse_args(argv)
    if args.command == "insights":
        print(report())
        return 0
    if args.command == 'analyze':
        import json
        from pathlib import Path
        from .decision_analysis import build_report
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(build_report(args.root), separators=(',', ':'), allow_nan=False), encoding='utf-8')
        print(f'Wrote {output}')
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
