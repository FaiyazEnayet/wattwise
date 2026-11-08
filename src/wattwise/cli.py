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
    household = sub.add_parser(
        'household',
        help='Summarize one household from its own local interval/bill '
             'CSVs and a declared tariff (see wattwise.household)')
    household.add_argument('--id', required=True,
                           help='Anonymous household label (lowercase slug)')
    household.add_argument('--intervals', required=True,
                           help='Interval CSV in the wattwise.household schema')
    household.add_argument('--format', default='canonical',
                           choices=['canonical', 'pseg'],
                           help='Interval CSV format (default: canonical)')
    household.add_argument('--bills', default=None,
                           help='Optional bill CSV in the wattwise.household schema')
    household.add_argument('--tariff-json', default=None,
                           help='JSON file with a tariff spec for tariff_from_spec')
    household.add_argument('--tariff', default=None, choices=['pseg-194'],
                           help='Built-in tariff to use instead of --tariff-json')
    household.add_argument('--supply-rate', type=float, default=None,
                           help='Supply rate for a built-in tariff')
    household.add_argument('--out', default=None,
                           help='Write the report JSON here (default: stdout)')
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
    if args.command == 'household':
        from pathlib import Path
        from .household import (HouseholdDataset, build_household_report,
                                load_household_bills, load_household_intervals,
                                rate194_tariff, report_bytes, tariff_from_spec)
        if args.tariff_json:
            import json as _json
            tariff = tariff_from_spec(
                _json.loads(Path(args.tariff_json).read_text(encoding='utf-8')))
        elif args.tariff == 'pseg-194':
            kw = {} if args.supply_rate is None else {'supply_rate': args.supply_rate}
            tariff = rate194_tariff(**kw)
        else:
            parser.error('household requires --tariff-json or --tariff')
        dataset = HouseholdDataset(
            household_id=args.id,
            intervals=load_household_intervals(args.intervals,
                                                input_format=args.format),
            bills=(load_household_bills(args.bills) if args.bills else None),
            tariff=tariff)
        text_out = report_bytes(build_household_report([dataset])).decode('utf-8') + '\n'
        if args.out:
            Path(args.out).write_text(text_out, encoding='utf-8')
            print(f'Wrote {args.out}')
        else:
            print(text_out, end='')
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
