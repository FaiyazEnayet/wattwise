"""Fetch public weather reanalysis for the household town, never an address."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latitude', type=float, default=40.9465)
    parser.add_argument('--longitude', type=float, default=-73.0693)
    parser.add_argument('--start', default='2025-09-19')
    parser.add_argument('--end', default='2026-10-18')
    parser.add_argument('--out', default='demo/household_weather.csv')
    args = parser.parse_args()
    query = dict(latitude=args.latitude, longitude=args.longitude,
                 start_date=args.start, end_date=args.end,
                 hourly='temperature_2m,relative_humidity_2m',
                 temperature_unit='fahrenheit', timezone='America/New_York',
                 models='era5')
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urlencode(query)
    with urlopen(url, timeout=60) as response:
        data = json.load(response)
    frame = pd.DataFrame(data['hourly']).rename(columns={'time': 'timestamp',
                           'temperature_2m': 'temperature_f',
                           'relative_humidity_2m': 'relative_humidity_pct'})
    if frame.temperature_f.notna().sum() < 100:
        raise ValueError('Weather provider returned insufficient nonmissing observations')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    out.with_suffix('.metadata.json').write_text(json.dumps({
        'source': 'Open-Meteo ERA5 reanalysis; modeled weather, not a home sensor',
        'location': 'Port Jefferson NY 11777 approximate town center',
        'request': query, 'url': url, 'rows': len(frame),
    }, indent=2))
    print(f'Wrote {len(frame)} weather rows to {out}')


if __name__ == '__main__':
    main()
