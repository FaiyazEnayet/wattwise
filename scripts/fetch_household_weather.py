"""Fetch public weather reanalysis for the household area.

The exact request location is a private parameter: it is never committed,
never defaulted in this script, and never written to the public metadata.
Provide it explicitly with --latitude/--longitude, with the
WATTWISE_WEATHER_LAT and WATTWISE_WEATHER_LON environment variables, or
with a local (git-ignored) JSON file passed as --config, shaped like
{"latitude": <lat>, "longitude": <lon>}.

The public metadata records only the provider, date range, variables and
row count, plus a note that exact request parameters are withheld. This
script never silently replaces historical weather: it only writes the CSV
the owner explicitly asks for.
"""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

PRIVATE_CONFIG_NAME = 'demo/weather.private.json'


def resolve_location(args):
    """Return (latitude, longitude) from CLI, env, or a local config file."""
    lat = args.latitude
    lon = args.longitude
    if lat is None:
        env_lat = os.environ.get('WATTWISE_WEATHER_LAT')
        lat = float(env_lat) if env_lat else None
    if lon is None:
        env_lon = os.environ.get('WATTWISE_WEATHER_LON')
        lon = float(env_lon) if env_lon else None
    config_path = args.config
    if config_path is None and (lat is None or lon is None):
        candidate = Path(PRIVATE_CONFIG_NAME)
        if candidate.exists():
            config_path = str(candidate)
    if config_path is not None and (lat is None or lon is None):
        config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        lat = lat if lat is not None else config.get('latitude')
        lon = lon if lon is not None else config.get('longitude')
    return lat, lon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latitude', type=float, default=None,
                        help='Request latitude (private; no committed default)')
    parser.add_argument('--longitude', type=float, default=None,
                        help='Request longitude (private; no committed default)')
    parser.add_argument('--config', default=None,
                        help='Local JSON file with latitude/longitude; keep private, never commit')
    parser.add_argument('--start', default='2025-09-19')
    parser.add_argument('--end', default='2026-10-18')
    parser.add_argument('--out', default='demo/household_weather.csv')
    args = parser.parse_args()
    latitude, longitude = resolve_location(args)
    if latitude is None or longitude is None:
        parser.error('Provide --latitude/--longitude, set WATTWISE_WEATHER_LAT and '
                     'WATTWISE_WEATHER_LON, or pass --config with a local JSON file. '
                     'Coordinates are private and have no committed defaults.')
    query = dict(latitude=latitude, longitude=longitude,
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
        'location_policy': ('Exact request parameters are private: withheld from '
                            'public artifacts and never committed. The data owner '
                            'reproduces the request from their private config.'),
        'date_range': {'start': args.start, 'end': args.end},
        'hourly_variables': ['temperature_2m', 'relative_humidity_2m'],
        'temperature_unit': 'fahrenheit',
        'timezone': 'America/New_York',
        'models': 'era5',
        'rows': len(frame),
    }, indent=2))
    print(f'Wrote {len(frame)} weather rows to {out}')


if __name__ == '__main__':
    main()
