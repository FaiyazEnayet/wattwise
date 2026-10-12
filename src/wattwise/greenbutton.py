"""Green Button XML import.

Parses the ESPI (Energy Services Provider Interface) XML that PSEG's
customer portal exports as "Green Button data": a feed of IntervalBlocks,
each holding timestamped IntervalReadings. Returns an hourly kWh series.

Value semantics follow the spec: energy_kwh = value * 10**powerOfTenMultiplier
/ 1000, where the raw value is in watt-hours. Durations other than one hour
are resampled to hourly by even allocation.
"""

import datetime as dt
import xml.etree.ElementTree as ET

import pandas as pd

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "espi": "http://naesb.org/espi",
}


def _text(el, path):
    node = el.find(path, _NS)
    return node.text.strip() if node is not None and node.text else None


def read_greenbutton_xml(path_or_xml: str) -> pd.Series:
    """Read a Green Button XML file into an hourly kWh pandas Series.

    Accepts a file path or a raw XML string. The series index is timezone-naive
    local timestamps; values are kWh per hour.
    """
    if "<feed" in path_or_xml[:5000] or path_or_xml.lstrip().startswith("<"):
        xml_text = path_or_xml
    else:
        with open(path_or_xml, "r", encoding="utf-8") as f:
            xml_text = f.read()

    root = ET.fromstring(xml_text)
    records = []
    for block in root.findall(".//espi:IntervalBlock", _NS):
        mult_txt = _text(block, ".//espi:powerOfTenMultiplier")
        try:
            multiplier = int(mult_txt) if mult_txt is not None else 0
        except ValueError:
            multiplier = 0
        for reading in block.findall("espi:IntervalReading", _NS):
            start = _text(reading, "espi:timePeriod/espi:start")
            duration = _text(reading, "espi:timePeriod/espi:duration")
            value = _text(reading, "espi:value")
            if start is None or value is None:
                continue
            try:
                ts = dt.datetime.fromtimestamp(int(start), tz=dt.timezone.utc)
                ts = ts.astimezone().replace(tzinfo=None)  # local naive
                wh = float(value) * (10 ** multiplier)
                dur_h = float(duration) / 3600.0 if duration else 1.0
            except (ValueError, OverflowError):
                continue
            if dur_h <= 0:
                continue
            # Evenly allocate sub/super-hourly intervals across whole hours.
            kwh_total = wh / 1000.0
            n_hours = max(int(round(dur_h)), 1)
            per_hour = kwh_total / n_hours
            base = ts.replace(minute=0, second=0, microsecond=0)
            for i in range(n_hours):
                records.append((base + dt.timedelta(hours=i), per_hour))

    if not records:
        raise ValueError("No interval readings found in Green Button XML.")
    df = pd.DataFrame(records, columns=["timestamp", "kwh"])
    series = df.groupby("timestamp")["kwh"].sum().sort_index()
    series.index = pd.to_datetime(series.index)
    # Collapse to clean hourly grid.
    full = pd.date_range(series.index.min().floor("h"),
                         series.index.max().ceil("h"), freq="h")
    return series.reindex(full, fill_value=0.0)
