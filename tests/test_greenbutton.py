"""Green Button XML import tests on a hand-built sample feed."""

import datetime as dt

import pandas as pd

from wattwise.greenbutton import read_greenbutton_xml

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:espi="http://naesb.org/espi">
  <entry>
    <content>
      <espi:IntervalBlock>
        <espi:IntervalReading>
          <espi:timePeriod><espi:start>1767225600</espi:start><espi:duration>3600</espi:duration></espi:timePeriod>
          <espi:value>1250</espi:value>
        </espi:IntervalReading>
        <espi:IntervalReading>
          <espi:timePeriod><espi:start>1767229200</espi:start><espi:duration>3600</espi:duration></espi:timePeriod>
          <espi:value>800</espi:value>
        </espi:IntervalReading>
        <espi:IntervalReading>
          <espi:timePeriod><espi:start>1767232800</espi:start><espi:duration>7200</espi:duration></espi:timePeriod>
          <espi:value>2000</espi:value>
        </espi:IntervalReading>
        <espi:readingType><espi:powerOfTenMultiplier>0</espi:powerOfTenMultiplier></espi:readingType>
      </espi:IntervalBlock>
    </content>
  </entry>
</feed>
"""
# 1767225600 = 2026-01-01 00:00:00 UTC


def test_parses_hourly_values():
    s = read_greenbutton_xml(SAMPLE_XML)
    # Values are Wh with multiplier 0 -> kWh.
    assert abs(s.iloc[0] - 1.25) < 1e-9
    assert abs(s.iloc[1] - 0.80) < 1e-9


def test_super_hourly_split_evenly():
    s = read_greenbutton_xml(SAMPLE_XML)
    # 2000 Wh over 2 hours -> 1.0 kWh each.
    assert abs(s.iloc[2] - 1.0) < 1e-9
    assert abs(s.iloc[3] - 1.0) < 1e-9


def test_total_energy_conserved():
    s = read_greenbutton_xml(SAMPLE_XML)
    assert abs(s.sum() - (1.25 + 0.80 + 2.0)) < 1e-9


def test_hourly_index():
    s = read_greenbutton_xml(SAMPLE_XML)
    assert isinstance(s.index, pd.DatetimeIndex)
    diffs = s.index.to_series().diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta(hours=1)


def test_empty_feed_raises():
    try:
        read_greenbutton_xml("<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty feed")


def test_multiplier_applied():
    xml = SAMPLE_XML.replace(
        "<espi:powerOfTenMultiplier>0</espi:powerOfTenMultiplier>",
        "<espi:powerOfTenMultiplier>-3</espi:powerOfTenMultiplier>")
    s = read_greenbutton_xml(xml)
    # 1250 * 1e-3 Wh -> 0.00125 kWh
    assert abs(s.iloc[0] - 0.00125) < 1e-9
