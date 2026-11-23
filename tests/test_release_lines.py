import datetime as dt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from release_lines import parse_release_line  # noqa: E402

UTC = dt.timezone.utc


def utc(*args):
    return dt.datetime(*args, tzinfo=UTC)


def test_single_eastern_zone():
    assert parse_release_line("For release on delivery 10:00 a.m. EDT May 6, 2010")["utc"] == utc(2010, 5, 6, 14)


def test_eastern_first_with_local_zone():
    line = "For release on delivery 9:30 a.m. EDT (8:30 a.m. CDT) May 6, 2010"
    assert parse_release_line(line)["utc"] == utc(2010, 5, 6, 13, 30)


def test_local_zone_first_uses_eastern():
    line = "For release on delivery 12:30 p.m. CDT (1:30 p.m. EDT) April 7, 2010"
    assert parse_release_line(line)["utc"] == utc(2010, 4, 7, 17, 30)


def test_noon_whole_hour_capitals_and_weekday():
    assert parse_release_line("For release on delivery Noon EST February 22, 2019")["utc"] == utc(2019, 2, 22, 17)
    assert parse_release_line("For release on delivery 10 a.m. EDT October 20, 2014")["utc"] == utc(2014, 10, 20, 14)
    line = "For release on delivery 2:00 P.M. ET Tuesday, May 12, 2020"
    assert parse_release_line(line)["utc"] == utc(2020, 5, 12, 18)


def test_central_only_is_converted():
    line = "For release on delivery 8:55 a.m. CDT March 31, 2014"
    assert parse_release_line(line)["utc"] == utc(2014, 3, 31, 13, 55)


def test_missing_time_or_phrase():
    assert parse_release_line("For release on delivery May 1, 2014")["utc"] is None
    assert parse_release_line("Remarks by the Chair") is None
