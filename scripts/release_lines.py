"""Parse "For release on delivery" lines from Fed speech and testimony PDFs.

Replaces the single-format pattern in ``verify_minutes_speech_versions.py``, which missed
lines such as "9:30 a.m. EDT (8:30 a.m. CDT) May 6, 2010", "12:30 p.m. CDT (1:30 p.m. EDT)",
capital "A.M.", "ET" and a weekday before the date. Returns the Eastern-time release instant.
"""
import datetime as dt
import re
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
PHRASE = re.compile(r"for\s+release\s+on\s+delivery", re.I)
CLOCK = re.compile(
    r"(?:\b(\d{1,2})(?:[:.](\d{2}))?\s*([ap])\.?\s*m\.?|(?:\b12[:.]00\s*)?\b(noon))\s*\(?\s*"
    r"(E\.?[SD]\.?T\.?|ET\b|Eastern(?:\s+(?:Standard|Daylight))?(?:\s+Time)?|"
    r"[CMP]\.?[SD]\.?T\.?|[CMP]T\b|GMT|BST|CET|CEST|JST|UTC)?", re.I)
DATE = re.compile(r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*)?"
                  r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+(\d{1,2}),?\s*(\d{4})", re.I)
US_OFFSETS = {"C": -1, "M": -2, "P": -3}  # hours relative to Eastern, same standard/daylight state


def parse_release_line(first_page_text):
    """Return {"text", "utc", "zone"} or None. ``utc`` is None when the line has no usable clock or date."""
    text = re.sub(r"\s+", " ", first_page_text or "")
    m = PHRASE.search(text)
    if not m:
        return None
    window = text[m.end():m.end() + 160]
    date = DATE.search(window)
    clocks = list(CLOCK.finditer(window[:date.start()] if date else window[:90]))
    result = {"text": text[m.start():m.end() + (date.end() if date else 90)][:160], "utc": None, "zone": None}
    if not clocks or not date:
        return result
    day = dt.datetime.strptime(f"{date.group(1).title()} {int(date.group(2))} {date.group(3)}", "%B %d %Y").date()

    def zone_of(c):
        z = (c.group(5) or "").upper().replace(".", "")
        if z.startswith("E") or z == "":
            return "ET" if z else ""
        if z[0] in US_OFFSETS and z.endswith("T"):
            return z[0]
        return "OTHER"

    eastern = [c for c in clocks if zone_of(c) == "ET"]
    chosen, shift = None, 0
    if eastern:
        chosen = eastern[0]
    elif len(clocks) == 1 and zone_of(clocks[0]) == "":
        chosen = clocks[0]  # bare time with no zone: Fed convention is Eastern
        result["zone"] = "assumed_eastern"
    else:
        us = [c for c in clocks if zone_of(c) in US_OFFSETS]
        if us:
            chosen, shift = us[0], -US_OFFSETS[zone_of(us[0])]
            result["zone"] = f"converted_from_{zone_of(us[0])}T"
    if chosen is None:
        result["zone"] = "non_us_zone_only"
        return result
    if chosen.group(4):
        hour, minute = 12, 0
    else:
        hour = int(chosen.group(1)) % 12 + (12 if chosen.group(3).lower() == "p" else 0)
        minute = int(chosen.group(2) or 0)
    local = dt.datetime.combine(day, dt.time(hour, minute), ET) + dt.timedelta(hours=shift)
    result["utc"] = local.astimezone(UTC)
    result["zone"] = result["zone"] or "eastern"
    return result
