# Coverage count: U.S. macro releases + Fed events on frozen cash SPX (candidates only)

Counted November 23, 2026. **No text was verified.** These are candidate forecast events that
have a cash-SPX window. Splits use the alignment v1 cut dates: train through 2019-05-20, validation
2019-05-28 to 2021-08-18, test 2021-08-24 to 2024-04-10. Prices end 2024-04-17.

## Distinct forecast events

Simultaneous releases are bundled: every premarket release on a day shares that day's opening
window, and intraday releases with the same five-minute origin share one window.

| Session | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Intraday (Fed events + JOLTS, some personal income at 10:00) | 1,053 | 222 | 229 | 1,504 |
| Premarket (8:30 releases; opening-window rule) | 1,105 | 206 | 242 | 1,553 |
| **All** | **2,158** | **428** | **471** | **3,057** (+5 before the first train date) |
| Premarket excluding days with only jobless claims | 660 | 130 | 156 | |
| Intraday events from Fed releases only | 913 | 192 | 196 | |

There are 3,062 events in all; 322 of them bundle two releases.

## Releases by source (before bundling)

| Source | Train | Val | Test |
|---|---:|---:|---:|
| BLS Employment Situation (8:30) | 133 | 26 | 31 |
| BLS CPI (8:30) | 135 | 26 | 32 |
| BLS PPI (8:30) | 136 | 27 | 31 |
| BLS JOLTS (10:00) | 135 | 27 | 32 |
| BEA GDP (8:30) | 134 | 27 | 32 |
| BEA Personal Income (8:30; 10:00 in a few years) | 132 | 27 | 31 |
| DOL initial claims (Thursday 8:30, approximate schedule) | 591 | 116 | 137 |
| Fed (alignment v1 main rows) | 978 | 207 | 216 |

Lost before counting: 91 releases fall outside price coverage; 11 come on non-session days
(e.g. jobs reports on Good Friday); 45 Fed releases come too late for a 30-minute target; 2 days
have no price bars.

## Caveats

- These are candidate counts. Each release still needs an exact-version proof. The partial Fed
  audits pass about 98% of FOMC statements but only about 17% of minutes and speeches checked.
- BLS index pages start in February 2008. BLS times are the standard release times and were not
  checked per release. Claims dates assume Thursday, moved to Wednesday on holidays, and need the
  DOL archive.
- Premarket and intraday events use different forecast rules, so they should be reported
  separately. Neither group reaches 2,000 training events alone.
- Test-period events are counted from the exchange calendar only; no test prices were read.

## Text-version pilot and evidence rule (decided November 23, 2026)

A pilot of 12 releases (`pilot_text_versions.json`, `scripts/pilot_macro_versions.py`) found
release-day archive copies with identical bytes for only 4: BLS jobs 2019, BEA GDP 2014, BEA
personal income 2017 and DOL claims 2020. The 2009 claims file was not found at the expected
address.

**Decision: macro releases use a document-based rule, recorded here as a limitation.**
A release qualifies when all three of these hold:

1. The release text states its own embargo time, no later than the text cutoff.
2. The PDF's internal modification date precedes that time.
3. Every archive copy of the file has the same content digest, so it never changed. A release-day
   copy with the same bytes also satisfies this.

This is weaker than the standard used for FOMC statements, whose minutes reproduce the text next to
its release time. The rule does not prove the file was public at the cutoff. It relies on the
agency's embargo practice and on the file being unchanged since it was prepared. Report this in
any model claim.

Pilot rows under this rule. Some conditions still need checking:

- Pass: BLS jobs 2008, PPI 2016, jobs 2019 (embargo line, pre-release date, one digest in all
  copies); BEA GDP 2014, personal income 2017; DOL claims 2020 (release-day match).
- Fail: BLS CPI 2011. Its internal date, April 14, 2011, is four weeks after release.
- Pending: BLS JOLTS 2013, whose embargo line was not read from the PDF. Also BEA GDP 2008,
  personal income 2010 and GDP 2021, which have an embargo line and a pre-release date but have not
  yet been checked for archive-copy stability. DOL claims 2009 has no file located.

## Reproduce

```sh
python scripts/count_macro_coverage.py --bls results/macro-coverage-v1/bls_releases.tsv \
  --bea results/macro-coverage-v1/bea_releases.tsv --price-csv ../fed-person3/downloads/SP500.csv \
  --out results/macro-coverage-v1/coverage.json
```

The BLS dates came from the bls.gov archive index pages for each series (bls.gov refuses
non-browser clients). The BEA dates were read from bea.gov/news/archive, filtered by product and year.
