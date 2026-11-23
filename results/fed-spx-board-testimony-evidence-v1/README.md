# Fed/SPX Board release and testimony text audit

Adaeze Ilodianya owns the Fed/SPX endpoint design. This audit uses PR #46 head
`c8c89b557f3a1a934b993d9241dea1d2de76eae2` and its frozen 20- and 30-minute
population. It leaves the price windows, five-minute processing buffer,
and split assignments unchanged.

| Release type and split | Rows reviewed | Verified by cutoff | No timely capture in checked index | Archive lookup incomplete |
| --- | ---: | ---: | ---: | ---: |
| Board releases, train | 419 | 0 | 104 | 315 |
| Board releases, validation | 55 | 0 | 0 | 55 |
| Testimony, train | 108 | 1 | 11 | 96 |
| Testimony, validation | 21 | 7 | 0 | 14 |
| Total | 603 | 8 | 115 | 480 |

All 603 current Fed HTML bodies matched the catalog hash. The historical
statuses preserve earlier CDX and Availability attempts, including connection
failures and rows skipped after rate limiting. The fresh two-event archive
pilot checked the modern and legacy Board URL for January 15, 2008. The
legacy URL surfaced a capture three days after the release. The Fed's
[auction archive](https://www.federalreserve.gov/monetarypolicy/taf_2008.htm)
and [auction terms](https://www.federalreserve.gov/monetarypolicy/tafterms.htm)
identify auction records and notification procedures. These sources did not
establish the exact January 15 press-release body by its cutoff. Wayback
returned HTTP 429 when replaying the previously identified June 22, 2021
Powell capture. Broader archival or timestamped wire coverage remains open.

The U.S. House committee repository supplied stronger evidence for testimony.
Eight of nine inspected House listings identified a witness PDF added or last
updated by the frozen cutoff. The audit retrieved each PDF, checked its event
date and cover identity, and fingerprinted both the file and the extracted
model body. For the April 17, 2018 Quarles row, the House listing shows the
document added at 10:19 a.m. ET, after the 10:00 cutoff. That row remains
unresolved. The June 22, 2021 Powell testimony illustrates the difference
between document and event clocks. The public House listing records the
document the previous afternoon, and the document cover gives a scheduled
release the previous day. The dataset keeps its frozen June 22 forecast
origin. Any study of the first market reaction to the written remarks would
need a separately aligned event definition.

- [June 22, 2021 Powell House listing](https://docs.house.gov/Committee/Calendar/ByEvent.aspx?EventID=112798)
- [April 17, 2018 Quarles House listing](https://docs.house.gov/Committee/Calendar/ByEvent.aspx?EventID=108192)
- [House repository documentation](https://docs.house.gov/committee/Help.aspx)

The output contains one entry per assigned event. Eligibility is a positive
finding about the identified House PDF body. The House repository does not
provide a historical checksum of the upload. The audit assumes its
document-level timestamps apply to the linked copy and uses the latest
displayed document timestamp when an update is present. The evidence does
not settle rows with unavailable searches or an unchecked alternate source.
No training or held-out scoring was performed.

## Reproduce

From the endpoint checkout, use Python with `requests`, `beautifulsoup4`,
`lxml`, `pypdf` and IANA timezone data. The first command uses the frozen
local population and the two local PR #46 archive ledgers in
`downloads/wayback-text-v1/` and `downloads/wayback-availability-v1/`.
`summary.json` records their hashes. Archive responses can vary on a rerun.

```powershell
python scripts/audit_fed_board_testimony_versions.py --out results/fed-spx-board-testimony-evidence-v1
python scripts/audit_fed_board_testimony_versions.py --archive-pilot --out results/fed-spx-board-testimony-evidence-v1
python scripts/audit_fed_board_testimony_versions.py --official-document-pass --out results/fed-spx-board-testimony-evidence-v1
```

The CSV contains hashes, timestamps, and evidence links. Extracted text is
written under `downloads/fed-spx-board-testimony-evidence-v1/verified-text/`
and stays local. The shared text cohort awaits the minutes and speech ledger.
