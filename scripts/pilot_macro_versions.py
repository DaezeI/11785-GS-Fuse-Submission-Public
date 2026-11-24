"""Pilot: does a release-day archive copy match the archived release file byte for byte?

For each release: (1) digest of the file we would use today (direct download where the
site allows it; for bls.gov, which refuses non-browser clients, the Wayback copies of
the archive URL); (2) Wayback CDX captures of the file that was live on release day,
from the release time to 24 hours later; (3) the embargo line and PDF dates read from
one copy. Same digest => the release-day file and today's file are byte-identical.
"""
import base64, datetime as dt, hashlib, io, json, re, sys, time
from zoneinfo import ZoneInfo
import requests

ET, UTC = ZoneInfo("America/New_York"), dt.timezone.utc
S = requests.Session(); S.headers["User-Agent"] = "CMU-11785-GS-Fuse-research (macro release version audit)"

def get(url, delay=2.5):
    for k in range(3):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(60); continue
            time.sleep(delay); return r
        except Exception as e:
            err = e; time.sleep(5)
    return None

def b32(b): return base64.b32encode(hashlib.sha1(b).digest()).decode()

def cdx(url, frm=None, to=None):
    q = f"https://web.archive.org/cdx/search/cdx?url={re.sub(r'^https?://','',url)}&output=json&fl=timestamp,statuscode,digest,original&limit=500"
    if frm: q += f"&from={frm:%Y%m%d%H%M%S}&to={to:%Y%m%d%H%M%S}"
    r = get(q)
    if r is None or not r.ok: return None
    rows = r.json()[1:] if r.text.strip() else []
    return [x for x in rows if x[1] == "200"]

def pdf_facts(b):
    import pypdf
    try:
        rd = pypdf.PdfReader(io.BytesIO(b)); m = rd.metadata or {}
        first = re.sub(r"\s+", " ", rd.pages[0].extract_text() or "")
    except Exception as e:
        return {"pdf_error": repr(e)[:100]}
    emb = re.search(r"(embargoed until[^.]{0,90})", first, re.I)
    return {"embargo_line": emb.group(1)[:110] if emb else None,
            "pdf_created": str(m.get("/CreationDate")), "pdf_modified": str(m.get("/ModDate"))}

R = [  # id, source, release ET, today file (direct or 'wayback:'), release-day live URLs
 ("bls_empsit_20080606", "BLS", "2008-06-06 08:30", "wayback:www.bls.gov/news.release/archives/empsit_06062008.pdf", ["www.bls.gov/news.release/pdf/empsit.pdf"]),
 ("bls_cpi_20110317", "BLS", "2011-03-17 08:30", "wayback:www.bls.gov/news.release/archives/cpi_03172011.pdf", ["www.bls.gov/news.release/pdf/cpi.pdf"]),
 ("bls_jolts_20130709", "BLS", "2013-07-09 10:00", "wayback:www.bls.gov/news.release/archives/jolts_07092013.pdf", ["www.bls.gov/news.release/pdf/jolts.pdf"]),
 ("bls_ppi_20160513", "BLS", "2016-05-13 08:30", "wayback:www.bls.gov/news.release/archives/ppi_05132016.pdf", ["www.bls.gov/news.release/pdf/ppi.pdf"]),
 ("bls_empsit_20190405", "BLS", "2019-04-05 08:30", "wayback:www.bls.gov/news.release/archives/empsit_04052019.pdf", ["www.bls.gov/news.release/pdf/empsit.pdf"]),
 ("bea_gdp_20080731", "BEA", "2008-07-31 08:30", "page:https://www.bea.gov/news/2008/gross-domestic-product-second-quarter-2008-advance", []),
 ("bea_pi_20100628", "BEA", "2010-06-28 08:30", "page:https://www.bea.gov/news/2010/personal-income-and-outlays-may-2010", []),
 ("bea_gdp_20140730", "BEA", "2014-07-30 08:30", "page:https://www.bea.gov/news/2014/gross-domestic-product-2nd-quarter-2014-advance-estimate-includes-historical-revisions", []),
 ("bea_pi_20170331", "BEA", "2017-03-31 08:30", "page:https://www.bea.gov/news/2017/personal-income-and-outlays-february-2017", []),
 ("bea_gdp_20210429", "BEA", "2021-04-29 08:30", "page:https://www.bea.gov/news/2021/gross-domestic-product-first-quarter-2021-advance-estimate", []),
 ("dol_claims_20090611", "DOL", "2009-06-11 08:30", "https://oui.doleta.gov/press/2009/061109.pdf", ["www.dol.gov/opa/media/press/eta/ui/current.htm", "www.dol.gov/ui/data.pdf"]),
 ("dol_claims_20200409", "DOL", "2020-04-09 08:30", "https://oui.doleta.gov/press/2020/040920.pdf", ["www.dol.gov/ui/data.pdf", "www.dol.gov/opa/media/press/eta/ui/current.htm"]),
]

out = []
sel = set(sys.argv[2:])
for rid, src, rel, today, live in R:
    if sel and rid not in sel: continue
    t0 = dt.datetime.strptime(rel, "%Y-%m-%d %H:%M").replace(tzinfo=ET).astimezone(UTC)
    row = {"id": rid, "source": src, "release_et": rel}
    today_digests, sample = set(), None
    if today.startswith("wayback:"):
        caps = cdx(today[8:]) or []
        today_digests = {x[2] for x in caps}
        row["today_file"] = today[8:]; row["today_basis"] = f"Wayback copies of archive URL ({len(caps)} captures, {len(today_digests)} distinct digests)"
        if caps:
            row["archive_first_capture"] = caps[0][0]; row["archive_last_capture"] = caps[-1][0]
            r = get(f"https://web.archive.org/web/{caps[-1][0]}id_/{caps[-1][3]}")
            if r is not None and r.ok: sample = r.content
    else:
        if today.startswith("page:"):
            p = get(today[5:], 1.0)
            if p is not None and p.ok:
                m = re.search(r"EMBARGOED UNTIL[^<]{0,80}", p.text); row["page_embargo_line"] = m.group(0).strip() if m else None
                pdfs = [h for h in re.findall(r'href="([^"]+\.pdf)"', p.text) if re.search(r"(gdp|pi)\w*\d\d\w*\.pdf$", h, re.I) and "fax" not in h and "tech" not in h]
                row["page_pdf_links"] = pdfs[:4]
                today = ("https://www.bea.gov" + pdfs[0]) if pdfs and pdfs[0].startswith("/") else (pdfs[0] if pdfs else None)
                if today:
                    fn = today.rsplit("/", 1)[1]; yr = rel[:4]; kind = "gdp" if "gdp" in rid else "pi"
                    live = [re.sub(r"^https?://", "", today), f"www.bea.gov/newsreleases/national/{kind}/{yr}/pdf/{fn}",
                            f"www.bea.gov/national/{kind}/{yr}/pdf/{fn}"]
                    live += [re.sub(r"^https?://(www\.)?", "www.", h) for h in pdfs[1:] if h.startswith("http")]
            else:
                row["today_basis"] = "release page unavailable"; today = None
        if today:
            r = get(today, 1.0)
            row["today_file"] = today
            if r is not None and r.ok and r.content[:4] == b"%PDF":
                today_digests = {b32(r.content)}; sample = r.content
                row["today_basis"] = "direct download"; row["today_server_written"] = r.headers.get("Last-Modified")
            else:
                row["today_basis"] = f"download failed {getattr(r, 'status_code', None)}"
    if sample: row.update(pdf_facts(sample))
    hits, seen = [], []
    for u in dict.fromkeys(live):
        caps = cdx(u, t0, t0 + dt.timedelta(hours=24))
        if caps is None: seen.append((u, "index unavailable")); continue
        seen.append((u, len(caps)))
        hits += [(x[0], x[3], x[2] in today_digests) for x in caps]
    row["release_day_lookups"] = seen
    row["release_day_captures"] = len(hits)
    match = [h for h in hits if h[2]]
    row["release_day_match"] = match[0][:2] if match else None
    if not today_digests:
        row["disposition"] = "blocked_or_missing_today_file"
    elif match:
        row["disposition"] = "eligible_same_bytes_on_release_day"
    elif hits:
        row["disposition"] = "version_conflict_review"
    else:
        row["disposition"] = "insufficient_no_release_day_copy"
    out.append(row)
    print(rid, row["disposition"], row.get("release_day_match"), row.get("embargo_line") or row.get("page_embargo_line"), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
