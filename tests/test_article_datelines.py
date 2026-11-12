import json

from scripts.audit_article_datelines import audit, dateline_date


def test_reuters_opening_dateline_and_unrelated_date():
    opening = "A December 15 photo. By Stephen CulpNEW YORK, Dec 20 (Reuters) - Stocks fell."
    assert dateline_date(opening) == ("1220", None)


def test_explicit_year_and_non_reuters_text():
    assert dateline_date("LONDON, January 3, 2024 (Reuters) - Example") == ("0103", "2024")
    assert dateline_date("Prices from Dec 19 are in the report") is None


def test_audit_flags_wrong_folder_and_keeps_unknown_separate(tmp_path):
    manifest = {"train_events": [r"data/event\9\20231216.txt"],
                "vali_events": [r"data/event\9\20231219.txt"],
                "test_events": [r"data/event\8\20231216.txt"]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    for category, folder, text in (
        ("9", "20231216", "NEW YORK, Dec 19 (Reuters) - Story"),
        ("9", "20231219", "NEW YORK, Dec 19 (Reuters) - Story"),
        ("8", "20231216", "A report about December 19"),
    ):
        parent = tmp_path / "event" / category / f"{folder}.txt"
        parent.mkdir(parents=True)
        (parent / f"{folder}.txt_full_summary.txt").write_text(text)
    result = audit(path, tmp_path / "event")
    assert result["counts"]["train:candidate_mismatch"] == 1
    assert result["counts"]["vali:folder_date_matches"] == 1
    assert result["counts"]["test:no_opening_reuters_dateline"] == 1
    assert result["candidates"][0]["event_id"] == "9/20231216.txt"
