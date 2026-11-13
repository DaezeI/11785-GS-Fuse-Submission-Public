from scripts.build_c1_date_screen import build_screened_manifest


def test_screen_removes_candidates_and_updates_counts():
    base = {"split_name": "C1", "schedule": [
        {"role": "train", "retained": 1},
        {"role": "vali", "retained": 2},
        {"role": "test", "retained": 2}],
        "train_count": 1, "vali_count": 2, "test_count": 2,
        "train_events": [r"data/event\1\20200101.txt"],
        "vali_events": [r"data/event\9\20210101.txt", r"data/event\9\20210102.txt"],
        "test_events": [r"data/event\8\20220101.txt", r"data/event\8\20220102.txt"]}
    screened = build_screened_manifest(base, {"9/20210102.txt"},
                                       {"8/20220101.txt"}, "abc")
    assert (screened["train_count"], screened["vali_count"], screened["test_count"]) == (1, 1, 1)
    assert screened["schedule"][1]["retained"] == 1
    assert screened["schedule"][2]["retained"] == 1
    assert len(screened["article_date_screen"]["excluded_event_ids"]) == 2
    assert base["vali_count"] == 2  # original stays unchanged
