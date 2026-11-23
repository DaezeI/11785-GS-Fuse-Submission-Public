"""Splits and the numpy dataset on synthetic data (no real prices or text)."""
import numpy as np
import pandas as pd
import pytest

from data.fed_price_dataset import FedPriceDataset
from scripts import build_fed_price_alignment as fpa
from test_fed_price_alignment import events_csv, make_prices, write_prices


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fed")
    prices = make_prices(start="2008-02-25", end="2009-06-30")
    price_csv = write_prices(tmp, prices)
    days = pd.bdate_range("2008-03-10", "2009-05-20")[::3]          # one event every 3 trading days
    items = [(f"e{i}", f"{d:%Y-%m-%d} 14:00:00", True) for i, d in enumerate(days)]
    items += [(f"dup{i}", f"{d:%Y-%m-%d} 14:00:00", True) for i, d in enumerate(days[:5])]   # concurrent
    items += [(f"pre{i}", f"{d:%Y-%m-%d} 08:30:00", True) for i, d in enumerate(days[10:20])]  # flagged subset
    ev = events_csv(tmp, items)
    out_dir = tmp / "out"
    fpa.main(["--events", str(ev), "--prices", str(price_csv), "--out-dir", str(out_dir),
              "--expect-price-sha256", ""])
    return {"csv": out_dir / "fed_price_alignment_v1.csv", "prices": price_csv, "tmp": tmp,
            "audit": out_dir / "price_alignment_audit_v1.json", "n_events": len(items)}


def ds(built, split, **kw):
    return FedPriceDataset(built["csv"], built["prices"], split, **kw)


def test_splits_are_chronological_with_expected_shares(built):
    rows = pd.read_csv(built["csv"], parse_dates=["available_et"])
    main = rows[(rows.exclude_reason.isna()) & rows.main_candidate]
    share = main.split.value_counts(normalize=True)
    assert 0.6 < share["train"] < 0.8 and 0.08 < share["val"] < 0.25 and 0.08 < share["test"] < 0.25
    last = {s: main[main.split == s].available_et.max() for s in ("train", "val", "test")}
    first = {s: main[main.split == s].available_et.min() for s in ("train", "val", "test")}
    assert last["train"] < first["val"] and last["val"] < first["test"]


def test_no_price_bar_is_shared_between_splits(built):
    seen = {}
    for split in ("train", "val", "test"):
        for subset in ("main", "flagged"):
            d = ds(built, split, subset=subset)
            bars = set()
            for i in range(len(d)):
                xi, yi = d.window_bar_times(i)
                bars |= set(xi.tolist()) | set(yi.tolist())
            seen.setdefault(split, set()).update(bars)
    assert not (seen["train"] & seen["val"]) and not (seen["val"] & seen["test"]) and not (seen["train"] & seen["test"])


def test_concurrent_events_stay_in_one_split(built):
    rows = pd.read_csv(built["csv"])
    kept = rows[rows.split.isin(["train", "val", "test"])]
    assert (kept.groupby("available_utc").split.nunique() == 1).all()
    assert (kept.n_concurrent_usable > 1).any()


def test_audit_reports_disjoint_windows(built):
    import json
    a = json.loads(built["audit"].read_text())["splits"]
    assert a["windows_disjoint"] and a["instants_not_split"]
    assert sum(a["rows_main"].values()) > 0


def test_verify_splits_detects_overlap():
    t = pd.Timestamp
    def frame(train_end):
        return pd.DataFrame({
            "split": ["train", "val", "test"], "available_utc": ["a", "b", "c"],
            "input_first_bar": [t("2008-01-01"), t("2008-01-10"), t("2008-01-20")],
            "target_last_bar": [t(train_end), t("2008-01-15"), t("2008-01-25")]})
    assert fpa.verify_splits(frame("2008-01-05"))["windows_disjoint"]
    with pytest.raises(AssertionError, match="share price bars"):
        fpa.verify_splits(frame("2008-01-12"))


def test_example_shapes_and_types(built):
    d = ds(built, "train")
    assert len(d) > 0
    e = d[0]
    assert e["x"].shape == (35, 4) and e["y"].shape == (140, 4)
    assert e["x"].dtype == np.float32 and e["is_main"] and e["text"] is None
    assert d[-1]["event_id"] == d.rows.event_id.iloc[-1]
    with pytest.raises(IndexError):
        d[len(d)]


def test_input_and_target_do_not_overlap_in_time_and_skip_the_reaction_bar(built):
    d = ds(built, "train")
    for i in range(len(d)):
        xi, yi = d.window_bar_times(i)
        gap = (yi[0] - xi[-1]) / np.timedelta64(1, "m")
        assert gap >= 10          # the 14:00-14:05 bar sits between 13:55 and 14:05 (start-to-start 10 min)


def test_return_mode_is_relative_to_last_input_close(built):
    d = ds(built, "train", x_mode="return_bp", y_mode="return_bp")
    raw = ds(built, "train")
    e, r = d[0], raw[0]
    assert abs(e["x"][-1, 3]) < 1e-3                                  # last input Close -> 0 bp
    expected = 1e4 * np.log(r["y"][:, 3] / r["last_close"])
    np.testing.assert_allclose(e["y"][:, 3], expected, rtol=1e-4, atol=1e-3)


def test_scaler_uses_train_only_and_is_applied_everywhere(built):
    train = ds(built, "train", x_mode="return_bp", y_mode="return_bp")
    sc = FedPriceDataset.fit_scaler(train)
    xs = np.concatenate([train._transform(train._raw(i)[0], train._raw(i)[2], "return_bp") for i in range(len(train))])
    np.testing.assert_allclose(sc["x_mean"], xs.mean(0))
    scaled = ds(built, "train", x_mode="return_bp", y_mode="return_bp", scaler=sc)
    x, _ = scaled.as_arrays()
    assert abs(x.reshape(-1, 4).mean()) < 0.05
    with pytest.raises(ValueError, match="train split only"):
        FedPriceDataset.fit_scaler(ds(built, "val"))
    with pytest.raises(ValueError, match="unscaled"):
        FedPriceDataset.fit_scaler(scaled)


def test_flagged_subset_and_text_join(built):
    flagged = ds(built, "train", subset="flagged")
    assert len(flagged) == 0 or not any(flagged[i]["is_main"] for i in range(len(flagged)))
    allrows = ds(built, "train", subset="all")
    assert len(allrows) >= len(ds(built, "train"))
    tdir = built["tmp"] / "text"
    tdir.mkdir(exist_ok=True)
    first = ds(built, "train").rows.event_id.iloc[0]
    (tdir / f"{first}.txt").write_text("Statement text.")
    withtext = ds(built, "train", text_dir=tdir)
    assert withtext[0]["text"] == "Statement text."
    assert withtext[1]["text"] is None
    with pytest.raises(FileNotFoundError):
        ds(built, "train", text_dir=tdir, require_text=True)[1]


def test_rejects_a_different_price_file(built, tmp_path):
    other = write_prices(tmp_path, make_prices(start="2008-02-25", end="2009-06-30", drop_days=("2008-03-12",)))
    with pytest.raises(ValueError, match="same SP500.csv"):
        FedPriceDataset(built["csv"], other, "train")
