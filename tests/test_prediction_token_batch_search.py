import pytest
from scripts.train_prediction_token_transformer import largest_batch


def test_batch_search_stops_at_dataset_size_when_all_windows_fit():
    trials = []
    def probe(batch):
        trials.append(batch)
        return True
    assert largest_batch(2500, probe) == 2500
    assert max(trials) == 2500


def test_batch_search_refines_a_memory_boundary():
    trials = []
    def probe(batch):
        trials.append(batch)
        return batch <= 1800
    selected = largest_batch(5000, probe, quantum=16)
    assert 1784 <= selected <= 1800
    assert any(batch > 1800 for batch in trials)


def test_no_usable_gpu_batch_is_an_explicit_failure():
    with pytest.raises(RuntimeError, match="No training batch"):
        largest_batch(100, lambda _: False)
