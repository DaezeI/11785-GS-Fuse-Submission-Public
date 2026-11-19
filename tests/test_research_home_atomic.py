import json

from scripts.research_home import REGISTRY, upsert, validate


def test_registry_upsert_closes_temporary_file_before_replacing(tmp_path):
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    before = len(registry["experiments"])
    candidate = dict(registry["experiments"][0], id="atomic-upsert-regression-v1",
                     status="planned", figure=None, related=[])
    path = tmp_path / "records.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    upsert(registry, candidate, path)
    actual = json.loads(path.read_text(encoding="utf-8"))
    validate(actual)
    assert len(actual["experiments"]) == before + 1
    assert actual["experiments"][0]["id"] == candidate["id"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["records.json"]
