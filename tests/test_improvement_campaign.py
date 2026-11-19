import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.improvement import campaign, submit


def profile():
    return dict(checkout="/private/code", python="/env/bin/python", snapshot="/data/snapshot",
                moment="/models/moment", text_model="/models/llama", run_root="/private/runs",
                pbs_directives=["#PBS -l ngpus=1", "#PBS -l walltime=01:00:00"],
                setup="source /env/activate", walltime_hours=1)


def arguments(tmp_path, **changes):
    private = tmp_path / "profile.json"
    private.write_text(json.dumps(profile()))
    values = dict(profile=private, out=tmp_path / "batch", variants=None, seeds=[0],
                  slots=2, split="vali", selection=None, phase="smoke", readiness=None)
    return argparse.Namespace(**{**values, **changes})


def test_screen_renders_linux_paths_and_no_submission(tmp_path):
    args = arguments(tmp_path)
    with patch.object(campaign.subprocess, "check_output", side_effect=["a" * 40, ""]):
        campaign.render(args)
    plan = json.loads((args.out / "plan.json").read_text())
    assert len(plan["tasks"]) == 5
    assert plan["max_gpu_hours"] == 5
    assert plan["phase"] == "smoke"
    assert plan["resource_dir"] == "/private/runs/batch"
    text = (args.out / "ref-s0.pbs").read_text()
    assert b"\r" not in (args.out / "ref-s0.pbs").read_bytes()
    assert "\\" not in text
    assert text.index("#PBS") < text.index("set -e")
    assert "/private/code/scripts/improvement/campaign.py" in text
    assert not (args.out / "submission.jsonl").exists()


def test_test_scoring_requires_locked_selection(tmp_path):
    with pytest.raises(ValueError, match="locked validation selection"):
        campaign.render(arguments(tmp_path, split="test"))


def test_declared_budget_must_match_pbs_walltime(tmp_path):
    args = arguments(tmp_path)
    data = profile()
    data["walltime_hours"] = .5
    args.profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="walltime matching"):
        campaign.render(args)


def package(tmp_path, count=3):
    tasks = []
    for index in range(count):
        script, config = tmp_path / f"task{index}.pbs", tmp_path / f"task{index}.json"
        script.write_text("#!/bin/bash\ntrue\n")
        config.write_text("{}")
        tasks.append(dict(id=f"task{index}", script=script.name, sha256=submit.digest(script),
                          task_file=config.name, task_sha256=submit.digest(config)))
    (tmp_path / "plan.json").write_text(json.dumps(dict(resource_dir=str(tmp_path.resolve()),
                                                        slots=2, tasks=tasks)))


def test_submit_two_lanes_journals_and_refuses_duplicate(tmp_path):
    package(tmp_path)
    replies = [SimpleNamespace(returncode=0, stdout=f"{i}.server\n", stderr="") for i in (1, 2, 3)]
    with patch.object(submit.subprocess, "run", side_effect=replies) as qsub:
        submit.submit(tmp_path, 2)
    calls = [call.args[0] for call in qsub.call_args_list]
    assert "-W" not in calls[0] and "-W" not in calls[1]
    assert calls[2][1:3] == ["-W", "depend=afterok:1.server"]
    assert len((tmp_path / "submission.jsonl").read_text().splitlines()) == 4
    with pytest.raises(FileExistsError):
        submit.submit(tmp_path, 2)


def test_submit_records_unexpected_scheduler_receipt(tmp_path):
    package(tmp_path, 1)
    with patch.object(submit.subprocess, "run", return_value=SimpleNamespace(
            returncode=0, stdout="ambiguous receipt", stderr="")):
        with pytest.raises(RuntimeError, match="inspect the journal"):
            submit.submit(tmp_path, 1)
    assert "ambiguous receipt" in (tmp_path / "submission.jsonl").read_text()


def test_submit_rejects_modified_task_before_any_submission(tmp_path):
    package(tmp_path, 1)
    (tmp_path / "task0.json").write_text('{"changed":true}')
    with patch.object(submit.subprocess, "run") as qsub:
        with pytest.raises(ValueError, match="package changed"):
            submit.submit(tmp_path, 1)
        qsub.assert_not_called()


def test_matrix_variants_keep_one_change_or_named_package():
    matrix = campaign.read(campaign.MATRIX)
    assert len(matrix["variants"]) == 17
    assert len(matrix["gated_designs"]) == 11
    assert campaign.resolved(matrix, "context280")["seq-len"] == 280
    assert campaign.resolved(matrix, "lora8")["use-lora"]
    assert "use-lora" not in campaign.resolved(matrix, "ref")
    for variant in matrix["variants"]:
        config = campaign.resolved(matrix, variant)
        assert config["batch-size"] == 2 and config["pred-len"] == 140
        assert config["no-4bit"] and config["tf32"]


def test_full_training_cannot_render_without_readiness(tmp_path):
    args = arguments(tmp_path, phase="full")
    with patch.object(campaign.subprocess, "check_output", side_effect=["a" * 40, ""]):
        with pytest.raises(ValueError, match="requires --readiness"):
            campaign.render(args)
    assert not args.out.exists()


def test_readiness_is_bound_to_code_snapshot_and_explicit_verified_checks():
    matrix = campaign.read(campaign.MATRIX)
    checks = dict.fromkeys(("failure_diagnosed", "disk_and_quota_budget_checked", "window_audits_passed",
                           "stage_handoff_smoke_passed", "gradient_smokes_passed"), True)
    record = dict(source_commit="a" * 40, snapshot_sha256=matrix["snapshot_sha256"],
                  checks=checks, timing_scope="date-screened-sensitivity", evidence=["private receipt + checksum"])
    campaign.validate_readiness(record, "a" * 40, matrix)
    with pytest.raises(ValueError, match="source commit"):
        campaign.validate_readiness(record, "b" * 40, matrix)
    record["checks"]["disk_and_quota_budget_checked"] = False
    with pytest.raises(ValueError, match="readiness checks"):
        campaign.validate_readiness(record, "a" * 40, matrix)
