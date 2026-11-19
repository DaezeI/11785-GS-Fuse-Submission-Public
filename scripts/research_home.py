#!/usr/bin/env python3
"""Validate the GitHub research registry or add a reviewed public record."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research_home/records.json"
REPOSITORY = "DaezeI/11785-GS-Fuse-Submission-Public"
STATUSES = {
    "completed", "queued", "running", "submitted", "planned",
    "awaiting_evidence", "failed", "canceled",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def has_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_url(url):
    require(has_text(url), "Missing link URL")
    parts = urlsplit(url)
    require(parts.scheme == "https" and parts.hostname and not parts.username
            and not parts.password, f"Expected a shareable HTTPS link: {url}")
    for prefix in (
        f"https://github.com/{REPOSITORY}/blob/main/",
        f"https://raw.githubusercontent.com/{REPOSITORY}/main/",
    ):
        if url.startswith(prefix):
            relative = url.removeprefix(prefix).split("#", 1)[0].split("?", 1)[0]
            path = (ROOT / relative).resolve()
            require(path.is_relative_to(ROOT) and path.is_file(),
                    f"Missing repository file for {url}")


def validate_links(links):
    require(isinstance(links, list) and links, "Expected one or more links")
    for link in links:
        require(has_text(link.get("label")), "Missing link label")
        validate_url(link.get("url"))


def parse_timestamp(value):
    require(has_text(value), "Missing update timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Use an ISO timestamp with a timezone")
    return parsed


def validate(registry):
    for key in ("title", "stage", "overview"):
        require(has_text(registry.get(key)), f"Missing workspace {key}")
    parse_timestamp(registry.get("updated_at"))
    validate_url(registry.get("source_url"))
    validate_links(registry.get("links"))
    for item in registry["next"]:
        require(has_text(item.get("title")) and has_text(item.get("reason")),
                "Each next step needs a title and reason")
    experiments = registry["experiments"]
    require(isinstance(experiments, list) and experiments, "No experiment records")
    ids = set()
    for record in experiments:
        run_id = record.get("id", "")
        require(isinstance(run_id, str) and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", run_id)
                and run_id not in ids, f"Invalid or duplicate id: {run_id}")
        ids.add(run_id)
        for key in ("title", "kind", "owner", "question", "configuration", "finding", "limitations"):
            require(has_text(record.get(key)), f"{run_id} needs {key}")
        require(record.get("status") in STATUSES, f"{run_id} has an unknown status")
        parse_timestamp(record.get("updated_at"))
        validate_links(record.get("evidence"))
        figure = record["figure"]
        if figure is not None:
            require(has_text(figure.get("alt")), f"{run_id} needs figure alt text")
            validate_url(figure.get("url"))
        else:
            require(record["status"] != "completed", f"{run_id} needs a figure")
        require(isinstance(record["related"], list), f"{run_id} needs a related list")
    for record in experiments:
        require(all(r in ids and r != record["id"] for r in record["related"]),
                f"Invalid related record for {record['id']}")
    for item in registry["next"]:
        dependency = item.get("until_completed")
        require(dependency is None or dependency in ids,
                "A next step must refer to an existing experiment")


def upsert(registry, candidate, path):
    records = registry["experiments"]
    previous = next((item for item in records if item["id"] == candidate["id"]), None)
    require(previous is None or parse_timestamp(candidate["updated_at"])
            >= parse_timestamp(previous["updated_at"]), "Refusing to replace a newer record")
    require(previous is None or previous["status"] != "completed"
            or candidate["status"] == "completed", "A completed run needs a new ID for a fresh attempt")
    if previous == candidate:
        validate(registry)
        return
    if previous is None:
        records.insert(0, candidate)
    else:
        records[records.index(previous)] = candidate
    if candidate["status"] == "completed":
        registry["next"] = [
            item for item in registry["next"]
            if item.get("until_completed") != candidate["id"]
        ]
    registry["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    validate(registry)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
            temporary = Path(tmp.name)
            json.dump(registry, tmp, indent=2, ensure_ascii=False, allow_nan=False)
            tmp.write("\n")
            tmp.flush()
        # Windows cannot replace a NamedTemporaryFile while its handle is open.
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY,
                        help="Alternate registry for a local dry run")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    add = commands.add_parser("upsert")
    add.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        if args.command == "upsert":
            candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
            upsert(registry, candidate, args.registry)
        else:
            validate(registry)
    except (ValueError, OSError, KeyError, AttributeError, TypeError) as error:
        parser.error(str(error))
    print(f"Validated {len(registry['experiments'])} research records")


if __name__ == "__main__":
    main()
