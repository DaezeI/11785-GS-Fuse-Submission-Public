"""Torch-free checks of the data-free distribution against the historical runner."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/"scripts/improvement/run_compact_treasury_crossattention.py"
CONFIG = ROOT/"experiments/improvement-v1/compact_treasury_crossattention.config.json"
ORIGINAL_NAME = "GS-Fuse-Compact-Treasury-CrossAttention.py"
EXPECTED_EXECUTED_SHA = "86ffa70cbdda87c2a02297cb2637bddf16f32aa67944d0e7127f2e482f25b0dc"
EXPECTED_PINS = {
    "train": ("5b958e6c15d5c051f2148ae366bb41fe46f4fbf3e17aa0c2cc7d4b8867d05e3c", 180, 36),
    "validation": ("9474f81626bf4b489fa9ab6f307e0fd2ba84d9c2b6d5522d43d98fdc2264feef", 60, 12),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def protected_nodes(tree):
    """Everything except the docstring, input loader/CLI, and embedded data packaging."""
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name in {"base64","zlib"}:
            continue
        if isinstance(node, ast.FunctionDef) and node.name in {"embedded_rows","load_pinned_rows","cli"}:
            continue
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_EMBEDDED" for t in node.targets):
            continue
        keep.append(node)
    return keep


def canonical(obj):
    if obj is Ellipsis:
        return {"ellipsis":True}
    if isinstance(obj, ast.AST):
        return [type(obj).__name__, [[k,canonical(v)] for k,v in ast.iter_fields(obj)
                                   if k != "type_params" or v]]
    if isinstance(obj, list):
        return [canonical(x) for x in obj]
    return obj


def ast_digest(tree):
    payload = json.dumps(canonical(protected_nodes(tree)),ensure_ascii=True,separators=(",",":"))
    return digest(payload.encode())


class CrossAttentionSourceDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_syntax_pins_and_protected_ast_checksum(self):
        compile(self.source,str(RUNNER),"exec")
        self.assertEqual(self.config["executed_runner_sha256"], EXPECTED_EXECUTED_SHA)
        self.assertEqual(self.config["protected_training_evaluation_ast_sha256"],ast_digest(self.tree))
        self.assertEqual(self.config["split_pins"],{k:list(v) for k,v in EXPECTED_PINS.items()})
        pins = next(x for x in self.tree.body if isinstance(x,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=="PINS" for t in x.targets))
        self.assertEqual(ast.literal_eval(pins.value),EXPECTED_PINS)
        self.assertNotIn("_EMBEDDED",self.source)
        self.assertNotIn("test.jsonl",self.source)

    def test_exact_executed_source_parity_when_available(self):
        if os.environ.get("GS_FUSE_EXECUTED_CROSSATTN"):
            path = Path(os.environ["GS_FUSE_EXECUTED_CROSSATTN"])
        else:
            path = next((p/ORIGINAL_NAME for p in ROOT.parents if (p/ORIGINAL_NAME).is_file()), None)
            if path is None:
                self.skipTest("Set GS_FUSE_EXECUTED_CROSSATTN to the historical, data-bearing runner")
        original = path.read_bytes()
        self.assertEqual(digest(original),EXPECTED_EXECUTED_SHA)
        self.assertEqual(canonical(protected_nodes(ast.parse(original))),
                         canonical(protected_nodes(self.tree)))

    def extracted_loader(self, pins):
        nodes = [n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=="load_pinned_rows"]
        scope = {"PINS":pins,"sha":digest}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(RUNNER),"exec"),scope)
        return scope["load_pinned_rows"]

    def test_load_opens_only_two_pinned_splits_and_rejects_changes(self):
        data = {"train":b"synthetic train\n","validation":b"synthetic validation\n"}
        loader = self.extracted_loader({k:(digest(v),None,None) for k,v in data.items()})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for k,v in data.items():
                (root/f"{k}.jsonl").write_bytes(v)
            (root/"test.jsonl").write_bytes(b"NEVER OPEN")
            seen = []
            read = Path.read_bytes
            def record(path):
                seen.append(path.name)
                if path.name == "test.jsonl":
                    raise AssertionError("Held-out file was opened")
                return read(path)
            with patch.object(Path,"read_bytes",record):
                self.assertEqual(loader(root),data)
                self.assertEqual(seen,["train.jsonl","validation.jsonl"])
                (root/"train.jsonl").write_bytes(b"changed train bytes\n")
                with self.assertRaisesRegex(ValueError,"not the exact frozen"):
                    loader(root)

    def test_cli_requires_external_dataset_except_smoke(self):
        nodes = [n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=="cli"]
        ns = {"argparse":argparse,"Path":Path}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(RUNNER),"exec"),ns)
        for args in (["--verify"],[],["--smoke","--data-dir","."]):
            with self.assertRaises(SystemExit) as failure:
                ns["cli"](args)
            self.assertEqual(failure.exception.code,2)


if __name__=="__main__":
    unittest.main()
