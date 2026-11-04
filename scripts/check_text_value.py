#!/usr/bin/env python3
"""Run the focused text-value tests without importing unrelated backbone tests."""
import argparse
import sys
import unittest
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--require-torch", action="store_true", help="Treat skipped PyTorch helper checks as failure")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestSuite()
    for name in ("test_shared_split_audit.py", "test_text_prediction_analysis.py",
                 "test_text_control_runner.py", "test_text_control_summary.py", "test_no_text_control.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(root / "tests"), pattern=name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not (args.require_torch and result.skipped) else 1


if __name__ == "__main__": sys.exit(main())
