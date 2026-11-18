"""Regression checks for the real recordings published by data.js."""
import json
from pathlib import Path
import unittest

from publish_results import DEFAULT_RUNS, bundle


ROOT = Path(__file__).resolve().parents[1]


class PublishedRunsTests(unittest.TestCase):
    def test_site_bundles_two_genuine_qwen_runs(self):
        content = (ROOT / "data.js").read_text(encoding="utf-8")
        prefix = "window.GAMBLER_RUNS="
        end = ";\nwindow.GAMBLER_DATA=window.GAMBLER_RUNS[0].data;"
        self.assertTrue(content.startswith(prefix))
        self.assertIn(end, content)
        published = json.loads(content[len(prefix):content.index(end)])
        self.assertEqual(len(published), 2)
        self.assertEqual(published[0]["data"]["meta"]["execution_policy"], "alternate")
        self.assertEqual(published[1]["data"]["meta"]["execution_policy"], "qwen")
        self.assertEqual(published[0]["data"]["meta"]["source"], "qwen")
        self.assertEqual(len(published[0]["data"]["rounds"]), 48)
        self.assertEqual(len(published[1]["data"]["rounds"]), 48)
        self.assertEqual(sum(r["proposed"] == "B" for r in published[0]["data"]["rounds"]), 47)
        self.assertEqual(sum(r["proposed"] == "B" for r in published[1]["data"]["rounds"]), 45)
        self.assertEqual([r["p_a"] for r in published[0]["data"]["rounds"]],
                         [r["p_a"] for r in published[1]["data"]["rounds"]])
        self.assertEqual([r["p_b"] for r in published[0]["data"]["rounds"]],
                         [r["p_b"] for r in published[1]["data"]["rounds"]])

    def test_bundle_matches_raw_records(self):
        from os import chdir, getcwd
        cwd = getcwd()
        try:
            chdir(ROOT)
            records = bundle(list(DEFAULT_RUNS))["runs"]
        finally:
            chdir(cwd)
        content = (ROOT / "data.js").read_text(encoding="utf-8")
        published = json.loads(content.split("window.GAMBLER_RUNS=", 1)[1].split(";\n", 1)[0])
        for actual, expected in zip(published, records):
            self.assertEqual(actual["data"], expected["data"])

    def test_site_exposes_run_controls_and_finding(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        for tag in ('id="runPicker"', 'id="runComparison"', 'id="verdict"',
                    'function renderAnalysis()', 'window.GAMBLER_RUNS',
                    'id="oddsChart"', 'id="lensChart"'):
            self.assertIn(tag, html)


if __name__ == "__main__":
    unittest.main()
