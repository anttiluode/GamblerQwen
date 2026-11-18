import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from gambler import (WATCH, choice_prompt, make_world, odds, report_metrics,
                     run, save, simulate_quiet, summarize_signal)


class GamblerTests(unittest.TestCase):
    def test_stationary_and_reversal(self):
        self.assertEqual(odds("stationary", 3, 40, 20), (0.72, 0.28))
        self.assertEqual(odds("stationary", 30, 40, 20), (0.72, 0.28))
        self.assertEqual(odds("reversal", 19, 40, 20), (0.76, 0.24))
        self.assertEqual(odds("reversal", 20, 40, 20), (0.24, 0.76))

    def test_hidden_events_are_reproducible_and_policy_independent(self):
        a = make_world("reversal", 48, 24, 9)
        b = make_world("reversal", 48, 24, 9)
        self.assertEqual(a, b)
        self.assertEqual(a[23]["p_a"], 0.76)
        self.assertEqual(a[24]["p_a"], 0.24)
        self.assertTrue(all(r["reward_a"] in (0, 1) for r in a))
        self.assertEqual(len(simulate_quiet(a, "alternate", 9)), 48)
        self.assertEqual(len(simulate_quiet(a, "window", 9)), 48)

    def test_prompt_never_reveals_rule(self):
        history = [{"round": 1, "action": "A", "reward": 0}]
        p = choice_prompt(history)
        for forbidden in ("0.76", "rugpull", "regime", "switch", "unfair"):
            self.assertNotIn(forbidden, p.casefold())
        self.assertIn("01 A 0", p)

    def test_signal_only_with_both_concept_families(self):
        self.assertIsNone(summarize_signal({}, 26))
        self.assertIsNone(summarize_signal({"26": {"changed": 4}}, 26))
        s = summarize_signal({"26": {"changed": 2, "repeat": -1}}, 26)
        self.assertEqual(s, 3)

    def test_demo_report_and_html_data(self):
        args = Namespace(mode="demo", scenario="rugpull", rounds=24,
                         change_round=12, seed=7, model="none",
                         device="cpu", lens=None, layers=[12, 20, 26],
                         execute="qwen")
        result = run(args)
        self.assertEqual(len(result["rounds"]), 24)
        self.assertEqual(result["meta"]["source"], "demo")
        self.assertEqual(result["meta"]["change_round"], 13)
        self.assertTrue(all(r["lens"] == {} for r in result["rounds"]))
        self.assertTrue(all(r["proposed"] is None for r in result["rounds"]))
        self.assertEqual(len(result["baselines"]), 4)
        self.assertEqual(result["metrics"]["wins"], result["rounds"][-1]["total"])
        with tempfile.TemporaryDirectory() as temp:
            js = Path(temp) / "data.js"
            j = Path(temp) / "latest.json"
            save(result, js, j)
            self.assertTrue(js.read_text().startswith("window.GAMBLER_DATA="))
            self.assertEqual(json.loads(j.read_text())["meta"]["seed"], 7)

    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            make_world("reversal", 4, 4, 9)


if __name__ == "__main__":
    unittest.main()
