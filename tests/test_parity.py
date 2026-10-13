"""Reject incomplete/nonfinite comparisons; align probability vectors by option ID."""
import copy
import json
import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from parity import compare
from report import load


class ParityIntegrity(unittest.TestCase):
    def setUp(self):
        self.ref = [{"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2}}}]

    def test_option_order_does_not_change_comparison(self):
        got = [{"q": {"type": "choice", "choice": "a", "probabilities": {"b": 0.2, "a": 0.8}}}]
        result = compare(self.ref, got)
        self.assertEqual(result["max_abs_prob_diff"], 0)
        self.assertEqual(result["top_answer_agreement"], 1)

    def test_truncated_dump_is_not_perfect_agreement(self):
        with self.assertRaises(ValueError):
            compare(self.ref * 2, self.ref)

    def test_option_loss_or_nonfinite_value_is_not_a_valid_score(self):
        for probabilities in ({"a": 1.0}, {"a": float("nan"), "b": 0.2}):
            got = copy.deepcopy(self.ref)
            got[0]["q"]["probabilities"] = probabilities
            with self.assertRaises(ValueError):
                compare(self.ref, got)

    def test_aggregate_only_evidence_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate.json").write_text(json.dumps({
                "method": {"rounds": 3}, "workloads": {"one": {"all": {"n": 150, "p50": 20.0}}}}))
            (root / "parity-candidate.json").write_text(json.dumps(self.ref))
            with self.assertRaisesRegex(ValueError, "raw measurement"):
                load(root, "candidate")


if __name__ == "__main__":
    unittest.main()
