"""Actual pinned MarkLLM integration with synthetic tokens, no model download."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import unittest

from codeguard.config import load_config
from codeguard.watermark import KGWAdapter

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = importlib.util.find_spec("torch") is not None and (ROOT / ".deps/MarkLLM/watermark/kgw/kgw.py").exists()


class IntegerTokenizer:
    def get_vocab(self):
        return {str(i): i for i in range(128)}

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(token) for token in text.split()]}


@unittest.skipUnless(AVAILABLE, "Install .[ml] and bootstrap MarkLLM for integration tests")
class WatermarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.config = load_config(ROOT / "configs/week1.yaml")
        cls.settings = cls.config["watermark"]

    def adapter(self, **overrides):
        settings = {**self.settings, **overrides}
        return KGWAdapter(self.config["markllm_path"], settings, IntegerTokenizer(), 128, "cpu")

    def test_processor_bias_matches_detector_and_correct_key(self):
        wm = self.adapter()
        ids = [19]
        # Choose from the actual biased logits; avoid repeated bigrams.
        used = set()
        for _ in range(100):
            logits = self.torch.zeros((1, 128))
            biased = wm.processor(self.torch.tensor([ids]), logits)
            candidates = [i for i in range(128) if biased[0, i] > 0 and (ids[-1], i) not in used]
            self.assertTrue(candidates)
            chosen = candidates[len(ids) % len(candidates)]
            used.add((ids[-1], chosen))
            ids.append(chosen)
        text = " ".join(map(str, ids))
        score = wm.score(text)
        self.assertEqual(score["green_count"], 100)
        self.assertTrue(score["is_watermarked"])
        self.assertEqual(score, self.adapter().score(text))
        wrong = self.adapter(hash_key=7777777).score(text)
        self.assertLess(wrong["z_score"], score["z_score"])
        self.assertFalse(wrong["is_watermarked"])

    def test_duplicate_bigram_filter_and_upstream_equivalence(self):
        wm = self.adapter(ignore_repeated_ngrams=False)
        text = "3 7 3 7 3 7"
        score = wm.score(text)
        upstream, _ = wm.kgw.utils.score_sequence(self.torch.tensor([3, 7, 3, 7, 3, 7]))
        self.assertAlmostEqual(score["z_score"], upstream)
        self.assertEqual(score["scored_tokens"], 5)
        self.assertEqual(self.adapter().score(text)["scored_tokens"], 2)

    def test_zero_bias_and_short_text(self):
        wm = self.adapter(delta=0)
        logits = self.torch.randn((1, 128))
        expected = logits.clone()
        self.assertTrue(self.torch.equal(wm.processor(self.torch.tensor([[1, 2]]), logits), expected))
        for text in ("", "3"):
            self.assertEqual(wm.score(text)["status"], "too_short")
            self.assertIsNone(wm.score(text)["z_score"])


if __name__ == "__main__":
    unittest.main()
