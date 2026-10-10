"""Run Transformers generation on a tiny random model, entirely offline."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from codeguard.backend import HFBackend
from codeguard.config import load_config

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = (importlib.util.find_spec("transformers") is not None
             and importlib.util.find_spec("torch") is not None
             and (ROOT / ".deps/MarkLLM/watermark/kgw/kgw.py").exists())


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def get_vocab(self):
        return {str(i): i for i in range(128)}

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "8 9 10"

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [int(i) for i in text.split()]
        if return_tensors:
            import torch
            from transformers import BatchEncoding
            return BatchEncoding({"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)})
        return {"input_ids": ids}

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if i != 0)


@unittest.skipUnless(AVAILABLE, "Install ML dependencies and bootstrap MarkLLM")
class BackendTests(unittest.TestCase):
    def test_actual_generate_respects_continuation_and_seeds(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        config = load_config(ROOT / "configs/pilot.yaml")
        config["model"].update(device="cpu", dtype="float32")
        config["generation"]["max_new_tokens"] = 8
        toy_config = GPT2Config(vocab_size=128, n_positions=64, n_embd=32, n_layer=1, n_head=2,
                                eos_token_id=0, pad_token_id=0, bos_token_id=0)
        tokenizer = ToyTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("transformers.AutoConfig.from_pretrained", return_value=toy_config), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=GPT2LMHeadModel(toy_config)):
            backend = HFBackend(config)
        first = backend.generate("def solve():", "dev/test", 42, "watermarked")
        second = backend.generate("def solve():", "dev/test", 42, "watermarked")
        self.assertEqual(first["generated_token_ids"], second["generated_token_ids"])
        self.assertEqual(first["prompt_tokens"], 3)
        self.assertLessEqual(first["generated_tokens"], 8)
        self.assertEqual(first["raw_text"], tokenizer.decode(first["generated_token_ids"]))
        self.assertEqual(len(tokenizer.messages), 2)
        self.assertIn("def solve():", tokenizer.messages[1]["content"])
        clean = backend.generate("def solve():", "dev/test", 42, "clean")
        self.assertEqual(clean["sample_seed"], first["sample_seed"])
        self.assertIn(backend.score(first["raw_text"])["status"], ("ok", "too_short"))


if __name__ == "__main__":
    unittest.main()
