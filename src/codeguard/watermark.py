"""Adapter for pinned MarkLLM KGW, with optional repeated-bigram filtering."""

import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from . import MARKLLM_REVISION


def check_markllm(root):
    root = Path(root).resolve()
    if not (root / "watermark/kgw/kgw.py").is_file():
        raise RuntimeError("MarkLLM is missing. Run python scripts/bootstrap_markllm.py")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != MARKLLM_REVISION:
        raise RuntimeError(f"MarkLLM revision mismatch: expected {MARKLLM_REVISION}, got {revision}")
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        if subprocess.run(["git", "-C", str(root), *args], check=False).returncode:
            raise RuntimeError("The pinned MarkLLM checkout has modified tracked files")
    return root


class KGWAdapter:
    def __init__(self, root, settings, tokenizer, vocab_size, device, model=None):
        root = check_markllm(root)
        sys.path.insert(0, str(root))
        module = importlib.import_module("watermark.kgw.kgw")
        transformer_module = importlib.import_module("utils.transformers_config")
        for imported in (module, transformer_module):
            if not Path(imported.__file__).resolve().is_relative_to(root):
                raise RuntimeError("MarkLLM module name collision; start a fresh Python process")
        self.settings = dict(settings)
        self.tokenizer = tokenizer
        self.device = device
        if not 0 < int(vocab_size * settings["gamma"]) < vocab_size:
            raise ValueError("gamma must produce a nonempty green and red list")
        if max(tokenizer.get_vocab().values()) >= vocab_size:
            raise ValueError("Tokenizer contains token IDs outside the model vocabulary")
        transformer_config = transformer_module.TransformersConfig(model, tokenizer, vocab_size=vocab_size, device=device)
        upstream_config = {key: value for key, value in settings.items() if key != "ignore_repeated_ngrams"}
        upstream_config["algorithm_name"] = "KGW"
        with tempfile.TemporaryDirectory(prefix="codeguard-kgw-") as temp:
            path = Path(temp) / "KGW.json"
            path.write_text(json.dumps(upstream_config), encoding="utf-8")
            self.kgw = module.KGW(str(path), transformer_config)
        self.processor = self.kgw.logits_processor

    def score(self, text):
        import torch

        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        result = {"status": "too_short", "token_count": len(ids), "scored_tokens": 0,
                  "green_count": 0, "z_score": None, "is_watermarked": None,
                  "ignore_repeated_ngrams": self.settings["ignore_repeated_ngrams"]}
        if len(ids) < 2:
            return result
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        seen, green, count = set(), 0, 0
        # This adapter intentionally supports h=1 only. Each scored trial is a bigram.
        for index in range(1, len(ids)):
            bigram = (ids[index - 1], ids[index])
            if self.settings["ignore_repeated_ngrams"] and bigram in seen:
                continue
            seen.add(bigram)
            greenlist = self.kgw.utils.get_greenlist_ids(tensor[:index])
            green += int((greenlist == tensor[index]).any().item())
            count += 1
        gamma = self.settings["gamma"]
        z_score = (green - gamma * count) / math.sqrt(count * gamma * (1 - gamma))
        result.update(status="ok", scored_tokens=count, green_count=green, z_score=z_score,
                      is_watermarked=z_score > self.settings["z_threshold"])
        return result
