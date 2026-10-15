"""Watermark comparison and attribution using the shared generation pipeline."""

from copy import deepcopy
import gc
from pathlib import Path

from .config import load_config
from .pipeline import generate

MODELS = {
    "qwen-1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a", 918273645),
    "qwen-0.5b": ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "ea3f2471cf1b1f0db85067f1ef93848e38e88c25", 271828183),
    "deepseek-1.3b": ("deepseek-ai/deepseek-coder-1.3b-instruct", "e063262dac8366fc1f28a4da0ff3c50ea66259ca", 314159269),
}


def base_config(path):
    return load_config(path)


def comparison_configs(path, output):
    base = base_config(path)
    result = {}
    for delta in (1, 2, 3):
        config = deepcopy(base)
        prefix = 'sweet_comparison' if base['watermark'].get('algorithm') == 'SWEET' else 'watermark_comparison'
        config["name"] = f"{prefix}-delta-{delta}"
        config["watermark"]["delta"] = float(delta)
        config["output_dir"] = str(Path(output).resolve() / f"delta-{delta}")
        result[f"delta-{delta}"] = config
    return result


def attribution_configs(path, output, delta=2.0):
    import math
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    base = base_config(path)
    result = {}
    for label, (name, revision, key) in MODELS.items():
        config = deepcopy(base)
        config["name"] = f"attribution-{label}"
        config["model"].update(name=name, revision=revision)
        config["watermark"].update(hash_key=key, delta=delta)
        config["output_dir"] = str(Path(output).resolve() / label)
        result[label] = config
    return result


def generate_runs(configs):
    for label, config in configs.items():
        print(f"\nExperiment: {label}", flush=True)
        generate(config)
        # Only one model occupies GPU memory at a time.
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
