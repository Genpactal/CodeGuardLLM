"""Strict configuration parsing; paths are relative to the YAML file."""

from copy import deepcopy
from pathlib import Path
import math
import re

import yaml


def _keys(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("Configuration sections must be mappings")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"Configuration keys: missing={sorted(missing)}, unknown={sorted(extra)}")


def _number(value, name, low, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside the supported range")


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _keys(config, ["name", "output_dir", "markllm_path", "dataset", "model", "generation", "watermark", "evaluation"])
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("name must be a nonempty string")
    for field in ("output_dir", "markllm_path"):
        config[field] = str((path.parent / config[field]).resolve())
    dataset = config["dataset"]
    if dataset.get("source") == "local":
        _keys(dataset, ["source", "path", "limit"])
        dataset["path"] = str((path.parent / dataset["path"]).resolve())
    elif dataset.get("source") == "huggingface":
        _keys(dataset, ["source", "name", "revision", "split", "limit"])
        if not re.fullmatch(r"[a-f0-9]{40}", str(dataset["revision"])):
            raise ValueError("Pin the dataset to a full commit SHA")
    else:
        raise ValueError("dataset.source must be local or huggingface")
    _number(dataset["limit"], "dataset.limit", 1, integer=True)
    model = config["model"]
    _keys(model, ["name", "revision", "device", "dtype"])
    if not re.fullmatch(r"[a-f0-9]{40}", str(model["revision"])):
        raise ValueError("Pin the model and tokenizer to a full commit SHA")
    if model["device"] not in ("cpu", "cuda") or model["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported device or dtype")
    if model["device"] == "cpu" and model["dtype"] != "float32":
        raise ValueError("Use float32 for CPU runs")
    generation = config["generation"]
    _keys(generation, ["seeds", "temperature", "top_p", "max_new_tokens"])
    if not isinstance(generation["seeds"], list) or not generation["seeds"]:
        raise ValueError("generation.seeds must be a nonempty list")
    for seed in generation["seeds"]:
        _number(seed, "seed", 0, 2**32 - 1, integer=True)
    if len(set(generation["seeds"])) != len(generation["seeds"]):
        raise ValueError("Duplicate seeds are not allowed")
    _number(generation["temperature"], "temperature", 0.0001)
    _number(generation["top_p"], "top_p", 0.0001, 1)
    _number(generation["max_new_tokens"], "max_new_tokens", 1, integer=True)
    wm = config["watermark"]
    _keys(wm, ["gamma", "delta", "hash_key", "prefix_length", "f_scheme", "window_scheme", "z_threshold", "ignore_repeated_ngrams"])
    _number(wm["gamma"], "gamma", 0.0001, 0.9999)
    _number(wm["delta"], "delta", 0)
    _number(wm["hash_key"], "hash_key", 1, 2**63 - 1, integer=True)
    _number(wm["z_threshold"], "z_threshold", 0)
    if (wm["prefix_length"], wm["f_scheme"], wm["window_scheme"]) != (1, "time", "left"):
        raise ValueError("This project supports the single-token time/left KGW scheme only")
    if type(wm["ignore_repeated_ngrams"]) is not bool:
        raise ValueError("ignore_repeated_ngrams must be a boolean")
    evaluation = config["evaluation"]
    _keys(evaluation, ["image", "timeout_seconds", "memory_mb"])
    _number(evaluation["timeout_seconds"], "timeout_seconds", 1, 60, integer=True)
    _number(evaluation["memory_mb"], "memory_mb", 64, 4096, integer=True)
    return config


def portable_config(config):
    """Exclude machine-specific locations from experiment identity."""
    result = deepcopy(config)
    result.pop("output_dir")
    result.pop("markllm_path")
    result["dataset"].pop("path", None)
    return result
