"""Hugging Face generation. Tests and canonical solutions never enter prompts."""

from importlib.metadata import version
import platform
import time

from . import MARKLLM_REVISION
from .storage import digest
from .watermark import KGWAdapter

SYSTEM_PROMPT = "You are a Python programmer. Return only the complete Python solution, without tests or explanation."
USER_PREFIX = "Implement the function described below. Include required imports and preserve the function name and signature. Return complete Python code.\n\n"


class HFBackend:
    def __init__(self, config, load_model=True):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install model dependencies: pip install -e '.[ml]'") from exc
        self.config = config
        algorithm = config['watermark'].get('algorithm', 'KGW')
        if algorithm not in ('KGW', 'SWEET'):
            raise ValueError(f'Unsupported watermark algorithm: {algorithm}')
        spec = config["model"]
        self.device = spec["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a GPU runtime, or use a new config with cpu/float32")
        common = dict(revision=spec["revision"], trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(spec["name"], **common)
        model_config = AutoConfig.from_pretrained(spec["name"], **common)
        self.model = None
        # SWEET also needs model probabilities when used only as a detector.
        if load_model or algorithm == 'SWEET':
            self.model = AutoModelForCausalLM.from_pretrained(
                spec["name"], torch_dtype=getattr(torch, spec["dtype"]), use_safetensors=True,
                attn_implementation="eager", **common,
            ).to(self.device).eval()
            if self.model.get_output_embeddings().weight.shape[0] != model_config.vocab_size:
                raise RuntimeError("Output vocabulary differs from the model configuration")
        adapter = KGWAdapter
        if algorithm == 'SWEET':
            from .sweet import SWEETAdapter
            adapter = SWEETAdapter
        self.watermark = adapter(config["markllm_path"], config["watermark"], self.tokenizer,
                                    model_config.vocab_size, self.device, self.model)
        self.metadata = {
            "backend": "huggingface-markllm", "model": spec["name"], "revision": spec["revision"],
            "tokenizer_revision": spec["revision"], "vocab_size": model_config.vocab_size,
            "markllm_revision": MARKLLM_REVISION, "device": self.device, "dtype": spec["dtype"],
            "python": platform.python_version(), "platform": platform.system(),
            "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "numpy", "PyYAML", "huggingface-hub")},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None,
            "prompt_template_hash": digest([SYSTEM_PROMPT, USER_PREFIX]),
        }
        if algorithm == 'SWEET':
            self.metadata.update(algorithm='SWEET', detector_version=1,
                                 entropy_context='text-only; no prompt; first token unscored',
                                 entropy_precision='float32', entropy_chunk_size=64)

    def generate(self, prompt, task_id, seed, condition):
        import torch
        from transformers import GenerationConfig, LogitsProcessorList, set_seed

        if self.model is None:
            raise RuntimeError("This backend was loaded for detection only")
        if condition not in ("clean", "watermarked"):
            raise ValueError("Unknown condition")
        sample_seed = int(digest([task_id, seed])[:8], 16)
        set_seed(sample_seed)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PREFIX + prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(self.device)
        settings = self.config["generation"]
        if inputs["input_ids"].shape[1] + settings["max_new_tokens"] > self.model.config.max_position_embeddings:
            raise ValueError("Prompt plus output budget exceeds the model context")
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        generation = GenerationConfig(
            do_sample=True, temperature=settings["temperature"], top_p=settings["top_p"], top_k=0,
            max_new_tokens=settings["max_new_tokens"], eos_token_id=eos,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            use_cache=True,
        )
        processors = LogitsProcessorList([self.watermark.processor] if condition == "watermarked" else [])
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=generation, logits_processor=processors)
        # Decode ONLY the continuation; never score prompt tokens.
        ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        elapsed = time.perf_counter() - started
        stop_ids = eos if isinstance(eos, list) else [eos]
        stopped = bool(ids and ids[-1] in stop_ids)
        return {
            "raw_text": self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "generated_token_ids": ids, "generated_tokens": len(ids),
            "prompt_tokens": inputs["input_ids"].shape[1], "sample_seed": sample_seed,
            "generation_seconds": elapsed, "finish_reason": "eos" if stopped else "length",
        }

    def score(self, text):
        return self.watermark.score(text)
