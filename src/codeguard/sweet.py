"""Pinned MarkLLM SWEET with bounded-memory entropy and optional bigram filtering."""

import importlib
import json
import math
from pathlib import Path
import sys
import tempfile

from .watermark import check_markllm


class SWEETAdapter:
    def __init__(self, root, settings, tokenizer, vocab_size, device, model=None):
        if model is None:
            raise RuntimeError('SWEET detection requires the generation model to calculate entropy')
        root = check_markllm(root)
        sys.path.insert(0, str(root))
        module = importlib.import_module('watermark.sweet.sweet')
        transformer_module = importlib.import_module('utils.transformers_config')
        for imported in (module, transformer_module):
            if not Path(imported.__file__).resolve().is_relative_to(root):
                raise RuntimeError('MarkLLM module name collision; start a fresh Python process')
        if not 0 < int(vocab_size * settings['gamma']) < vocab_size:
            raise ValueError('gamma must produce a nonempty green and red list')
        if max(tokenizer.get_vocab().values()) >= vocab_size:
            raise ValueError('Tokenizer contains token IDs outside the model vocabulary')
        self.settings, self.tokenizer, self.device, self.model = dict(settings), tokenizer, device, model
        transformer_config = transformer_module.TransformersConfig(model, tokenizer, vocab_size=vocab_size, device=device)
        upstream = {key: settings[key] for key in
                    ('gamma', 'delta', 'hash_key', 'prefix_length', 'z_threshold', 'entropy_threshold')}
        upstream['algorithm_name'] = 'SWEET'
        with tempfile.TemporaryDirectory(prefix='codeguard-sweet-') as temp:
            path = Path(temp) / 'SWEET.json'
            path.write_text(json.dumps(upstream), encoding='utf-8')
            self.sweet = module.SWEET(str(path), transformer_config)
        self.processor = self.sweet.logits_processor

    def token_entropies(self, ids, chunk_size=64):
        """Entropy before each token, from code alone; cached prefixes remain intact."""
        import torch

        entropies = [None]  # The first token has no available preceding context.
        past = None
        context_limit = getattr(self.model.config, 'max_position_embeddings', None)
        if context_limit is not None and len(ids) > context_limit:
            raise ValueError('SWEET detector input exceeds the model context; refusing to truncate')
        with torch.inference_mode():
            for start in range(0, len(ids) - 1, chunk_size):
                inputs = ids[start:min(start + chunk_size, len(ids) - 1)].unsqueeze(0)
                output = self.model(input_ids=inputs, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                # FP32 matches the logits precision used by Transformers sampling.
                log_probs = output.logits.float().log_softmax(dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                if not torch.isfinite(entropy).all().item():
                    raise RuntimeError('Non-finite SWEET detector entropy; use a fresh float32 run')
                entropies.extend(entropy[0].cpu().tolist())
                del output, log_probs, entropy
        return entropies

    def score(self, text):
        import torch

        ids = self.tokenizer(text, add_special_tokens=False)['input_ids']
        result = {'status': 'too_short', 'token_count': len(ids), 'scored_tokens': 0,
                  'green_count': 0, 'z_score': None, 'is_watermarked': None,
                  'ignore_repeated_ngrams': self.settings['ignore_repeated_ngrams'],
                  'entropy_threshold': self.settings['entropy_threshold'],
                  'high_entropy_tokens': 0, 'low_entropy_tokens': 0,
                  'repeated_bigrams_skipped': 0}
        if len(ids) < 2:
            return result
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        entropies = self.token_entropies(tensor)
        seen, green, count = set(), 0, 0
        for index in range(1, len(ids)):
            eligible = entropies[index] > self.settings['entropy_threshold']
            result['high_entropy_tokens' if eligible else 'low_entropy_tokens'] += 1
            bigram = (ids[index - 1], ids[index])
            if self.settings['ignore_repeated_ngrams'] and bigram in seen:
                result['repeated_bigrams_skipped'] += 1
                continue
            seen.add(bigram)  # Keep the first occurrence, even if it is low entropy.
            if not eligible:
                continue
            greenlist = self.sweet.utils.get_greenlist_ids(tensor[:index])
            green += int((greenlist == tensor[index]).any().item())
            count += 1
        if not count:
            result['status'] = 'no_eligible_tokens'
            return result
        gamma = self.settings['gamma']
        z = (green - gamma * count) / math.sqrt(count * gamma * (1 - gamma))
        result.update(status='ok', scored_tokens=count, green_count=green, z_score=z,
                      is_watermarked=z > self.settings['z_threshold'])
        return result
