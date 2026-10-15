"""SWEET integration against pinned MarkLLM and a tiny offline causal model."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from codeguard.backend import HFBackend
from codeguard.config import load_config, portable_config
from codeguard.experiments import comparison_configs
from codeguard.sweet import SWEETAdapter
from codeguard.storage import digest
from test_backend import AVAILABLE, ToyTokenizer

ROOT = Path(__file__).resolve().parents[1]


class SweetConfigTests(unittest.TestCase):
    def test_sweet_is_explicit_and_cannot_resume_kgw(self):
        kgw = load_config(ROOT / 'configs/gpu.yaml')
        sweet = load_config(ROOT / 'configs/sweet.yaml')
        self.assertNotIn('algorithm', kgw['watermark'])  # preserve legacy identities
        self.assertEqual(sweet['watermark']['algorithm'], 'SWEET')
        self.assertEqual(sweet['dataset'], kgw['dataset'])
        self.assertEqual(sweet['generation'], kgw['generation'])
        self.assertNotEqual(digest(portable_config(kgw)), digest(portable_config(sweet)))
        configs = comparison_configs(ROOT / 'configs/sweet.yaml', ROOT / '.tmp/unused-sweet')
        self.assertTrue(all(c['name'].startswith('sweet_comparison-') for c in configs.values()))

    def test_invalid_entropy_and_unknown_algorithms_are_rejected(self):
        source = yaml.safe_load((ROOT / 'configs/sweet.yaml').read_text())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.yaml'
            cases = [('entropy_threshold', -1), ('entropy_threshold', float('nan')),
                     ('algorithm', 'unknown'), ('algorithm', 'KGW')]
            for key, value in cases:
                config = deepcopy(source)
                config['watermark'][key] = value
                path.write_text(yaml.safe_dump(config))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    load_config(path)
            del source['watermark']['entropy_threshold']
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, 'requires'):
                load_config(path)

    def test_sweet_notebook_uses_isolated_source_and_results(self):
        notebook = json.loads((ROOT / 'notebooks/sweet_comparison_colab.ipynb').read_text(encoding='utf-8'))
        sources = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call) and len(node.args) > 1
                            and isinstance(node.args[0], ast.Constant) and node.args[0].value == '-c'):
                        ast.parse(ast.literal_eval(node.args[1]))
                self.assertEqual(cell['outputs'], [])
        self.assertIn('/content/CodeGuardLLM-SWEET', sources['project'])
        self.assertIn('codeguard-sweet-colab.zip', sources['project'])
        self.assertIn('configs/sweet.yaml', sources['install'])
        self.assertIn("/ 'sweet_comparison'", sources['install'])
        self.assertIn("'config': portable_config(config)", sources['install'])
        self.assertNotIn("'evaluate'", ''.join(sources.values()))


@unittest.skipUnless(AVAILABLE, 'Install ML dependencies and bootstrap MarkLLM')
class SweetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        cls.torch = torch
        torch.manual_seed(123)
        cls.model_config = GPT2Config(vocab_size=128, n_positions=64, n_embd=32,
                                     n_layer=1, n_head=2, eos_token_id=0, pad_token_id=0)
        cls.model = GPT2LMHeadModel(cls.model_config).eval()
        cls.config = load_config(ROOT / 'configs/sweet.yaml')

    def adapter(self, **overrides):
        return SWEETAdapter(self.config['markllm_path'], {**self.config['watermark'], **overrides},
                            ToyTokenizer(), 128, 'cpu', self.model)

    def test_generation_gates_bias_by_entropy(self):
        wm = self.adapter()
        ids = self.torch.tensor([[3, 7]])
        uniform = self.torch.zeros((1, 128))
        biased = wm.processor(ids, uniform.clone())
        green = wm.sweet.utils.get_greenlist_ids(ids[0])
        expected = uniform.clone()
        expected[0, green] += self.config['watermark']['delta']
        self.assertTrue(self.torch.equal(biased, expected))
        certain = uniform.clone()
        certain[0, 17] = 100
        self.assertTrue(self.torch.equal(wm.processor(ids, certain.clone()), certain))
        zero = self.adapter(delta=0)
        self.assertTrue(self.torch.equal(zero.processor(ids, uniform.clone()), uniform))

    def test_cached_entropy_matches_full_prefix_and_upstream_scores(self):
        wm = self.adapter(ignore_repeated_ngrams=False)
        ids = self.torch.tensor([3, 7, 3, 7, 11, 19, 23, 28, 29, 32, 54])
        upstream_entropies = wm.sweet.utils.calculate_entropy(self.model, ids)
        cached = wm.token_entropies(ids, chunk_size=3)
        for actual, expected in zip(cached[1:], upstream_entropies[1:]):
            self.assertAlmostEqual(actual, expected, places=5)
        upstream_z, _, weights = wm.sweet.utils.score_sequence(ids, upstream_entropies)
        score = wm.score(' '.join(map(str, ids.tolist())))
        self.assertEqual(score['scored_tokens'], sum(w == 1 for w in weights))
        self.assertAlmostEqual(score['z_score'], upstream_z)
        repeated = self.adapter().score('3 7 3 7 3 7')
        self.assertEqual(repeated['scored_tokens'], 2)
        self.assertEqual(repeated['repeated_bigrams_skipped'], 3)

    def test_biased_tokens_are_detected_with_the_matching_key(self):
        wm = self.adapter()
        ids, seen = [19], set()
        for _ in range(30):
            biased = wm.processor(self.torch.tensor([ids]), self.torch.zeros((1, 128)))
            candidates = [i for i in range(128) if biased[0, i] > 0 and (ids[-1], i) not in seen]
            chosen = candidates[len(ids) % len(candidates)]
            seen.add((ids[-1], chosen))
            ids.append(chosen)
        text = ' '.join(map(str, ids))
        score = wm.score(text)
        self.assertEqual(score['green_count'], 30)
        self.assertTrue(score['is_watermarked'])
        self.assertLess(self.adapter(hash_key=7777777).score(text)['z_score'], score['z_score'])

    def test_first_occurrence_and_entropy_boundary(self):
        wm = self.adapter(entropy_threshold=1)
        # 3->7 first occurs at low entropy, and is not revived by later repetition.
        with patch.object(wm, 'token_entropies', return_value=[None, 1.0, 2.0, 3.0]):
            score = wm.score('3 7 3 7')
        self.assertEqual(score['scored_tokens'], 1)
        self.assertEqual(score['low_entropy_tokens'], 1)

    def test_no_eligible_tokens_short_text_and_invalid_entropy(self):
        wm = self.adapter(entropy_threshold=100)
        result = wm.score('3 7 11')
        self.assertEqual(result['status'], 'no_eligible_tokens')
        self.assertIsNone(result['z_score'])
        self.assertIsNone(result['is_watermarked'])
        for text in ('', '3'):
            self.assertEqual(wm.score(text)['status'], 'too_short')
        with self.assertRaisesRegex(RuntimeError, 'requires'):
            SWEETAdapter(self.config['markllm_path'], self.config['watermark'], ToyTokenizer(), 128, 'cpu')
        with self.assertRaisesRegex(ValueError, 'context'):
            wm.token_entropies(self.torch.ones(65, dtype=self.torch.long))

    def test_backend_loads_weights_for_sweet_detection_and_generates(self):
        config = deepcopy(self.config)
        config['model'].update(device='cpu', dtype='float32')
        config['generation']['max_new_tokens'] = 8
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=ToyTokenizer()), \
             patch('transformers.AutoConfig.from_pretrained', return_value=self.model_config), \
             patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=self.model) as loader:
            backend = HFBackend(config, load_model=False)
            loader.assert_called_once()
        self.assertEqual(backend.metadata['algorithm'], 'SWEET')
        self.assertEqual(backend.score('3 7 11')['status'], 'ok')
        first = backend.generate('def f():', 'test/sweet', 42, 'watermarked')
        second = backend.generate('def f():', 'test/sweet', 42, 'watermarked')
        self.assertEqual(first['generated_token_ids'], second['generated_token_ids'])
        self.assertEqual(first['prompt_tokens'], 3)
        self.assertLessEqual(first['generated_tokens'], 8)


if __name__ == '__main__':
    unittest.main()
