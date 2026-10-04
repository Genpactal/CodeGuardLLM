# CodeGuardLLM — week one

KGW watermark experiments on Python code, using Qwen2.5-Coder-1.5B-Instruct and
a pinned MarkLLM implementation. Week one provides a resumable 10-problem pilot:
20 completions, one clean and one watermarked per problem, with immediate detection.

## Start with the cloud notebook

1. Open `notebooks/week1_cloud.ipynb` in Colab or Kaggle and select a GPU runtime.
2. Upload `dist/codeguard-week1.zip` (build it with `python scripts/package_cloud.py`).
3. Run the notebook setup and validation cells, then generation.
4. Download the run archive and evaluate it locally using Docker as described below.

The notebook contains instructions for persistent Google Drive output in Colab
and Kaggle working-directory output. Model weights download on the first run.
No model fine-tuning, paid API, or API key is required. Availability of a free GPU
depends on the provider. The notebook never executes generated Python.

## Local installation

Use Python 3.11 or 3.12 and Git. From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[ml,data]'
.\.venv\Scripts\python.exe scripts/bootstrap_markllm.py
.\.venv\Scripts\python.exe -m codeguard validate --check-markllm
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux, create a virtual environment with `python3 -m venv .venv` and substitute
`.venv/bin/python` for the Python executable above. For configuration checks and
mock pipeline tests only, `pip install -e .` is sufficient; ML integration tests
are skipped until their dependencies and MarkLLM are available.

The bootstrap script downloads a fixed MarkLLM commit into `.deps/MarkLLM`.
It does not install MarkLLM's unrelated algorithm dependencies. Core package
versions are pinned in `pyproject.toml`; runtime package versions and GPU details
are also recorded in each run manifest. This is not a complete transitive lockfile.

## Generate and detect

```powershell
.\.venv\Scripts\python.exe -m codeguard generate --config configs/week1.yaml
.\.venv\Scripts\python.exe -m codeguard summarize --run-dir runs/week1
```

The default configuration uses CUDA and float16. For CPU experiments, copy the
configuration, change `model.device` to `cpu`, `model.dtype` to `float32`, and choose
a different output directory. Full-model CPU generation is intended as a fallback.

Repeat the same generate command to resume. Every completed sample is flushed to
JSONL. An interrupted final JSONL fragment is preserved in a recovery file before
truncation. Invalid complete records cause an error. A run rejects changed
parameters, task data, or runtime metadata; use a new output directory for changes.
Only one process/machine should write to a run directory at a time. An already
completed run can be checked without loading a model.

Standalone detection does not load model weights, but uses the matching tokenizer,
vocabulary, key, hashing scheme, and CPU/CUDA random-number backend:

```powershell
.\.venv\Scripts\python.exe -m codeguard detect --config configs/week1.yaml --file solution.py
```

Provide generated code only. Do not prepend the original task prompt. Using a CPU
detector on a CUDA-generated watermark changes the random partitions in this
implementation and invalidates the scores; keep the original device setting.

## Evaluate correctness with Docker

Start Docker Desktop with its Linux engine. Build the evaluator once:

```powershell
docker build -t codeguard-eval:week1 sandbox
$env:CODEGUARD_DOCKER_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_evaluation.py -v
.\.venv\Scripts\python.exe -m codeguard evaluate --run-dir runs/week1
.\.venv\Scripts\python.exe -m codeguard summarize --run-dir runs/week1
```

For cloud outputs, extract the downloaded archive so that `runs/week1` contains
`manifest.json`, `tasks.json`, and `generations.jsonl`, then run evaluation.
Evaluation requires only the base project installation, not GPU/model dependencies.
The run is portable between machines for evaluation, even though generation
resumption requires a matching environment.

Each sample runs in a fresh container with no network, no host filesystem mounts,
a read-only root, a non-root user, dropped capabilities, and process/memory/time
limits. There is no host execution fallback. A stopped Docker engine is an
infrastructure error, never an incorrect-answer result. Reference solutions are
checked before generated samples. Tests, assertion failures, and timeouts are
evaluated inside the container. An early process exit is not counted as a pass.

The container is a practical isolation boundary for research samples, not a proof
against deliberately adversarial container exploits or manipulation of a Python
test harness. Do not run it with privileged mode or mount credentials into it.
The Docker base image is a version tag; the resolved evaluator image ID is recorded
and must remain unchanged when resuming evaluations.

## Files and outputs

| Location | Purpose |
|---|---|
| `configs/week1.yaml` | Ten development problems, one seed, two conditions |
| `configs/humaneval.yaml` | Pinned 164-problem benchmark for later work |
| `data/dev.jsonl` | Ten project-authored problems, references, and tests |
| `src/codeguard/backend.py` | Hugging Face model adapter and paired seeded sampling |
| `src/codeguard/watermark.py` | MarkLLM integration and optional repeated-bigram filtering |
| `src/codeguard/pipeline.py` | Resumable generation and immediate detection |
| `src/codeguard/evaluation.py` | Docker evaluation and reference validation |
| `src/codeguard/reporting.py` | Pilot summary and correctness accounting |
| `notebooks/week1_cloud.ipynb` | Colab/Kaggle setup, generation, and export |
| `tests/` | Storage, extraction, generation, watermark, and evaluator checks |
| `docs/week1_protocol.md` | Experimental decisions, limitations, and next milestones |

Each run contains:

- `manifest.json`: experiment identity, configuration, revisions, runtime metadata.
- `tasks.json`: exact task snapshot, including tests; only `prompt` enters generation.
- `generations.jsonl`: raw continuation, token IDs, extracted code, timing, and scores.
- `evaluation_manifest.json`, `references.jsonl`, `evaluations.jsonl`: evaluator
  identity, reference checks, and per-sample correctness results, created on evaluation.
- `summary.json`: counts, Pass@1 when complete, mean Z-score, positive rate, and speed.

`pass_at_1: null` means evaluation is incomplete, not zero correctness. Extraction
errors, invalid Python, and missing entry points count as incorrect once evaluated.
Too-short detector inputs remain explicitly unscored. Raw and extracted-code scores
are stored separately; prompt tokens are excluded.

## Scope and verification

Week one implements the experiment infrastructure. It does not yet provide final
ROC/AUC plots, human negative controls, confidence intervals, parameter tuning,
multi-model attribution, or edit robustness; those belong to weeks two through four.
Pilot numbers from ten simple tasks are not benchmark conclusions.

Tests with a tiny random Transformers model and synthetic tokens exercise real
generation and MarkLLM code without downloading weights. Pipeline tests use clearly
labelled doubles and do not measure model quality. Real Docker tests are opt-in.
The full Qwen cloud run is an additional acceptance check, not simulated by these tests.

Sources: [KGW paper](https://arxiv.org/abs/2301.10226),
[MarkLLM](https://github.com/THU-BPM/MarkLLM),
[Qwen model](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
[HumanEval](https://github.com/openai/human-eval).
