# CodeGuardLLM

Experiments with KGW and SWEET watermarks in Python code: generation, watermark detection,
correctness evaluation, human-reference controls, and research reporting. The project uses
Hugging Face models and a pinned MarkLLM implementation.

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/watermark_comparison.ipynb` | Full core workflow: generation, reference controls, Docker correctness, plots, written report |
| `notebooks/watermark_comparison_colab.ipynb` | KGW on Colab: generation, reference scoring, reports, and Drive checkpoints |
| `notebooks/sweet_comparison_colab.ipynb` | SWEET on Colab through VS Code, with separate results and checkpoints |

For local work, open the comparison notebook in VS Code and select
**Select Kernel → Python Environments → .venv\Scripts\python.exe**. Run the cells
in order. The notebooks use CPU and float32 through `configs/local.yaml`. They
keep short report-style Markdown; new results appear after execution.

Models download on first use and load one at a time. CPU generation is slower than
the earlier cloud run. The comparison uses Qwen2.5-Coder-1.5B-Instruct on all 164
HumanEval problems, producing 984 completions (164 × clean/watermarked × three
strengths). It also scores 164 reference solutions once and shares these control
scores across strengths. References are never supplied to the model.

Results stay in `runs/local/humaneval/watermark_comparison`.
Generation and reference scoring can resume after interruption. The notebook
creates a local results ZIP containing the report and its underlying records.
No Colab connection or Google Drive authorization is needed for local experiments.

## GPU acceleration on this laptop

The separate `configs/gpu.yaml` uses the same model and all 164 tasks with CUDA and
bfloat16. The comparison honors the configured precision. In the first notebook
cell, `USE_GPU = True` selects `.venv-gpu`, this configuration, and the separate
`runs/gpu/humaneval/watermark_comparison` directory. The default remains CPU.

To create that environment on a new installation:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-gpu\Scripts\python.exe -m pip install -e '.[ml,data,analysis,notebook]'
```

The CUDA wheel follows the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
Setup checks CUDA and bfloat16 support. Stop the CPU generation cell before starting
a GPU experiment, then rerun notebook setup with `USE_GPU = True`. The notebook's
display kernel may stay on the existing environment: experiment commands use the
selected environment's Python executable. Old CPU samples remain saved but must
not be mixed with GPU samples or rescored with the GPU detector. In particular,
changing devices starts a fresh, comparable run rather than resuming CPU records.

On this RTX 3050 laptop, a six-completion smoke benchmark reached 30.44 generated
tokens/second including detection (3.03 seconds/completion on average), with
2.91 GiB peak PyTorch-allocated GPU memory. This is a small timing check, not a
full-runtime estimate or correctness result. Its record is `runs/verification/gpu_benchmark.json`.

## Google Colab through VS Code

1. Install or update the official [Colab extension by Google](https://marketplace.visualstudio.com/items?itemName=Google.colab).
2. Open `notebooks/watermark_comparison_colab.ipynb` locally in VS Code.
3. Click **Select Kernel > Select Another Kernel** (if shown), then **Colab >
   New Colab Server**. Choose an available GPU such as T4 and sign in with Google.
4. In VS Code Explorer, right-click `.tmp/codeguard-colab.zip` and choose
   **Upload to Colab**. Wait for completion; the notebook expects it at
   `/content/codeguard-colab.zip` on the remote server.
5. Run cells from the top and approve Google Drive authorization. The project
   archive is cached in Drive for later sessions.

The [official user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
describes kernel selection and file transfer. The notebook stays on the laptop;
its cells run on Google's server. Local project files are not automatically synced.
The browser is needed for account authorization, but notebook work stays in VS Code.

`USE_VSCODE=True` uses extension file transfers. After the export cell, open the
Colab sidebar's **Contents**, refresh, and right-click `codeguard-colab-results.zip`
then **Download...**. For the Colab website instead, upload the notebook there,
select a GPU runtime, and set `USE_VSCODE=False` to enable browser file dialogs.

The notebook creates a separate environment with pip, pinned CUDA PyTorch, and the
project dependencies. There is no kernel restart between installs and generation.
It uses native bfloat16 where supported; T4 uses float32 to avoid the previous
float16 numerical failure. The experiment still has 984 completions and 164
reference controls. A faster completion time is not guaranteed by moving to Colab.

Generation runs on local Colab storage. During progress, a ZIP checkpoint is saved
roughly every 60 seconds and at each command's end to
`MyDrive/CodeGuardLLM/colab/<runtime-id>/results.zip`. Only one session should write
that run. After disconnection, rerun the notebook: matching GPU and software
versions restore the latest checkpoint and skip saved samples. Work since the last
successful checkpoint may need repeating. A different runtime gets a separate
folder; laptop records are kept separately. Drive and GPU availability are subject
to [Colab's limits](https://research.google.com/colaboratory/faq.html).

If Drive authorization fails, use **Colab: Mount Google Drive to Server...** from
the VS Code command palette, execute the cell it adds, then rerun setup.
Alternatively, set `USE_DRIVE=False`; then storage is temporary and results must be
manually downloaded using the export cell before the runtime ends. That cell also
works after interrupting generation. The notebook never invents missing results.

Correctness remains pending in Colab. Download `codeguard-colab-results.zip`, start
Docker Desktop on the laptop, and extract into a new folder before evaluation:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\codeguard-colab-results.zip" -DestinationPath runs/colab-import
.\.venv\Scripts\python.exe scripts/compare_watermarks.py evaluate --output runs/colab-import/watermark_comparison
.\.venv\Scripts\python.exe scripts/compare_watermarks.py analyze --output runs/colab-import/watermark_comparison
```

Evaluation uses the saved task snapshots and experiment manifests. It does not
load the model or redo GPU detection. The final report is
`runs/colab-import/watermark_comparison/analysis/report.md`.

## SWEET on Colab through VS Code

`notebooks/sweet_comparison_colab.ipynb` runs the same Qwen 1.5B model on all 164
HumanEval tasks, with clean and watermarked generation at delta 1, 2, and 3 (984
completions). The new `configs/sweet.yaml` selects SWEET explicitly. KGW configs,
existing notebooks, and saved KGW runs keep their original identities.

1. Open the SWEET notebook in VS Code and select **Colab > New Colab Server > GPU**.
2. Right-click `.tmp/codeguard-sweet-colab.zip` in Explorer and select **Upload to
   Colab**, then run the notebook from the top. This archive is prepared locally;
   after cloning or changing the project, rebuild it with:

```powershell
.\.venv\Scripts\python.exe scripts/package_colab.py --sweet
```

The notebook extracts into `/content/CodeGuardLLM-SWEET`, checks generation and
scoring on two short setup examples, and installs packages in its own environment.
Setup examples are not included in the experiment. GPU precision is bfloat16 where
natively supported, otherwise float32. Drive checkpoints are stored under
`MyDrive/CodeGuardLLM/colab-sweet/<runtime-and-config-id>/results.zip`; changed
settings or hardware get a separate folder. Keep only one writer per run.

[SWEET](https://arxiv.org/abs/2305.15060) applies the green-token bias only above an
entropy cutoff. We use the [pinned MarkLLM implementation](https://github.com/THU-BPM/MarkLLM/blob/0a4fe8c642b31b8a8c4993615b1c70a0c300d918/watermark/sweet/sweet.py)
with gamma 0.25, entropy cutoff 0.9 nats (an untuned upstream default), and Z > 4.
The generation model and sampling settings match the KGW configuration. The native
SWEET green-list seeding differs from native KGW; this compares implementations,
not a controlled change of entropy gating alone.

SWEET detection loads Qwen to estimate entropy. It uses cached 64-token chunks to
bound vocabulary-logit memory and FP32 entropy arithmetic. The project applies its
existing repeated-bigram filter: only the first occurrence can contribute, and it
must exceed the entropy cutoff. This filtering extends upstream SWEET. Empty,
one-token, and zero-eligible-token outputs remain unscored instead of crashing or
being labelled negative. The first token is never scored. Detection receives only
the supplied text; removing prompts and Markdown can change early entropy gates.
Recorded eligible-token counts describe this detector, not an entropy measurement
of the original generation context.

Reference controls are rescored using SWEET and its model. They cannot reuse KGW
scores. ROC/AUROC, score distributions, Pass@1, correctness among detected solutions,
and joint correct-and-detected rates use the same report pipeline. No improved
performance is claimed before the full experiment. Compare methods at calibrated,
matched false-alarm rates; a common Z threshold alone does not ensure equal rates.

The export cell creates `/content/codeguard-sweet-results.zip`. Download it through
**Colab > Contents > Download...** and evaluate on the laptop with Docker:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\codeguard-sweet-results.zip" -DestinationPath runs/sweet-import
.\.venv\Scripts\python.exe scripts/compare_watermarks.py evaluate --config configs/sweet.yaml --output runs/sweet-import/sweet_comparison
.\.venv\Scripts\python.exe scripts/compare_watermarks.py analyze --config configs/sweet.yaml --output runs/sweet-import/sweet_comparison
```

These two laptop steps use saved scores and do not load Qwen. Pass@1 and detected
correctness stay pending in Colab until Docker evaluation is available.

## Installation

Use Python 3.11 or 3.12 and Git. From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[ml,data,analysis,notebook]'
.\.venv\Scripts\python.exe scripts/bootstrap_markllm.py
.\.venv\Scripts\python.exe -m codeguard validate --check-markllm
```

The bootstrap script places the pinned MarkLLM checkout in `.deps/MarkLLM`.
On Linux, use `.venv/bin/python`. Runtime versions, model revisions, and device
details are recorded in each run manifest; this is not a full transitive lockfile.

## Commands

The same experiments are available outside Jupyter:

```powershell
.\.venv\Scripts\python.exe scripts/compare_watermarks.py generate
.\.venv\Scripts\python.exe scripts/compare_watermarks.py controls
.\.venv\Scripts\python.exe scripts/compare_watermarks.py evaluate
.\.venv\Scripts\python.exe scripts/compare_watermarks.py analyze
```

`python scripts/compare_watermarks.py run` executes all four steps with the project
interpreter. `evaluate` checks Docker's Linux engine and builds the evaluator image
when needed. If Docker is stopped, evaluation gives an actionable error; `analyze`
can still produce a report marked `correctness_pending` after generation and
reference scoring. A report becomes `complete` only when correctness is evaluated
for every generated sample. None of these statuses imply a positive research finding.

For a single clean/watermarked baseline (328 completions):

```powershell
.\.venv\Scripts\python.exe -m codeguard generate
.\.venv\Scripts\python.exe -m codeguard summarize
.\.venv\Scripts\python.exe -m codeguard detect --file solution.py
```

These commands default to the local CPU configuration. The baseline saves to
`runs/local/humaneval/baseline`. Detection receives only generated code, without the prompt.
CPU and CUDA use different KGW random partitions; score and resume with the same
device and runtime used for generation. Changed settings require a new output
directory. Complete runs can be analyzed without regenerating their samples.

## Correctness evaluation

Start Docker Desktop's Linux engine. The comparison notebook builds the evaluator
when needed. The equivalent commands are:

```powershell
docker build -t codeguard-eval:v1 sandbox
.\.venv\Scripts\python.exe scripts/compare_watermarks.py evaluate
.\.venv\Scripts\python.exe scripts/compare_watermarks.py analyze
```

For the single baseline, use `python -m codeguard evaluate` with the project interpreter.
Each sample executes in a fresh non-root container with no network or host mounts,
a read-only root, and memory/process/time limits. Reference solutions are checked
first. Infrastructure errors stop evaluation; they do not count as wrong answers.
There is no host execution fallback. Pass@1 remains null until evaluation completes.
Docker is practical isolation for research samples, not a proof against deliberate
container exploits or test-harness manipulation.

## Reading the results

The comparison reports mean Z, scored/unscored counts, positive rates at Z > 4,
ROC/AUROC against both clean code and human references, and Pass@1 when available.
The report shows watermark strength versus detection and correctness, code-length
groups, truncated-output counts, and examples of weak/strong watermark scores.
ROC/AUROC use scorable code. Clean/reference positive rates estimate false alarms;
watermarked rates estimate detection. References are reconstructed as the HumanEval
prompt plus canonical solution, including docstrings; they are never executed during
scoring. Repeated clean outputs and shared references are not independent controls.

Correctness among detected solutions is the fraction of watermarked outputs with
Z above the configured threshold that pass all tests. The report also gives the
fraction of all watermarked outputs that are both detected and correct, including
unscored outputs in that denominator. Counts accompany the rates: no detections
means conditional correctness is unavailable, and missing tests remain pending.
These metrics are saved in `analysis/detected_correctness.csv` and shown in both
comparison notebooks. A few detected solutions cannot establish general accuracy.

The `analysis/` folder contains `report.md`, `core_summary.json`, CSV tables,
`controls_detection.png`, `tradeoff.png`, selected examples, and provenance hashes.
The written report derives its numbers from saved records and discusses KGW,
HumanEval, limitations, and pending evaluation. It does not invent missing results.

The default local configuration now uses the full HumanEval test split, pinned to
a fixed dataset revision in `configs/local.yaml`, using CPU and float32.
The three strengths are compared descriptively; selecting the best one on these
same tasks does not produce an independent held-out result. Threshold calibration,
repeated seeds, confidence intervals, and an independent human-code corpus remain
possible follow-up work. Robustness is a separate follow-up experiment described below.
Fingerprinting is excluded from the core workflow. Earlier attribution files are retained as a deferred prototype and are
not required for this project stage.
Tests and reference solutions never enter generation prompts.

## Robustness of saved code

Open `notebooks/robustness.ipynb` to prepare and inspect attacks across the three SWEET
strengths. The notebook keeps scoring and Docker evaluation opt-in because they can
take time. The same CLI supports saved KGW runs. No solutions are regenerated.

Each source sample, clean and watermarked, gets an unchanged identity baseline plus
four deterministic attacks: comment removal, AST formatting normalization, docstring
removal, and conservative local-variable renaming. Renaming preserves parameters and
function names and skips nested scopes, comprehensions, imports, exception bindings,
pattern matching, f-strings, and direct reflection calls. Skipped/no-op attacks and
invalid code remain explicit. Docstring removal and renaming also apply AST formatting;
use the `reformat` arm to distinguish this shared effect. Test passing is required to
claim a correctness-preserving evasion; transformations are not assumed equivalent.

For one saved strength, run these steps from the project root:

```powershell
$robustnessRun = 'runs/sweet-import/0eedc6730e049860/sweet_comparison/delta-3'
.\.venv-gpu\Scripts\python.exe scripts/robustness.py prepare --run-dir $robustnessRun
.\.venv-gpu\Scripts\python.exe scripts/robustness.py score --run-dir $robustnessRun
.\.venv\Scripts\python.exe scripts/robustness.py evaluate --run-dir $robustnessRun
.\.venv\Scripts\python.exe scripts/robustness.py analyze --run-dir $robustnessRun
```

Start Docker Desktop for `evaluate`. `score` needs the ML dependencies and pinned
MarkLLM checkout; SWEET loads the generation model to recompute entropy. Defaults use
the saved model/device/precision. For a deliberate CPU run, use `--device cpu --dtype
float32`. Identity and attacked code are both rescored in the new detector environment;
original generation scores are not mixed into comparisons. Saved scoring metadata
prevents resuming with changed settings or a changed detector environment. A completed
score stage can be reused without loading the model. `run` performs all four stages.

Artifacts are kept under `RUN_DIR/robustness` (or `--output`): frozen variants, source
hashes, detector/evaluator manifests, resumable scores and evaluations, JSON/CSV
summaries, `report.md`, and a plot when both stages finish. Source files remain intact.
Preparing variants records the Python version because AST rendering can differ across
versions; later scoring/evaluation/reporting can run elsewhere on the frozen variants.
Keep the source run and its `robustness` folder together when transferring machines.

Reports include matched-clean AUROC, detection retention, paired Z-score changes,
Pass@1, and correctness retention. Correct evasion requires a baseline that both
passes and is detected, a changed candidate that still passes, and a valid attacked
score at or below the fixed threshold. Unscored code is never counted as an evasion.
Zero eligible baselines gives an unavailable rate, not a successful attack. Missing
stages remain pending, and no-op/failed attacks stay in denominators. This suite does
not cover adaptive or LLM-assisted rewriting, and small detected subsets remain weak
evidence even when a reported rate is high.

The format transformation uses Python's [AST unparser](https://docs.python.org/3/library/ast.html#ast.unparse).
Comments are located with the [Python tokenizer](https://docs.python.org/3/library/tokenize.html).

## Project structure

| Location | Purpose |
|---|---|
| `configs/local.yaml` | All 164 HumanEval problems, CPU, float32 |
| `configs/pilot.yaml`, `configs/pilot_fp32.yaml` | Earlier CUDA pilot configurations |
| `data/dev.jsonl` | Small fixtures for software tests and the archived pilot |
| `src/codeguard/backend.py`, `watermark.py` | Model generation and MarkLLM integration |
| `src/codeguard/pipeline.py` | Resumable generation and immediate detection |
| `src/codeguard/experiments.py` | Comparison settings and registered model keys |
| `src/codeguard/analysis.py` | Detection and attribution metrics, tables, plots |
| `src/codeguard/controls.py` | Resumable, provenance-checked reference-code scores |
| `src/codeguard/core_report.py` | Core comparison plots and a results-based written report |
| `src/codeguard/evaluation.py`, `reporting.py` | Docker correctness checks and summaries |
| `scripts/` | Experiment commands and MarkLLM bootstrap |
| `tests/` | Software checks, including opt-in Docker integration tests |
| `docs/protocol.md` | Experimental choices and limitations |

Each run stores its manifest, task snapshot, generated text/token IDs, extracted
code, timing, and detection scores. Evaluation adds reference/sample outcomes.
Reports include JSON/CSV tables and PNG plots. Truncated or invalid outputs are
retained; there is no automatic code repair or best-of-N selection.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Set `CODEGUARD_DOCKER_TESTS=1` to include real Docker checks after building the image.
Synthetic-score tests and a tiny random model validate software behavior, not model
quality. Real generated results must come from an executed experiment.

Sources: [KGW paper](https://arxiv.org/abs/2301.10226),
[MarkLLM](https://github.com/THU-BPM/MarkLLM),
[Qwen model](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
[HumanEval](https://github.com/openai/human-eval).
