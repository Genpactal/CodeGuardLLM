# Experiment protocol

## Current HumanEval experiments

Local notebooks use all 164 HumanEval test problems, pinned by dataset revision.
Each setting produces 328 completions (one clean and one watermarked per task);
three strengths or three models produce 984 completions per notebook. Local CPU
runs use float32; GPU runs use explicit precision recorded in their manifests.
Each runtime and algorithm has separate result directories.
Tests and reference solutions are withheld from generation. Strength comparisons
are descriptive; choosing a strength on these tasks is not held-out evaluation.

SWEET uses `configs/sweet.yaml`: the same model, tasks, gamma=0.25, key and sampling
settings as KGW, with an untuned entropy cutoff of 0.9 nats and delta 1, 2, 3.
Generation uses pinned MarkLLM SWEET. Detection recomputes text-only entropy with
cached 64-token chunks and FP32 arithmetic, then retains high-entropy positions
and filters repeated bigrams by first occurrence. The first token is excluded;
zero eligible positions are unscorable. The detector needs the model, including
for reference controls. Native SWEET and KGW green-list seeding differs, so this
is a comparison of implementations rather than entropy gating alone. Removed
prompts and Markdown may affect early entropy estimates. Correctness remains a
separate Docker test. No SWEET performance claim is made before the full run.

Robustness is a separate paired experiment on frozen saved code, implemented by
`scripts/robustness.py`. Identity, comment removal, AST reformatting, docstring removal,
and conservative local-variable renaming are applied to both clean and watermarked
samples. Identity and attacked variants are rescored together with recorded detector
metadata. The original generation detector scores are not reused. Docker validates
canonical references before evaluating variants. A successful evasion requires an
originally correct and detected sample, an actually changed and still-correct variant,
and a valid attacked score at or below the unchanged threshold. Unscored variants do
not establish evasion. Report applicability, no-op/error counts, baseline eligible
counts, and missing stages alongside rates. AST-based edits share formatting changes;
compare with the reformat arm. No adaptive attack or semantic-equivalence guarantee is
claimed. Original experiments and their core summaries remain separate.

The following records the earlier development pilot.

The implemented core workflow includes reference-code detection, ROC/AUROC against
clean and reference negatives, Docker Pass@1, strength/quality plots, scored-length
groups, and a written report. Reference controls are complete prompt-plus-canonical
solutions, retaining docstrings. They are scored once per dataset/detector and
shared across delta values, without pretending they are independent observations.
At that stage, fingerprinting and robustness were deferred. A report is marked complete only after
generation, controls, and all correctness evaluations are complete.

## Question and scope

Can the infrastructure generate Python with KGW, preserve experiment identity across
interrupted sessions, and evaluate correctness separately from watermark evidence?
The initial pilot validates the pipeline. It does not establish reliable watermark detection.

## Experiment components

1. Set up Python, pinned MarkLLM/model revisions, configuration, and separate development data.
2. Integrate chat prompting, clean/watermarked sampling, and prompt-free output extraction.
3. Add token scoring, repeated-bigram filtering, manifests, and interruption recovery.
4. Add isolated correctness evaluation and end-to-end integration tests.
5. Run the 10-task cloud pilot, inspect failures, measure throughput, and export results.

The repository supplies steps 1–4 and the notebook for step 5. The real pilot requires
a GPU session; Docker tests require a running Linux Docker engine.

## Fixed initial choices

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-Coder-1.5B-Instruct, revision in YAML |
| Dataset | 10 project-authored development tasks |
| Seeds | 42; actual per-task seed derived from task ID and seed |
| Conditions | Clean and KGW, same prompt and sampling settings |
| Sampling | temperature 0.8, top-p 0.95, top-k disabled |
| Maximum output | 512 new tokens, with truncation recorded |
| KGW | gamma 0.25, delta 2.0, h=1, time/left scheme |
| Key | Fixed research key in YAML, not a production secret |
| Detection | Z > 4 reference threshold, repeated bigrams ignored |
| Quality | One sampled completion per task/condition; fraction passing all tests |

The KGW adapter uses the exact upstream green lists and the model output vocabulary
size. It rejects tokenizer IDs outside that vocabulary. Generation and detection
must use the same random-number backend; changing between CUDA and CPU changes
the partitions. This property is recorded, not silently normalized away.

For text-only detection, the first token is unscored because its preceding generation
context is unavailable. T is the number of scored distinct bigrams when filtering is
enabled, not the total text length. Green-list size is rounded down upstream; the
implementation retains upstream gamma in the nominal Z-score. Treat the Z > 4 decision
as uncalibrated on code. No normal-null false-positive guarantee is claimed.

## Output handling and accounting

- A single Python fenced block is extracted, or the whole unfenced continuation.
- Internal whitespace is preserved. There is no code repair or automatic formatting.
- Complete function definitions are requested; bodies alone fail the entry-point check.
- Raw continuation and extracted code are both retained and independently scored.
- Text re-tokenization can differ from generation token IDs; both artifacts are saved
  for later analysis. Special tokens are removed from decoded text.
- Invalid syntax and extraction failures remain in quality denominators.
- Unexpected generation failures stop the run; retries resume missing samples with
  the same task seed. There is no automatic best-of-N retry selection.
- Evaluation validates references first. Infrastructure failures stop evaluation.
- Partial evaluations show null Pass@1 until all expected samples are evaluated.

## Acceptance checks

- The development fixture contains ten tasks and plans twenty completions; the default CLI now plans 328 HumanEval completions.
- Generation resumes after interruption without duplicate samples.
- Modified configurations, task snapshots, or runtime metadata are rejected.
- The real KGW logits processor and detector agree on synthetic token sequences.
- A wrong-key control loses the synthetic watermark signal.
- Repeated bigrams and short inputs have explicit, tested behavior.
- A tiny offline Transformers model exercises actual clean and watermarked generation.
- Docker evaluates reference solutions, rejects incorrect programs and early exits,
  and terminates infinite loops.
- A real Qwen GPU pilot creates twenty records and can export them for local evaluation.

## Further experiments

For independent tuning, use separate development prompts and freeze settings before
reporting held-out results. The current HumanEval strength sweep is descriptive. HumanEval's tests and
reference solutions are public but must remain withheld from generation prompts.
Possible follow-up work adds independent human controls, threshold calibration,
repeated seeds, problem-level confidence intervals, fingerprinting, and controlled edits.

Estimate runtime from measured per-condition throughput; do not extrapolate from
model size alone. On a new GPU or dependency environment, start a separate run.
