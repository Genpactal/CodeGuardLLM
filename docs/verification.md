# Implementation verification — 2026-10-04

- 22 tests discovered: 20 passed, 2 Docker integration tests skipped.
- Actual pinned MarkLLM checked on CPU: logits/detector agreement, zero bias,
  deterministic scoring, wrong-key control, repeated bigrams, and short inputs.
- Actual Transformers generation exercised with a tiny randomly initialized model;
  no trained model weights were used in this test.
- Pinned Qwen tokenizer and model configuration downloaded successfully. The chat
  template and text-only detector were checked with the actual tokenizer on CPU.
- CLI validation confirmed 10 development tasks and 20 planned completions.
- All authored reference solutions passed their own tests in the trusted fixture test.
- Python sources compiled and all notebook code cells parsed successfully.
- The source ZIP excludes virtual environments, dependency checkouts, and run data.

Not verified here: full Qwen generation on a cloud GPU, notebook execution in a real
Colab/Kaggle session, or Docker container execution. The local Docker CLI was present,
but its engine was unavailable and its Desktop launcher could not find the required
installation registry entry. Run the opt-in Docker checks from the README on a
machine with a working Linux Docker engine.

Local tests were run with temporary files redirected to the repository's `.tmp`
directory because the execution environment denied access inside its default
Windows temporary directory. This did not require changes to the library behavior.
