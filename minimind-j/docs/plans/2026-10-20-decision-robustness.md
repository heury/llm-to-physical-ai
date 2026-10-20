# Decision robustness implementation plan

Goal: Correct question-weighted gradient accumulation and evaluate instruction/choice-order sensitivity.
Architecture: Backpropagate summed question losses, divide accumulated gradients by the window's actual question count before clipping. A standalone CPU/CUDA evaluator compares original records against explicit instruction paraphrases and reversed choice lists; align predictions and soft labels by value.
Tech stack: Python, PyTorch, existing MiniMind-J checkpoint/collator/metrics.

1. Add regression tests for unequal microbatch gradients and final short windows in tests/test_typed_training.py; observe failure, then implement normalization in scripts/runpod_typed/train.py.
2. Add tests for non-mutating variants, soft-target reordering, semantic probability alignment, and missing paraphrases in tests/test_robustness.py; implement scripts/evaluate_robustness.py and an auditable paraphrase mapping.
3. Run the full pytest suite and a local trained-model CLI smoke. Document commands, metric interpretation and the unavailable local 64M weights. Do not infer full-model robustness from smoke fixtures.

Scope: Keep architecture and backbone unchanged. No retraining or new remote compute required. Existing approval covers implementation. This workspace is not a Git repository, so no commits/worktree.
