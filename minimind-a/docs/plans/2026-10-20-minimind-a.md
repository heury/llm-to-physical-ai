> Superseded on 2026-10-20 by the user-requested SigLIP-only revision.
> The separate patch encoder/byte-tokenizer design below is historical.

# MiniMind-A implementation plan

Goal: Standalone educational VLA based on MiniMind-V, inspired by SmolVLA.

Design: Keep the native MiniMind-V SigLIP → projector → causal language backbone.
Use its final hidden states as memory for a small noncausal Transformer action
expert, conditioned on state, noisy actions, time and learned chunk positions.
Train straight-line conditional flow matching; integrate noise to actions with Euler.
The context is computed once per sampled chunk. Native VLM weights can initialize
only the backbone; action expert is newly trained. A tiny patch encoder and UTF-8
byte tokenizer make an explicitly separate offline educational mode.

Alternatives: Direct MSE regression is simpler but omits flow matching. Full SmolVLA
layerwise expert/KV integration preserves more upstream architecture but is too
large for the requested teaching scope. Use a final-hidden-state cross-attention
expert. Single camera, one observation, fixed chunk length; no real robot runtime.
Inputs must use consistent normalized state/action units; padded actions ignored.

1. Write CPU behavioral tests for shape, loss, gradient flow, masking, conditioning,
   backbone initialization, sampling context reuse and save/load; run RED.
2. Copy native backbone sources, tokenizer and Apache license unchanged; implement
   model/model_vla.py with small offline encoder and actual SigLIP path.
3. Add dataset/vla_dataset.py for JSONL, real images, token padding and action masks;
   add synthetic image-reaching dataset generator, train_vla.py and eval_vla.py.
4. Run GREEN tests, actual CLI optimization and checkpoint inference, then repeat
   from an isolated project copy with no sibling dependency.
5. Document Korean walkthrough, equations, dimensions, commands, pretrained setup,
   simplifications and actual measured verification. Do not launch paid compute.

All writes remain in minimind-a; existing siblings and running jobs are untouched.
