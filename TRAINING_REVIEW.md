# Josi v3 — Build Plan

**Date:** 2026-02-09
**Decision:** Train SmolLM2 from scratch with clean data, proper eval, and reproducible pipeline.

---

## Why start over

The current model (`josi-smollm2-merged-v2-q4_k_m.gguf`) was trained with:
- No training script in version control — cannot reproduce
- Duplicated data (byte-identical copies inflating the dataset)
- I6 refusal examples at ~1-2% of training data (should be 15-25%)
- Manual chat formatting that doesn't match SmolLM2's template
- No post-quantization validation
- No structured output (JSON/schema) training
- Missing zones (Z7, Z8) and inconsistent readiness naming

The knowledge base is solid. The dataset generator has good bones. The model choice (SmolLM2) is right for the architecture. But the training discipline needs to match the constraints of a small model. **360M amplifies every data flaw.**

---

## What we build

### Phase 1 — Clean dataset

| Task | Detail |
|---|---|
| Remove all duplication | Delete every `for _ in range(N)` loop. Zero byte-identical examples. |
| Deduplicate by content hash | Canonicalize text, hash, drop collisions |
| Rebalance I6 refusals to 15-25% | Expand violation attempts, add diversity per persona |
| Add Z7 and Z8 | Full zone coverage matching knowledge cards |
| Align readiness naming | Use green/amber/red consistently everywhere |
| Add structured output examples | Train model to produce `LLMIntent` JSON when required |
| Increase template diversity | More question phrasings, more response variants per persona |
| Stratified train/val split | Split by task type, no near-duplicate leakage |

### Phase 2 — Training script (SmolLM2-specific)

| Task | Detail |
|---|---|
| New `finetune_smollm2.py` | Purpose-built for SmolLM2-360M (and optionally 1.7B) |
| Use `tokenizer.apply_chat_template()` | No manual formatting — correct tokens guaranteed |
| System message always included | I6 contract and persona in every example |
| LoRA config sized for 360M | Conservative rank (8-16), attention layers only |
| Lower learning rate | 1e-4 or lower — small model, clean data |
| Early stopping | Patience 2-3 on validation loss |
| Experiment tracking | Weights & Biases enabled from day one |
| max_seq_length enforced | Explicitly passed to SFTConfig |

### Phase 3 — Three-layer evaluation

**Layer 1: Schema validity**
- Does every response parse as valid `LLMIntent` JSON?
- Run on 100% of eval set, pass/fail, no exceptions

**Layer 2: Contract compliance**
- I6: refuses to modify plans (test with 30+ violation prompts)
- Tier gating: respects readiness constraints
- No invented numbers: never fabricates HR, pace, power values
- Persona consistency: tone matches requested style

**Layer 3: Quality**
- Warmth scoring (persona-aware thresholds)
- Response length (120-180 tokens max)
- No forbidden words (GATC internals)
- No jargon leakage
- N-gram repetition check (catch template parroting)
- Expand from 50 to 150+ prompts

**Run all three layers on:**
1. Base fp16 checkpoint
2. LoRA-merged checkpoint
3. Quantized GGUF (q4_k_m) — the artifact you ship

### Phase 4 — Export and compare

| Task | Detail |
|---|---|
| Export SmolLM2-360M q4_k_m | Default deployment target |
| Export SmolLM2-1.7B q4_k_m | Compare side-by-side in same eval harness |
| Post-quant eval on both | All 3 layers, same prompts, same thresholds |
| Decision: single model or hybrid | 360M default + 1.7B for "Explain more" if quality gap is real |

---

## What we keep

- **Knowledge cards** (18 .md files) — research-backed, well-structured, no changes needed
- **Generated modules** (context.py, tables.py) — good programmatic access to knowledge
- **Gold examples** (678 curated) — use as eval anchors and high-quality training seed
- **Persona definitions** (4 styles) — Balanced, Drill Sergeant, Dutch Directness, Science Nerd
- **GGUF export pipeline** — works, just add post-quant validation step

## What we delete or replace

- `finetune_mistral.py` — not the deployed model, causes confusion
- `finetune_ministral3b.py` — wrong model class, drops system message, no validation
- Repetition loops in `generate_dataset.py` — root cause of data quality issues
- Manual `[INST]...[/INST]` formatting — replaced by `apply_chat_template()`

---

## Order of work

```
1. Fix dataset generator     → clean data, no dupes, balanced I6, all zones
2. Write finetune_smollm2.py → SmolLM2-360M with proper LoRA + tracking
3. Build 3-layer eval suite  → schema, contract, quality
4. Train SmolLM2-360M        → clean run with tracking
5. Eval on fp16 + GGUF       → all 3 layers
6. Train SmolLM2-1.7B        → same data, same eval
7. Compare and decide        → ship one or both
```

---

*This replaces the previous review. All findings from that review are addressed in this plan.*
