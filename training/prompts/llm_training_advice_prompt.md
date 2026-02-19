# Prompt: Best approach to fine-tune a local LLM for AI sports coaching

I'm building **MiValta Josi**, an AI sports coaching assistant that runs **100% on-device** (Android via llama.cpp, no network calls). I need advice on the best training strategy given my hardware and constraints.

## What Josi does

Josi has two separate fine-tuned models:

### 1. Interpreter model (structured output)
- Reads an athlete's chat message + context (readiness, current session, sport, memory)
- Outputs a **single JSON object** (GATCRequest) that routes to the coaching engine
- 5 possible actions: `create_workout`, `replan`, `explain`, `answer_question`, `clarify`
- Each action has specific required fields and enum values (sport, replan_type, goal, fatigue_hint)
- Must extract time durations, infer sport from context, use athlete memory to skip clarification
- System prompt is ~2,800 chars, completions are short (30-50 tokens of JSON)
- **Training data**: 1,450 train + 149 validation examples

### 2. Explainer model (conversational coaching)
- Receives the coaching engine's decision and explains it to the athlete in warm, concise language
- Must stay under 100 words, use plain language, never invent training data
- Has personality guidelines (empathetic, direct, never recommends other apps)
- System prompt is ~2,400 chars, completions are 50-150 tokens of plain text
- **Training data**: 1,120 train + 128 validation examples

## Current hardware

- **Training GPU**: NVIDIA RTX 4000 SFF Ada Generation — **20 GB VRAM**
- **Deployment**: Android phones via llama.cpp (Q4_K_M quantization, targeting ~2-3 GB model size)
- Single GPU, no multi-GPU setup available

## Current approach and problems

**Base model**: Google Gemma 3n E2B-it
- 6B raw parameters, 2B effective (MatFormer selective activation / AltUp)
- Designed for on-device: small effective compute, good quality
- Vocabulary: 262,144 tokens (256K) — this is 8x larger than typical LLMs

**Training setup**:
- bf16 loading (~10.2 GB VRAM for model weights)
- LoRA fine-tuning: r=16, alpha=32, targeting attention + MLP projections (q/k/v/o_proj, gate/up/down_proj)
- Batch size=1, gradient accumulation=16 (effective batch=16)
- Completion-only loss (only train on model response, not system prompt)
- Learning rate: 2e-5, cosine schedule, 3 epochs, early stopping patience=3
- Gradient checkpointing enabled

**Problem**: CUDA OOM during training
- The 256K vocabulary means the cross_entropy logits tensor is massive: `seq_len x 262,144 x 4 bytes`
- With ~1,400 token sequences (mostly system prompt), the logits tensor alone is ~1.43 GB
- After model weights (10.2 GB) + activations + optimizer states, the 20 GB GPU runs out of memory
- **4-bit QLoRA is NOT possible** — Gemma 3n's AltUp layer uses `clamp_()` which crashes on quantized weights
- We trimmed the system prompt and reduced max_seq_length to 1024, but we're still tight on memory

**Deployment pipeline**: LoRA merge → full model → GGUF Q4_K_M export → llama.cpp on Android

## What I want advice on

Given these constraints (20GB training GPU, on-device deployment, ~1,450 training examples, structured JSON + conversational output), what is the best approach? Specifically:

1. **Is Gemma 3n E2B the right base model?** Or would a different model be better for this task and hardware? Consider:
   - Must run on-device via llama.cpp (GGUF export required)
   - 256K vocab creates training memory problems — would a smaller-vocab model be more practical?
   - Alternatives to consider: Gemma 3 1B, Phi-3.5-mini, Qwen2.5-1.5B/3B, SmolLM2-1.7B, Llama 3.2 1B/3B, or others
   - The interpreter task is essentially classification + field extraction — does it even need a 2B+ model?

2. **Training strategy for 20GB VRAM**:
   - Is bf16 + LoRA the best approach, or are there better alternatives?
   - Would `liger-kernel` (fused cross-entropy) help with the 256K vocab problem?
   - Gradient checkpointing settings — any optimizations beyond `use_reentrant=False`?
   - Is there a way to make QLoRA work with models that use in-place operations like `clamp_()`?
   - Optimal LoRA rank for structured JSON output with ~1,450 examples?

3. **Data efficiency**:
   - Is 1,450 examples enough for reliable structured JSON output?
   - Would data augmentation help, and if so, what kind?
   - Should the two models (interpreter + explainer) be trained separately or as a single multi-task model?
   - Is completion-only loss the right choice, or would full-sequence loss work better for such short completions?

4. **Deployment optimization**:
   - Q4_K_M quantization — is this the best quant level for a 2B-effective model on Android?
   - Context window: we cap at 1024 tokens — is this sufficient or should we plan for more?
   - Any llama.cpp-specific considerations for Gemma 3n or the model you'd recommend?

5. **Alternative architectures**:
   - Would a smaller model (1B or less) with higher LoRA rank outperform a larger model with lower rank?
   - Is LoRA the right PEFT method, or would something like DoRA, IA3, or full fine-tuning (with gradient offloading) be better?
   - Would distillation from a larger model (e.g., Gemma 3 27B → 2B) give better results than direct fine-tuning?

Please be specific and practical — I have a single 20GB GPU and need a solution that actually trains without OOM while producing good quality for on-device inference.
