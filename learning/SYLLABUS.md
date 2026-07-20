# SLM Foundation Syllabus

This is the executable prerequisite course for building SLM. It implements
Parts 0–V of the
[Humor Machine V2 master syllabus](https://app.notion.com/p/39d7d956b4c881178b90e844a55fcecb)
as 40 focused notebooks. The master page defines the complete Parts 0–XV
lifecycle; this document controls local lesson order and the boundary between
`learning/`, `brain/`, and `engine/`.

The notebooks use one global number because this repository's learning course
ends at the Part V readiness capstone. Parts VI–XV become implementation
milestones in `brain/` and `engine/`, rather than hundreds of flat prerequisite
notebooks. If those milestones later gain their own courses, they should use
part-specific directories inside their owning project.

## Completion standard for every notebook

Every completed lesson must include:

1. the problem, historical motivation, and SLM connection;
2. the smallest useful mathematical or systems model;
3. shape, dtype, device, layout, or resource annotations as appropriate;
4. a visible manual Python or NumPy implementation;
5. a PyTorch comparison when appropriate;
6. at least one controlled experiment;
7. numerical, shape, gradient, performance, or memory verification;
8. common failure modes and observable symptoms;
9. deliberate-practice exercises that change the implementation;
10. an explicit exit condition.

A notebook is ready only when it compiles and executes from a fresh kernel.
Reading or rerunning provided cells alone does not complete a lesson.

## Part 0 — Orientation

| # | Notebook | Notion coverage | Build and exit condition | Status |
| --- | --- | --- | --- | --- |
| 00 | [What an SLM actually is](notebooks/00_slm_map.ipynb) | 0.1–0.2 | Generate from a handcrafted probability table and distinguish architecture, weights, tokenizer, runtime, and application | Ready |

**Gate:** Trace user text through tokenization, next-token prediction, sampling,
decoding, and application behavior while naming the owner of every step.

## Part I — Computer systems for machine learning

| # | Notebook | Notion coverage | Build and exit condition | Status |
| --- | --- | --- | --- | --- |
| 01 | [CPU, GPU, and accelerator mental models](notebooks/01_cpu_gpu_accelerators.ipynb) | 1.1 | Compare control-heavy and data-parallel work; measure device work with synchronization and transfers | Ready |
| 02 | [Memory hierarchy and data movement](notebooks/02_memory_hierarchy_data_movement.ipynb) | 1.2 | Inspect strides, views, copies, locality, allocation, reuse, and arithmetic intensity | Ready |
| 03 | [Numbers inside computers](notebooks/03_numbers_precision.ipynb) | 1.3 | Compare FP64/32/16, BF16, mixed accumulation, stable formulas, and INT8/INT4-like quantization | Ready |
| 04 | [Parallelism, kernels, and synchronization](notebooks/04_parallelism_kernels_sync.ipynb) | 1.4 | Partition work into conceptual kernel invocations; identify races, barriers, launch cost, and fusion | Ready |
| 05 | [Matrix multiplication as a systems problem](notebooks/05_matmul_cache_tiling.ipynb) | 1.5 | Implement loop orders and boundary-safe tiling; verify square and rectangular GEMM | Ready |
| 06 | [Benchmarking and profiling](notebooks/06_benchmarking_profiling.ipynb) | 1.6 | Build a warm, repeated, synchronized percentile harness and reproducible report | Ready |
| 07 | [Model resource budgets](notebooks/07_model_resource_budget.ipynb) | 1.7 | Estimate weights, training state, activation floor, KV cache, LoRA state, and headroom | Ready |
| 08 | [Systems profiler capstone](notebooks/08_systems_profiler_capstone.ipynb) | Part I capstone | Produce a systems report and evidence-backed compute, memory, transfer, precision, allocation, and launch hypotheses | Ready |

**Gate:** Given a slow tensor program, propose separate measurements for
compute, movement, layout, allocation, precision, transfer, and launch overhead.

## Part II — Mathematical foundations

| # | Notebook | Notion coverage | Build and exit condition | Status |
| --- | --- | --- | --- | --- |
| 09 | [Tensors and numerical Python](notebooks/09_tensors.ipynb) | 2.1 | Predict scalar, vector, matrix, sequence, and batch shapes, dtypes, devices, and reductions | Ready |
| 10 | [Broadcasting and vectorization](notebooks/10_broadcasting_vectorization.ipynb) | 2.1, 2.3 | Replace loops with broadcast, matmul, batching, and Einstein notation while preserving semantics | Ready |
| 11 | [Vectors and geometric intuition](notebooks/11_vectors_geometry.ipynb) | 2.2 | Implement norms, dot products, cosine similarity, orthogonality, and projection | Ready |
| 12 | [Matrices and linear transformations](notebooks/12_matrices_linear_maps.ipynb) | 2.3–2.4 | Implement matrix multiplication and affine layers; reason about basis, rank, and conditioning | Ready |
| 13 | [Optional: eigenvalues, SVD, and low-rank structure](notebooks/13_optional_eigen_svd_low_rank.ipynb) | Enrichment for Parts X and XII | Reconstruct with truncated SVD and connect low-rank factors to LoRA and compression | Ready |
| 14 | [Derivatives and local sensitivity](notebooks/14_derivatives_sensitivity.ipynb) | 2.5 | Derive and numerically verify local sensitivity, gradients, and curvature | Ready |
| 15 | [Chain rule and computation graphs](notebooks/15_chain_rule_graphs.ipynb) | 2.6, 3.3 | Manually backpropagate through branches and verify vector–Jacobian products | Ready |
| 16 | [Probability and statistics](notebooks/16_probability_statistics.ipynb) | 2.7 | Compute expectation, variance, covariance, conditional probability, and sampling error | Ready |
| 17 | [Likelihood and information](notebooks/17_likelihood_information.ipynb) | 2.8 | Implement log-likelihood, entropy, cross-entropy, KL divergence, NLL, and perplexity | Ready |
| 18 | [Softmax, cross-entropy, and stability](notebooks/18_softmax_cross_entropy.ipynb) | 2.9–2.10 | Implement stable log-sum-exp and batched next-token cross-entropy; gradient-check `p-y` | Ready |
| 19 | [Optimization](notebooks/19_optimization.ipynb) | 2.11–2.12, 3.12 | Implement gradient descent, momentum, Adam, and AdamW and diagnose trajectories | Ready |
| 20 | [Mathematical foundations capstone](notebooks/20_mathematical_capstone.ipynb) | 2.15 | Build, derive, gradient-check, train, and evaluate multiclass softmax regression without autograd | Ready |

Initialization and normalization mathematics from Modules 2.13–2.14 are
implemented where their consequences can be measured across deep networks in
Lessons 26–27.

**Gate:** Derive stable cross-entropy and all gradients in the mathematical
capstone, make NumPy/finite-difference/PyTorch results agree, and evaluate on a
held-out split.

## Part III — Neural networks and automatic differentiation

| # | Notebook | Notion coverage | Build and exit condition | Status |
| --- | --- | --- | --- | --- |
| 21 | [Learning from data](notebooks/21_learning_from_data.ipynb) | Added baseline, 3.10 | Train linear/logistic baselines with leakage-safe train, validation, and test roles | Ready |
| 22 | [Neurons, activations, and MLPs](notebooks/22_neurons_activations_mlps.ipynb) | 3.1–3.2 | Implement a two-layer MLP, inspect activation derivatives, and solve XOR | Ready |
| 23 | [Scalar autograd](notebooks/23_scalar_autograd.ipynb) | 3.3–3.4 | Build a dynamic scalar reverse-mode autograd engine with topological traversal | Ready |
| 24 | [Tensor autograd and gradient checking](notebooks/24_tensor_autograd_gradcheck.ipynb) | 3.5, 3.11 | Derive affine tensor gradients, unbroadcast them, and compare analytical/numerical/autograd results | Ready |
| 25 | [Modules and the training loop](notebooks/25_modules_training_loop.ipynb) | 3.6–3.7 | Implement modules, minibatches, train/eval modes, clipping, metrics, and resume state | Ready |
| 26 | [Initialization and gradient flow](notebooks/26_initialization_gradient_flow.ipynb) | 2.13, 3.8 | Measure activation and gradient variance across bad, Xavier, and He initialization | Ready |
| 27 | [Residuals, normalization, and regularization](notebooks/27_residuals_normalization_regularization.ipynb) | 2.14, 3.9–3.10 | Implement LayerNorm, test residual flow, dropout modes, generalization tools, and curve diagnosis | Ready |

**Gate:** Train an MLP, verify gradients directly, intentionally overfit, and
distinguish numerical, optimization, data, and generalization failures.

## Part IV — Language, tokenization, and data representation

| # | Notebook | Notion coverage | Build and exit condition | Status |
| --- | --- | --- | --- | --- |
| 28 | [Text, Unicode, characters, words, and bytes](notebooks/28_text_unicode_bytes.ipynb) | 4.1–4.4 | Define normalization and prove strict multilingual/emoji byte round trips | Ready |
| 29 | [Learned n-gram language models](notebooks/29_ngram_language_model.ipynb) | Added bridge | Learn, smooth, evaluate, and sample a bigram model before neural context models | Ready |
| 30 | [BPE, unigram tokenization, and vocabulary design](notebooks/30_bpe_unigram_vocab.ipynb) | 4.5–4.7 | Train BPE merges, run unigram Viterbi segmentation, and evaluate vocabulary tradeoffs | Ready |
| 31 | [Special tokens, padding, controls, and prompts](notebooks/31_special_tokens_padding_prompts.ipynb) | 4.8–4.9, 4.13 | Define stable protocol IDs, prompt grammar, shifted targets, loss masks, and collision handling | Ready |
| 32 | [Context windows, attention masks, and packing](notebooks/32_context_windows_attention_masks_packing.ipynb) | 4.10–4.12 | Build causal/padding/document/loss masks, boundary-safe windows, packing, and storage metadata | Ready |

**Gate:** Turn raw documents into trustworthy training tensors and explain every
ID, normalization decision, shifted target, mask, document boundary, and stored
metadata field.

## Part V — Transformer architecture from first principles

| # | Notebook | Notion coverage | Build and exit condition | Status |
| --- | --- | --- | --- | --- |
| 33 | [Embeddings, residual stream, position, and RoPE](notebooks/33_embeddings_positions_rope.ipynb) | 5.1–5.3 | Implement lookup, learned/sinusoidal positions, RoPE Q/K rotation, norm preservation, and relative-position behavior | Ready |
| 34 | [Queries, keys, values, and scaled attention](notebooks/34_scaled_attention.ipynb) | 5.4–5.5 | Implement causal scaled dot-product attention and match PyTorch SDPA | Ready |
| 35 | [Causal multi-head attention](notebooks/35_causal_multihead_attention.ipynb) | 5.6–5.8 | Split/join heads, mask, retrieve, output-project, and verify residual-width updates | Ready |
| 36 | [Normalization, gated FFNs, and a decoder block](notebooks/36_norms_feedforward_decoder_block.ipynb) | 5.9–5.12 | Compare LayerNorm/RMSNorm and GELU/SwiGLU; assemble pre-norm block and prove causality | Ready |
| 37 | [Decoder stack and language-model head](notebooks/37_decoder_stack_lm_head.ipynb) | 5.13–5.14 | Stack blocks, produce `[B,T,V]` logits, tie embeddings, count parameters, and verify gradients | Ready |
| 38 | [Transformer complexity and failure analysis](notebooks/38_transformer_complexity_failures.ipynb) | 5.15–5.16 | Estimate compute/memory, design invariant tests, benchmark carefully, and limit attention interpretations | Ready |
| 39 | [Tiny decoder readiness capstone](notebooks/39_tiny_decoder_capstone.ipynb) | Part V capstone | Assemble a decoder, prove causality/gradient contracts, overfit one batch, and generate autoregressively | Ready |

**Gate:** Derive every tensor shape, identify every learned parameter, explain
training versus inference state, prove no future influence, and intentionally
overfit the tiny decoder before moving to `brain/`.

## One-to-one Notion traceability audit

The grouped lesson tables above show study order. This audit makes combined and
cross-part coverage explicit so a Notion module cannot disappear inside a broad
notebook title.

| Notion module | Primary local evidence |
| --- | --- |
| 0.1 | Lesson 00 — SLM component map and handcrafted autoregressive generator |
| 0.2 | The shared ten-part notebook contract and Lesson 00 exit ticket |
| 1.1 | Lesson 01 — CPU/GPU/accelerator mental models |
| 1.2 | Lesson 02 — hierarchy, locality, views, copies, allocation, and movement |
| 1.3 | Lesson 03 — floating point, accumulation, stability, and quantization |
| 1.4 | Lesson 04 — invocation indexing, races, barriers, launches, and fusion |
| 1.5 | Lesson 05 — loop order, tiling, rectangular GEMM, and verification |
| 1.6 | Lesson 06 — warm-up, synchronization, percentiles, throughput, and reports |
| 1.7 | Lesson 07 — model/training/cache/LoRA memory estimator |
| Part I capstone | Lesson 08 — systems profiler report |
| 2.1 | Lessons 09–10 — tensors, axes, broadcasting, reductions, and vectorization |
| 2.2 | Lesson 11 — vector geometry |
| 2.3 | Lessons 10 and 12 — matrix operations and implementations |
| 2.4 | Lesson 12 — linear/affine maps, rank, basis, and conditioning |
| 2.5 | Lesson 14 — derivatives and local sensitivity |
| 2.6 | Lesson 15 — chain rule and backpropagation mathematics |
| 2.7 | Lesson 16 — probability distributions and conditioning |
| 2.8 | Lesson 17 — logarithms, likelihood, and information |
| 2.9 | Lesson 18 — logits, softmax, and log-sum-exp stability |
| 2.10 | Lesson 18 — next-token cross-entropy and `p-y` gradient |
| 2.11 | Lesson 19 — landscapes and gradient descent |
| 2.12 | Lesson 19 — SGD, momentum, Adam, and AdamW |
| 2.13 | Lesson 26 — initialization, variance, and signal propagation |
| 2.14 | Lesson 27 — normalization mathematics and axis verification |
| 2.15 | Lesson 20 — mathematical capstone |
| 3.1 | Lessons 21–22 — linear baselines to multilayer perceptrons |
| 3.2 | Lesson 22 — activation functions and derivatives |
| 3.3 | Lessons 15 and 23 — computation graphs |
| 3.4 | Lesson 23 — scalar autograd engine |
| 3.5 | Lesson 24 — tensor-level automatic differentiation |
| 3.6 | Lesson 25 — parameters, modules, and composition |
| 3.7 | Lesson 25 — complete training and evaluation loop |
| 3.8 | Lesson 26 — deep-network gradient flow |
| 3.9 | Lesson 27 — normalization, residual paths, and stable depth |
| 3.10 | Lessons 21 and 27 — leakage, regularization, and generalization |
| 3.11 | Lesson 24 — finite-difference gradient checking and debugging |
| 3.12 | Lesson 19 — manual optimizer implementations |
| 4.1 | Lesson 28 — text, code points, graphemes, bytes, and IDs |
| 4.2 | Lesson 28 — character-level units and vocabulary tradeoffs |
| 4.3 | Lesson 28 — word-level units and unknown-word tradeoffs |
| 4.4 | Lesson 28 — UTF-8 byte encoding and strict round trips |
| 4.5 | Lesson 30 — BPE from scratch |
| 4.6 | Lesson 30 — unigram probabilistic segmentation |
| 4.7 | Lesson 30 — vocabulary design and tokenizer evaluation |
| 4.8 | Lesson 31 — special and control tokens |
| 4.9 | Lesson 31 — prompt formats, roles, and loss masking |
| 4.10 | Lesson 32 — context windows, shifted sequences, padding, and masks |
| 4.11 | Lesson 32 — packing and document isolation |
| 4.12 | Lesson 32 — token storage schema and provenance |
| 4.13 | Lesson 31 — protocol collisions and humor-control failure analysis |
| 5.1 | Lesson 33 — token embeddings and residual stream |
| 5.2 | Lesson 33 — learned and sinusoidal position representation |
| 5.3 | Lesson 33 — rotary position embeddings |
| 5.4 | Lesson 34 — queries, keys, and values |
| 5.5 | Lesson 34 — scaled dot-product attention |
| 5.6 | Lessons 34–35 — causal masking and future-influence tests |
| 5.7 | Lesson 35 — multi-head split, retrieval, and concatenation |
| 5.8 | Lesson 35 — output projection and residual-width update |
| 5.9 | Lesson 36 — LayerNorm/RMSNorm and pre/post-norm ordering |
| 5.10 | Lesson 36 — position-wise feed-forward network |
| 5.11 | Lesson 36 — SwiGLU and gated alternatives |
| 5.12 | Lesson 36 — assembled decoder block |
| 5.13 | Lesson 37 — stacked decoder-only transformer |
| 5.14 | Lesson 37 — LM head and true weight tying |
| 5.15 | Lesson 38 — attention inspection and interpretability limits |
| 5.16 | Lesson 38 — attention complexity, memory, and targeted failure tests |
| Part V capstone | Lesson 39 — verified tiny decoder overfit and generation |

Lesson 13 is intentionally optional enrichment. It does not pretend to replace
an early Notion module; it prepares for LoRA and compression in Parts X and XII.
Lesson 29 is an added conceptual bridge between handwritten probabilities and a
neural language model.

## Readiness gate for `brain/`

Begin MiniGPT in `brain/` when you can:

- form testable systems hypotheses and create a reproducible benchmark;
- estimate parameter, activation, optimizer, workspace, and KV-cache memory;
- predict every Transformer tensor shape and layout transformation;
- implement and gradient-check stable cross-entropy;
- train an MLP and distinguish optimization from generalization failure;
- implement a tokenizer and leakage-safe sequence builder;
- implement causal multi-head attention without `torch.nn.Transformer`;
- explain additive positions versus RoPE and LayerNorm/GELU versus RMSNorm/SwiGLU;
- intentionally overfit a tiny decoder on one batch;
- record enough state to reproduce and resume an experiment.

## After readiness — the Humor Machine lifecycle

The remaining V2 parts become project milestones rather than more prerequisites.

### Brain track

| Milestone | V2 coverage | Result |
| --- | --- | --- |
| B1 — Train MiniGPT | Part VI | Reproducible GPT-style baseline trained from random weights |
| B2 — Modernize the architecture | Part VII | Measured RoPE, RMSNorm, SwiGLU, GQA, efficient attention, and scaling decisions |
| B3 — Build the data pipeline | Part VIII | Licensed, filtered, deduplicated, tokenized, packed, and documented corpora |
| B4 — Reliable training systems | Part IX | Mixed precision, schedules, checkpoints, profiling, and exact resume |
| B5 — Specialize Humor Machine | Part X | SFT, LoRA/QLoRA, preference data, and justified DPO use |
| B6 — Evaluate and protect quality | Part XI | Humor, originality, controllability, safety, and regression suites |
| B7 — Build and operate the product | Parts XIV–XV | Product integration, monitoring, model cards, and iteration |

### Engine track

| Milestone | V2 coverage | Result |
| --- | --- | --- |
| E1 — Numerical performance | Deeper Part I implementation | Native kernels, tiled GEMM, synchronization, profiling, and reproducible benchmarks |
| E2 — Inference runtime | Part XII | Sampling, streaming, KV caching, memory reuse, formats, and quantization |
| E3 — Browser runtime | Part XIII | WGSL kernels, WebGPU execution, model loading, and browser profiling |

## Reading policy

The V2 academic map is a reference library, not a paper-counting exercise.

- Read at least one mapped primary source for every master module.
- Start with the abstract, problem, diagrams, key result, and limitation.
- Connect every adopted claim to a derivation, implementation, or measurement.
- Record the baseline, evidence, and one tradeoff for every technique.
- Revisit papers after the related notebook, when the equations have concrete meaning.
