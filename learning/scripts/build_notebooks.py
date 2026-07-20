"""Build the executable notebook curriculum from reviewable Python sources.

Run from ``learning/`` with ``uv run python scripts/build_notebooks.py``.
The generated notebooks are committed; this script exists so their structure is
consistent and large JSON documents do not have to be edited by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
WRITTEN: set[Path] = set()


def _source(text: str) -> list[str]:
    clean = dedent(text).strip("\n")
    return [f"{line}\n" for line in clean.splitlines()]


def md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def lesson(
    *,
    number: int,
    title: str,
    coverage: str,
    why: str,
    objectives: list[str],
    cells: list[dict[str, Any]],
    failures: list[str],
    exercises: list[str],
    exit_condition: str,
    next_lesson: str,
) -> dict[str, Any]:
    objective_list = "\n".join(f"- {item}" for item in objectives)
    failure_list = "\n".join(
        f"- **{name.split(':', 1)[0]}:**{name.split(':', 1)[1]}" for name in failures
    )
    exercise_list = "\n".join(f"{i}. {item}" for i, item in enumerate(exercises, 1))
    header = md(
        f"# {number:02d} — {title}\n\n"
        f"**Master syllabus coverage:** {coverage}\n\n"
        "## Why this matters\n\n"
        f"{why}\n\n"
        "## Learning objectives\n\n"
        f"{objective_list}\n\n"
        "Work through the notebook in order. Predict shapes and results before running each "
        "code cell. An assertion failure is part of the lesson: read it before changing code."
    )
    ending = [
        md(
            "## Failure modes to recognize\n\n"
            f"{failure_list}\n\n"
            "These are diagnostic patterns, not trivia. When a future model fails, reduce the "
            "problem to the smallest example in this notebook and test the relevant invariant."
        ),
        md(
            "## Deliberate practice\n\n"
            f"{exercise_list}\n\n"
            "Do not merely rerun the provided cells. Make a prediction, change one variable, "
            "and write down why the result did or did not match your prediction.\n\n"
            "## Exit ticket\n\n"
            f"**You are ready to continue when:** {exit_condition}\n\n"
            f"**Next:** {next_lesson}"
        ),
    ]
    notebook_cells = [header, *cells, *ending]
    for index, cell in enumerate(notebook_cells):
        cell["id"] = f"lesson-{number:02d}-{index:02d}"
    return {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "ml-learning (Python 3.12)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(filename: str, notebook: dict[str, Any]) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / filename
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    WRITTEN.add(path)
    print(f"wrote {path.relative_to(ROOT)} ({len(notebook['cells'])} cells)")


def build_00() -> None:
    cells = [
        md(
            r"""
            ## 1. The five layers people casually call “the model”

            | Layer | What it is | Example artifact |
            | --- | --- | --- |
            | Architecture | The computation graph and tensor shapes | decoder-only Transformer |
            | Parameters | Numbers learned by optimization | embedding and projection weights |
            | Tokenizer | Reversible mapping between text and token IDs | vocabulary + merge rules |
            | Runtime | Code that executes the graph and samples tokens | PyTorch now, C++ later |
            | Application | Prompts, controls, safety, UI, and monitoring | Humor Machine |

            Training repeatedly asks: “Given all earlier tokens, what probability should the
            next token receive?” Inference repeatedly samples from that distribution.

            $$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

            An SLM is “small” relative to frontier models, but it uses the same probabilistic
            interface: context in, next-token logits out.
            """
        ),
        code(
            """
            import numpy as np

            rng = np.random.default_rng(42)
            vocabulary = ["<BOS>", "why", "did", "byte", "cross", "road", "?", "<EOS>"]
            token_to_id = {token: index for index, token in enumerate(vocabulary)}
            id_to_token = dict(enumerate(vocabulary))

            print(token_to_id)
            assert len(token_to_id) == len(id_to_token)
            """
        ),
        md(
            r"""
            ## 2. A model without a neural network

            The table below is a tiny architecture whose “parameters” are handwritten
            probabilities. Row $i$ represents the current token; column $j$ represents the
            possible next token. Every row is a categorical distribution and must sum to one.

            Shape: `transition[current_token, next_token] = [V, V]`, where $V=8$.
            """
        ),
        code(
            """
            V = len(vocabulary)
            transition = np.zeros((V, V), dtype=np.float64)

            def set_row(current: str, choices: dict[str, float]) -> None:
                for following, probability in choices.items():
                    transition[token_to_id[current], token_to_id[following]] = probability

            set_row("<BOS>", {"why": 1.0})
            set_row("why", {"did": 1.0})
            set_row("did", {"byte": 1.0})
            set_row("byte", {"cross": 0.75, "?": 0.25})
            set_row("cross", {"road": 1.0})
            set_row("road", {"?": 1.0})
            set_row("?", {"<EOS>": 1.0})
            set_row("<EOS>", {"<EOS>": 1.0})

            np.testing.assert_allclose(transition.sum(axis=1), 1.0)
            print("parameter count:", transition.size)
            """
        ),
        md(
            r"""
            ## 3. Logits, temperature, sampling, and autoregression

            Neural networks normally emit unrestricted **logits**, not probabilities. Softmax
            turns logits into probabilities. Temperature changes distribution sharpness:

            $$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

            Greedy decoding picks the largest value. Sampling draws from the distribution and
            can produce different valid continuations. Autoregression appends one selected token
            and feeds the longer context back into the model.
            """
        ),
        code(
            """
            def choose_next(probabilities: np.ndarray, temperature: float = 1.0) -> int:
                if temperature <= 0:
                    return int(np.argmax(probabilities))
                safe = np.clip(probabilities, 1e-12, None)
                logits = np.log(safe) / temperature
                scaled = np.exp(logits - logits.max())
                scaled /= scaled.sum()
                return int(rng.choice(len(scaled), p=scaled))

            def generate(temperature: float = 0.0, max_new_tokens: int = 12) -> list[str]:
                tokens = ["<BOS>"]
                for _ in range(max_new_tokens):
                    row = transition[token_to_id[tokens[-1]]]
                    next_token = id_to_token[choose_next(row, temperature)]
                    tokens.append(next_token)
                    if next_token == "<EOS>":
                        break
                return tokens

            greedy = generate(temperature=0.0)
            sampled = [generate(temperature=0.8) for _ in range(4)]
            print("greedy:", greedy)
            print("sampled:", sampled)
            assert greedy[-1] == "<EOS>"
            """
        ),
        md(
            """
            ## 4. Training versus inference

            During **training**, the correct next token is known. We measure its negative log
            probability, backpropagate, and update parameters. During **inference**, labels are
            absent; the runtime repeatedly selects from predicted probabilities. The tokenizer
            is required in both directions.

            ```text
            text → tokenizer → IDs → architecture + weights → logits → sampler → next ID
                 ↑                                                        │
                 └────────────────────── decode IDs ←──────────────────────┘
            ```

            Later lessons replace the table with learned n-grams, an MLP, and finally a decoder
            Transformer. The interface remains next-token prediction.
            """
        ),
        code(
            """
            target_sequence = ["<BOS>", "why", "did", "byte", "cross", "road", "?", "<EOS>"]
            losses = []
            for current, target in zip(target_sequence, target_sequence[1:]):
                probability = transition[token_to_id[current], token_to_id[target]]
                losses.append(-np.log(probability))

            mean_nll = float(np.mean(losses))
            perplexity = float(np.exp(mean_nll))
            print(f"mean NLL={mean_nll:.4f}, perplexity={perplexity:.4f}")
            assert perplexity >= 1.0
            """
        ),
    ]
    write(
        "00_slm_map.ipynb",
        lesson(
            number=0,
            title="What an SLM Actually Is",
            coverage="V2 0.1–0.2",
            why="A clean component map prevents architecture, weights, tokenizer, runtime, and product behavior from becoming one vague idea. This distinction will guide the Python brain and future C++ engine boundary.",
            objectives=[
                "Name the five major layers of a language-model product.",
                "Explain training and inference as two uses of next-token probabilities.",
                "Implement greedy and stochastic autoregressive generation.",
                "Measure a sequence with negative log-likelihood and perplexity.",
            ],
            cells=cells,
            failures=[
                "Probabilities do not sum to one: sampling is no longer a valid categorical draw.",
                "No stop condition: autoregressive generation continues until an arbitrary length limit.",
                "Tokenizer/model mismatch: IDs refer to different tokens than the weights expect.",
                "Calling everything the model: ownership and debugging boundaries disappear.",
            ],
            exercises=[
                "Add an alternate continuation after `byte`, then compare greedy output with 20 sampled outputs.",
                "Change temperature to 0.25, 1.0, and 2.0; describe diversity without using the word random.",
                "Give one concrete file or data structure that will eventually represent each of the five layers.",
            ],
            exit_condition="you can trace text to token IDs, logits/probabilities, a sampled ID, and decoded text while naming which component owns every step.",
            next_lesson="01 — CPU, GPU, and accelerator mental models.",
        ),
    )


def build_system_cpu_gpu() -> None:
    cells = [
        md(
            """
            ## 1. Different hardware optimizes different work

            A CPU has relatively few sophisticated cores optimized for low-latency control flow,
            branch prediction, and varied tasks. A GPU has many simpler execution lanes optimized
            for high-throughput data-parallel arithmetic. An accelerator is useful only when the
            workload exposes enough parallel work to repay launch, scheduling, and transfer costs.

            Integrated GPUs such as Apple Silicon share a physical memory pool with the CPU, while
            discrete GPUs have separate device memory. Shared physical memory does not mean every
            framework operation or synchronization is free.
            """
        ),
        code(
            """
            import platform
            import time
            import numpy as np
            import torch

            def available_device() -> torch.device:
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")

            device = available_device()
            print("machine:", platform.machine())
            print("processor:", platform.processor() or "not reported")
            print("PyTorch threads:", torch.get_num_threads())
            print("selected accelerator:", device)
            """
        ),
        md(
            """
            ## 2. Python loop versus compiled vector kernel

            Comparing a Python loop with NumPy does not isolate CPU versus GPU—the NumPy operation
            is compiled, vectorized, and often multithreaded. It demonstrates why execution strategy
            matters even on identical hardware. Always name every difference in a benchmark.
            """
        ),
        code(
            """
            rng = np.random.default_rng(42)
            left = rng.normal(size=200_000).astype(np.float32)
            right = rng.normal(size=200_000).astype(np.float32)

            start = time.perf_counter()
            loop_result = np.empty_like(left)
            for index in range(left.size):
                loop_result[index] = left[index] + right[index]
            loop_time = time.perf_counter() - start

            start = time.perf_counter()
            vector_result = left + right
            vector_time = time.perf_counter() - start
            np.testing.assert_array_equal(loop_result, vector_result)
            print(f"Python loop={loop_time*1e3:.2f} ms, NumPy kernel={vector_time*1e3:.3f} ms")
            """
        ),
        md(
            """
            ## 3. Launch and synchronization change measured latency

            Accelerator operations are commonly asynchronous: the host queues work and continues.
            Synchronize immediately before starting and after ending a timed region. Time transfers
            separately when they are part of the user-visible operation.
            """
        ),
        code(
            """
            def synchronize(target: torch.device) -> None:
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                elif target.type == "mps":
                    torch.mps.synchronize()

            cpu_tensor = torch.from_numpy(left)
            start = time.perf_counter()
            device_tensor = cpu_tensor.to(device)
            synchronize(device)
            transfer_time = time.perf_counter() - start

            for _ in range(3):
                _ = device_tensor + 1
            synchronize(device)
            start = time.perf_counter()
            result = device_tensor + 1
            synchronize(device)
            kernel_time = time.perf_counter() - start
            print(f"CPU→{device} transfer={transfer_time*1e3:.3f} ms, one kernel={kernel_time*1e3:.3f} ms")
            assert result.shape == device_tensor.shape
            """
        ),
        md(
            """
            ## 4. Latency and throughput are different objectives

            Latency measures time for one request; throughput measures work per unit time. Batching
            often improves throughput by exposing more parallelism while increasing the time one
            request waits. Humor Machine will care about model-load time, time to first token, decode
            tokens per second, total response latency, and interface responsiveness.
            """
        ),
        code(
            """
            def timed_matmul(size: int, repeats: int = 5) -> tuple[float, float]:
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                for _ in range(2):
                    _ = a @ b
                synchronize(device)
                start = time.perf_counter()
                for _ in range(repeats):
                    output = a @ b
                synchronize(device)
                seconds = (time.perf_counter() - start) / repeats
                operations = 2 * size**3
                assert torch.isfinite(output).all()
                return seconds, operations / seconds

            for size in (32, 128, 512):
                seconds, operations_per_second = timed_matmul(size)
                print(f"n={size:3}: latency={seconds*1e3:8.3f} ms, rough throughput={operations_per_second/1e9:7.2f} GFLOP/s")
            """
        ),
    ]
    write(
        "01_cpu_gpu_accelerators.ipynb",
        lesson(
            number=1,
            title="CPU, GPU, and Accelerator Mental Models",
            coverage="V2 1.1",
            why="SLM workloads mix control-heavy host work with large parallel tensor operations. Knowing which device is suited to which work prevents the vague and often false rule that a GPU is always faster.",
            objectives=[
                "Contrast CPU latency-oriented execution with GPU throughput-oriented execution.",
                "Separate hardware differences from interpreter and kernel implementation differences.",
                "Measure asynchronous device work with explicit synchronization.",
                "Distinguish latency, throughput, transfer cost, and workload size.",
            ],
            cells=cells,
            failures=[
                "GPU-always-faster assumption: tiny or control-heavy workloads lose to overhead.",
                "Unsynchronized timing: the measurement captures queueing rather than completion.",
                "Transfer omitted: an accelerator benchmark excludes a cost the application must pay.",
                "Confounded comparison: Python versus compiled code is described as only CPU versus GPU.",
            ],
            exercises=[
                "Benchmark vector addition at four sizes on CPU and your available accelerator and identify the crossover, if any.",
                "Measure one matmul versus a batch of matmuls and compare latency with throughput.",
                "Draw the CPU/accelerator ownership path for loading data, training, checkpointing, and generation.",
            ],
            exit_condition="you can predict which workload properties favor a CPU or accelerator and design a timing that includes synchronization and transfers.",
            next_lesson="02 — Memory hierarchy and data movement.",
        ),
    )


def build_system_memory() -> None:
    cells = [
        md(
            """
            ## 1. Arithmetic waits for data

            The memory hierarchy moves from tiny/fast registers and caches to larger/slower main or
            device memory and finally storage. **Latency** is delay for one access; **bandwidth** is
            sustained bytes per second. Contiguous, predictable access improves cache lines,
            prefetching, and GPU memory coalescing. Many ML operations are limited by movement rather
            than arithmetic throughput.
            """
        ),
        code(
            """
            import time
            import numpy as np
            import torch

            array = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
            transposed = array.T
            copied = np.ascontiguousarray(transposed)
            for name, value in {"array": array, "transpose view": transposed, "contiguous copy": copied}.items():
                print(name, "shape=", value.shape, "strides(bytes)=", value.strides,
                      "contiguous=", value.flags.c_contiguous, "owns data=", value.flags.owndata)
            assert np.shares_memory(array, transposed)
            assert not np.shares_memory(array, copied)
            """
        ),
        md(
            """
            ## 2. Views change metadata; copies move bytes

            Reshape and transpose can often create a new interpretation of existing storage. A copy
            allocates storage and moves every value. A later kernel may require a contiguous copy,
            so a seemingly free view can defer cost rather than eliminate it. Inspect storage pointers,
            strides, and allocation at operation boundaries.
            """
        ),
        code(
            """
            base = torch.arange(3 * 4 * 5).reshape(3, 4, 5)
            view = base.transpose(0, 1)
            materialized = view.contiguous()
            print("base/view/materialized strides:", base.stride(), view.stride(), materialized.stride())
            assert base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
            assert view.untyped_storage().data_ptr() != materialized.untyped_storage().data_ptr()
            """
        ),
        md(
            """
            ## 3. Access order changes locality

            A C-contiguous matrix stores each row adjacently. Traversing the last axis in the inner
            loop visits neighboring values; column-first scalar traversal jumps by a row stride.
            Python overhead limits this demonstration, but the access-order principle persists in
            compiled kernels where it can determine cache and memory efficiency.
            """
        ),
        code(
            """
            matrix = np.random.default_rng(42).normal(size=(384, 384)).astype(np.float32)

            def scalar_sum_rows(value):
                total = 0.0
                for row in range(value.shape[0]):
                    for column in range(value.shape[1]):
                        total += value[row, column]
                return total

            def scalar_sum_columns(value):
                total = 0.0
                for column in range(value.shape[1]):
                    for row in range(value.shape[0]):
                        total += value[row, column]
                return total

            timings = {}
            for name, function in (("row-major", scalar_sum_rows), ("column-major", scalar_sum_columns)):
                start = time.perf_counter(); total = function(matrix); timings[name] = time.perf_counter() - start
                print(name, f"{timings[name]*1e3:.1f} ms", "sum=", total)
            row_sum = scalar_sum_rows(matrix)
            column_sum = scalar_sum_columns(matrix)
            print("rounding difference from accumulation order:", abs(float(row_sum - column_sum)))
            np.testing.assert_allclose(row_sum, column_sum, rtol=1e-5, atol=1e-4)
            """
        ),
        md(
            """
            ## 4. Allocation versus reuse

            Allocation requests storage and can trigger bookkeeping, synchronization, or garbage
            collection. Reusing an output buffer reduces allocation traffic and peak live memory.
            Reuse is valuable only when ownership and lifetimes remain clear—incorrect aliasing can
            silently overwrite needed activations.
            """
        ),
        code(
            """
            left = np.ones(500_000, dtype=np.float32)
            right = np.ones_like(left)
            output = np.empty_like(left)

            def measure(function, repeats=100):
                start = time.perf_counter()
                for _ in range(repeats): function()
                return time.perf_counter() - start

            allocate_time = measure(lambda: left + right)
            reuse_time = measure(lambda: np.add(left, right, out=output))
            print(f"allocate each time={allocate_time:.4f}s, reuse output={reuse_time:.4f}s")
            np.testing.assert_array_equal(output, left + right)
            """
        ),
        md(
            r"""
            ## 5. Arithmetic intensity gives a performance hypothesis

            Arithmetic intensity is operations divided by bytes moved. Low-intensity operations such
            as elementwise addition often become bandwidth-bound. Large matrix multiplication reuses
            tiles many times and can become compute-bound. The roofline idea is a hypothesis tool:
            attainable performance is limited by the lower of peak compute and bandwidth × intensity.
            """
        ),
        code(
            """
            # Approximate float32 vector add: one addition, two reads, one write.
            vector_add_intensity = 1 / (3 * 4)
            # Approximate n×n matmul: 2n³ operations, read A/B and write C once (optimistic).
            for n in (32, 256, 2048):
                matmul_intensity = 2 * n**3 / (3 * n**2 * 4)
                print(f"n={n:4}: vector-add={vector_add_intensity:.3f} op/byte, matmul≈{matmul_intensity:.1f} op/byte")
            """
        ),
    ]
    write(
        "02_memory_hierarchy_data_movement.ipynb",
        lesson(
            number=2,
            title="Memory Hierarchy and Data Movement",
            coverage="V2 1.2",
            why="Weights, activations, and KV caches spend much of their time moving through a hierarchy. Layout, copies, allocation, and reuse can dominate mathematically identical programs.",
            objectives=[
                "Explain latency, bandwidth, cache locality, and arithmetic intensity.",
                "Inspect contiguous and strided views in NumPy and PyTorch.",
                "Distinguish metadata-only views from materialized copies.",
                "Measure access-order and allocation-reuse effects without overstating one benchmark.",
            ],
            cells=cells,
            failures=[
                "Hidden materialization: a layout conversion increases latency and peak memory.",
                "Poor locality: workers wait for scattered memory despite low arithmetic count.",
                "Allocation churn: temporary buffers dominate a repeated small workload.",
                "Unsafe reuse: an output buffer overwrites a tensor still needed later.",
            ],
            exercises=[
                "Benchmark reduction on a contiguous tensor, its transpose, and a contiguous transpose copy.",
                "Measure copy time and effective bandwidth for three buffer sizes.",
                "Classify embedding lookup, LayerNorm, and large matmul as likely bandwidth- or compute-sensitive and defend each hypothesis.",
            ],
            exit_condition="you can explain a speed difference using layout, movement, reuse, and arithmetic intensity rather than operation count alone.",
            next_lesson="03 — Numbers inside computers.",
        ),
    )


def build_system_numbers() -> None:
    cells = [
        md(
            r"""
            ## 1. Finite bits approximate real numbers

            Floating point stores sign, exponent, and significand. More exponent bits increase range;
            more significand bits increase relative precision. Machine epsilon is the gap between 1
            and the next representable value. Overflow produces infinity; underflow produces tiny
            subnormal values or zero. Rounding occurs after nearly every operation.
            """
        ),
        code(
            """
            import math
            import numpy as np
            import torch

            print("0.1 + 0.2 =", format(0.1 + 0.2, ".18f"))
            for dtype in (np.float64, np.float32, np.float16):
                info = np.finfo(dtype)
                print(dtype.__name__, "bytes=", np.dtype(dtype).itemsize,
                      "epsilon=", info.eps, "tiny=", info.tiny, "max=", info.max)
            """
        ),
        md(
            """
            ## 2. Precision and range are separate

            FP16 and BF16 both use 16 bits. FP16 has more fraction precision but a much smaller
            exponent range; BF16 preserves float32-like range with coarser spacing. FP8 formats make
            different exponent/fraction tradeoffs. Integers require an explicit scale and often a
            zero point to approximate real values.
            """
        ),
        code(
            """
            values = torch.tensor([1e-8, 1.0001, 1e4, 1e8], dtype=torch.float32)
            for dtype in (torch.float16, torch.bfloat16):
                restored = values.to(dtype).float()
                print(dtype, restored.tolist())
            assert torch.isinf(values.to(torch.float16)[-1])
            assert torch.isfinite(values.to(torch.bfloat16)[-1])
            """
        ),
        md(
            """
            ## 3. Accumulation precision matters

            Adding a small value to a much larger running total can round the increment away. Order
            changes floating-point sums because arithmetic is not perfectly associative. Pairwise or
            compensated summation and higher-precision accumulators reduce error.
            """
        ),
        code(
            """
            increments = np.full(100_000, 1e-3, dtype=np.float16)
            sum_fp16 = np.sum(increments, dtype=np.float16)
            sum_fp32 = np.sum(increments, dtype=np.float32)
            expected = 100.0
            print("FP16 accumulation:", float(sum_fp16), "FP32 accumulation:", float(sum_fp32))
            assert abs(float(sum_fp32) - expected) < abs(float(sum_fp16) - expected)

            rng = np.random.default_rng(42)
            a = rng.normal(size=4096).astype(np.float16)
            b = rng.normal(size=4096).astype(np.float16)
            reference = np.dot(a.astype(np.float64), b.astype(np.float64))
            low = np.sum(a * b, dtype=np.float16)
            mixed = np.sum(a * b, dtype=np.float32)
            print("dot errors: FP16=", abs(float(low) - reference), "mixed=", abs(float(mixed) - reference))
            """
        ),
        md(
            """
            ## 4. Stable formulas are part of correctness

            Exponentiation overflows quickly. Softmax subtracts the maximum logit because adding a
            constant does not change the mathematical distribution. Similar rewrites stabilize
            log-likelihood, norms, variance, and normalization.
            """
        ),
        code(
            """
            logits = np.array([1000.0, 1001.0, 1002.0])
            with np.errstate(over="ignore", invalid="ignore"):
                naive = np.exp(logits) / np.exp(logits).sum()
            shifted = logits - logits.max()
            stable = np.exp(shifted) / np.exp(shifted).sum()
            print("naive:", naive, "stable:", stable)
            assert np.isnan(naive).any() and np.isfinite(stable).all()
            """
        ),
        md(
            r"""
            ## 5. Quantization maps floats to a finite integer grid

            Symmetric signed quantization selects scale $s=max|x|/q_{max}$, stores
            $q=\operatorname{clip}(\operatorname{round}(x/s))$, and reconstructs $\hat x=sq$.
            INT4 stores 16 levels conceptually; physical packing of two 4-bit values per byte is a
            separate systems concern.
            """
        ),
        code(
            """
            def symmetric_quantize(x: np.ndarray, bits: int):
                qmax = 2 ** (bits - 1) - 1
                scale = np.max(np.abs(x)) / qmax
                q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)
                return q, scale, q.astype(np.float32) * scale

            signal = np.linspace(-2.5, 2.5, 1000, dtype=np.float32)
            for bits in (8, 4):
                q, scale, reconstructed = symmetric_quantize(signal, bits)
                error = np.sqrt(np.mean((signal - reconstructed) ** 2))
                conceptual_bytes = signal.size * bits / 8
                print(f"INT{bits}: scale={scale:.5f}, RMSE={error:.5f}, conceptual bytes={conceptual_bytes:.0f}")
            """
        ),
    ]
    write(
        "03_numbers_precision.ipynb",
        lesson(
            number=3,
            title="Numbers Inside Computers",
            coverage="V2 1.3",
            why="Precision determines range, error, memory traffic, and hardware throughput. Mixed precision and quantization are numerical policies, not merely smaller file formats.",
            objectives=[
                "Explain sign, exponent, significand, range, precision, and machine epsilon.",
                "Contrast FP16 and BF16 behavior.",
                "Measure accumulation error and use stable formulas.",
                "Implement symmetric INT8- and INT4-like quantization and measure reconstruction error.",
            ],
            cells=cells,
            failures=[
                "Overflow/underflow: finite mathematical values become infinity or zero.",
                "Low-precision accumulation: small contributions vanish from a large sum.",
                "FP16/BF16 equivalence assumption: range and fraction tradeoffs are ignored.",
                "Uncalibrated quantization: clipping or coarse scales destroy important values.",
            ],
            exercises=[
                "Decode the conceptual roles of sign, exponent, and significand for one IEEE-754 value.",
                "Compare dot-product error across input and accumulation dtype combinations.",
                "Quantize a skewed distribution per tensor and per channel and compare error.",
            ],
            exit_condition="you can choose storage and accumulation precision while predicting range, rounding, memory, and reconstruction consequences.",
            next_lesson="04 — Parallelism, kernels, and synchronization.",
        ),
    )


def build_system_parallelism() -> None:
    cells = [
        md(
            """
            ## 1. A kernel applies one program across many indices

            Host code launches a kernel; many invocations compute different output elements. CPUs
            expose threads and SIMD vectors; GPUs organize SIMT lanes into warps/wavefronts and
            workgroups. Physical execution details vary, but every parallel algorithm must define
            work partitioning, memory ownership, and dependencies.
            """
        ),
        code(
            """
            import time
            import numpy as np
            import torch

            left = np.arange(17, dtype=np.float32)
            right = np.ones_like(left)

            def vector_add_invocation(index: int, a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
                if index < out.size:  # boundary guard for rounded-up workgroups
                    out[index] = a[index] + b[index]

            output = np.empty_like(left)
            workgroup_size = 8
            launched_invocations = ((left.size + workgroup_size - 1) // workgroup_size) * workgroup_size
            for index in range(launched_invocations):
                vector_add_invocation(index, left, right, output)
            np.testing.assert_array_equal(output, left + right)
            print("elements:", left.size, "launched invocations:", launched_invocations)
            """
        ),
        md(
            """
            ## 2. Parallel writes need ownership or coordination

            Independent elementwise outputs require no barrier because each worker owns one index.
            Reductions make many workers contribute to shared state and require staged partial sums,
            atomics, locks, or another coordination design. A race condition means the result depends
            on unpredictable execution interleaving.
            """
        ),
        code(
            """
            values = np.arange(1, 17, dtype=np.float32)
            groups = values.reshape(4, 4)
            partial_sums = groups.sum(axis=1)  # each conceptual workgroup owns one result
            total = partial_sums.sum()         # second synchronized stage
            assert total == values.sum()
            print("stage-one partial sums:", partial_sums, "stage-two total:", total)

            # Lost-update illustration: two workers both read 0, then both write 1.
            shared = 0
            worker_a_read = shared
            worker_b_read = shared
            shared = worker_a_read + 1
            shared = worker_b_read + 1
            print("racy result:", shared, "expected serialized result: 2")
            assert shared == 1
            """
        ),
        md(
            """
            ## 3. Launches and synchronization have fixed overhead

            Many tiny operations can spend more time in Python/framework dispatch and kernel launch
            than arithmetic. Fusion combines compatible operations, reduces intermediate memory
            traffic, and reduces launches. Fusion can increase compilation complexity or register use,
            so measure the real shapes.
            """
        ),
        code(
            """
            x = torch.randn(200_000)
            repeats = 100

            start = time.perf_counter()
            for _ in range(repeats):
                a = x + 1
                b = a * 2
                separate = torch.relu(b)
            separate_time = time.perf_counter() - start

            start = time.perf_counter()
            for _ in range(repeats):
                # One expression is not guaranteed one physical kernel, but exposes fusion opportunity.
                fused_candidate = torch.relu((x + 1) * 2)
            candidate_time = time.perf_counter() - start
            torch.testing.assert_close(separate, fused_candidate)
            print(f"separate expressions={separate_time:.4f}s, fusion candidate={candidate_time:.4f}s")
            """
        ),
        md(
            """
            ## 4. Dependencies bound parallelism

            A prefix sum has a sequential-looking recurrence; specialized scan algorithms reorganize
            it into parallel stages. Autoregressive decoding likewise cannot generate token `t+1`
            before selecting token `t`, although work *inside* one token step is highly parallel.
            Training knows all target tokens and computes sequence positions in parallel under a mask.
            """
        ),
        code(
            """
            sequence = torch.arange(1, 9)
            serial = []
            running = 0
            for value in sequence:
                running += int(value)
                serial.append(running)
            parallel_library = torch.cumsum(sequence, dim=0)
            assert serial == parallel_library.tolist()
            print("prefix sum:", serial)
            """
        ),
        md(
            """
            ## 5. Workgroup size is a hardware-informed tuning parameter

            Too little work wastes lanes and launch capacity; too much per group can exhaust registers
            or local/shared memory and reduce occupancy. Divergent branches can make lanes execute
            both paths while subsets remain idle. Portable code starts correct, then measures candidate
            configurations on target hardware.
            """
        ),
        code(
            """
            elements = 1_000
            for group_size in (32, 64, 128, 256):
                groups = (elements + group_size - 1) // group_size
                launched = groups * group_size
                utilization = elements / launched
                print(f"group={group_size:3}: groups={groups:2}, boundary utilization={utilization:.3f}")
            """
        ),
    ]
    write(
        "04_parallelism_kernels_sync.ipynb",
        lesson(
            number=4,
            title="Parallelism, Kernels, and Synchronization",
            coverage="V2 1.4",
            why="Fast tensor libraries divide operations among parallel workers. Correct indexing, ownership, synchronization, and launch granularity are prerequisites for understanding GPU and future WGSL engine work.",
            objectives=[
                "Map one output operation to conceptual kernel invocations and workgroups.",
                "Identify independent work, dependencies, reductions, barriers, and races.",
                "Explain launch overhead and operation-fusion opportunities.",
                "Relate workgroup size, divergence, and utilization without assuming one universal optimum.",
            ],
            cells=cells,
            failures=[
                "Out-of-bounds invocation: rounded-up launch sizes access invalid storage.",
                "Race condition: multiple workers update shared state without coordination.",
                "Missing synchronization: a later stage reads incomplete partial results.",
                "Tiny-kernel storm: fixed dispatch and memory costs dominate arithmetic.",
            ],
            exercises=[
                "Design a two-stage parallel maximum reduction including ownership and synchronization points.",
                "Count launches and temporary arrays in a small unfused activation pipeline.",
                "Explain which parts of autoregressive generation are sequential and which remain parallel.",
            ],
            exit_condition="you can partition an operation among workers, state who writes every location, and identify necessary synchronization and launch costs.",
            next_lesson="05 — Matrix multiplication as a systems problem.",
        ),
    )


def build_system_matmul() -> None:
    cells = [
        md(
            r"""
            ## 1. Same equation, different execution orders

            For $A=[M,K]$, $B=[K,N]$, output $C=[M,N]$ uses $2MKN$ approximate floating-point
            operations. Six loop permutations are mathematically equivalent but access rows and
            columns differently. Optimized GEMM implementations also vectorize, parallelize, pack,
            tile, and select hardware-specific instructions.
            """
        ),
        code(
            """
            import time
            import numpy as np
            import torch

            def matmul_ijk(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                m, k = a.shape; k2, n = b.shape
                assert k == k2
                out = np.zeros((m, n), dtype=np.float32)
                for i in range(m):
                    for j in range(n):
                        for p in range(k):
                            out[i, j] += a[i, p] * b[p, j]
                return out

            def matmul_ikj(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                m, k = a.shape; k2, n = b.shape
                assert k == k2
                out = np.zeros((m, n), dtype=np.float32)
                for i in range(m):
                    for p in range(k):
                        a_value = a[i, p]
                        for j in range(n):
                            out[i, j] += a_value * b[p, j]
                return out

            rng = np.random.default_rng(42)
            a = rng.normal(size=(48, 48)).astype(np.float32)
            b = rng.normal(size=(48, 48)).astype(np.float32)
            reference = a @ b
            for name, function in (("ijk", matmul_ijk), ("ikj", matmul_ikj)):
                start = time.perf_counter(); result = function(a, b); elapsed = time.perf_counter() - start
                np.testing.assert_allclose(result, reference, rtol=2e-5, atol=2e-5)
                print(name, f"{elapsed*1e3:.1f} ms")
            """
        ),
        md(
            """
            ## 2. Tiling increases reuse in faster memory

            A tiled algorithm operates on submatrices so pieces of A and B can be reused from cache
            or GPU shared/local memory before eviction. Tile size must fit the hierarchy and execution
            resources. The Python implementation exposes structure but cannot demonstrate the full
            benefit of a compiled vectorized kernel.
            """
        ),
        code(
            """
            def tiled_matmul(a: np.ndarray, b: np.ndarray, tile: int) -> np.ndarray:
                m, k = a.shape; k2, n = b.shape
                assert k == k2
                out = np.zeros((m, n), dtype=np.float32)
                for i0 in range(0, m, tile):
                    for p0 in range(0, k, tile):
                        for j0 in range(0, n, tile):
                            i1, p1, j1 = min(i0 + tile, m), min(p0 + tile, k), min(j0 + tile, n)
                            out[i0:i1, j0:j1] += a[i0:i1, p0:p1] @ b[p0:p1, j0:j1]
                return out

            rectangular_a = rng.normal(size=(37, 53)).astype(np.float32)
            rectangular_b = rng.normal(size=(53, 29)).astype(np.float32)
            for tile in (4, 8, 16, 32):
                result = tiled_matmul(rectangular_a, rectangular_b, tile)
                np.testing.assert_allclose(result, rectangular_a @ rectangular_b, rtol=2e-5, atol=2e-5)
            print("tiled rectangular multiplication passed boundary tiles")
            """
        ),
        md(
            """
            ## 3. Verification includes tolerance and shape families

            Floating-point accumulation order changes small rounding details. Test maximum absolute
            and relative error with precision-appropriate tolerances. Include non-square matrices,
            dimensions not divisible by tile size, zeros, large/small magnitudes, and incompatible
            shapes.
            """
        ),
        code(
            """
            result = tiled_matmul(rectangular_a, rectangular_b, tile=16)
            expected = rectangular_a @ rectangular_b
            absolute = np.max(np.abs(result - expected))
            relative = np.max(np.abs(result - expected) / np.maximum(np.abs(expected), 1e-6))
            print("output shape:", result.shape, "max abs error:", absolute, "max relative error:", relative)
            assert result.shape == (37, 29)
            """
        ),
        md(
            """
            ## 4. Transformer matmuls have varied shapes

            Embedding-width projections, QKᵀ, attention-value products, MLP expansion/contraction,
            and LM heads are rectangular and batched. Kernel choice depends on all dimensions, batch,
            dtype, layout, and device—not just total FLOPs.
            """
        ),
        code(
            """
            B, T, C, H, D, V = 4, 128, 256, 8, 32, 20_000
            workloads = {
                "QKV projection": ((B * T, C), (C, 3 * C)),
                "one-head QKᵀ": ((T, D), (D, T)),
                "MLP expansion": ((B * T, C), (C, 4 * C)),
                "LM head": ((B * T, C), (C, V)),
            }
            for name, (left_shape, right_shape) in workloads.items():
                operations = 2 * left_shape[0] * left_shape[1] * right_shape[1]
                print(f"{name:16}: {left_shape} @ {right_shape}, ≈{operations/1e6:.1f} MFLOP")
            """
        ),
        md(
            """
            ## 5. Optimized libraries are algorithm selectors

            NumPy BLAS and PyTorch dispatch to tuned implementations. A fair comparison separates
            setup, warms up, repeats, synchronizes accelerators, and checks output. This benchmark is
            a local observation, not evidence that one library wins on all hardware.
            """
        ),
        code(
            """
            large_a = torch.randn(256, 384)
            large_b = torch.randn(384, 192)
            for _ in range(3): _ = large_a @ large_b
            start = time.perf_counter()
            for _ in range(20): optimized = large_a @ large_b
            elapsed = (time.perf_counter() - start) / 20
            print(f"PyTorch {tuple(large_a.shape)} @ {tuple(large_b.shape)}: {elapsed*1e3:.3f} ms")
            assert optimized.shape == (256, 192)
            """
        ),
    ]
    write(
        "05_matmul_cache_tiling.ipynb",
        lesson(
            number=5,
            title="Matrix Multiplication as a Systems Problem",
            coverage="V2 1.5",
            why="Matmul dominates Transformer compute, yet the three-loop equation says little about real performance. Loop order, tiling, packing, vectorization, precision, and shape-specific dispatch determine hardware efficiency.",
            objectives=[
                "Implement and compare two loop orders for matrix multiplication.",
                "Implement boundary-safe tiled multiplication.",
                "Verify numerical error across square and rectangular shapes.",
                "Identify the major Transformer matmul shape families and their operation counts.",
            ],
            cells=cells,
            failures=[
                "Poor loop locality: mathematically identical code repeatedly reloads data.",
                "Broken boundary tile: non-divisible dimensions are skipped or accessed out of range.",
                "Exact-equality test: valid floating-point reorderings fail an inappropriate comparison.",
                "Square-only benchmark: conclusions do not transfer to Transformer projections.",
            ],
            exercises=[
                "Implement and time a third loop order while explaining its access pattern.",
                "Plot timing across tile sizes for two rectangular shapes without claiming a universal best tile.",
                "Estimate operations and optimistic bytes moved for QKV, MLP, and LM-head projections.",
            ],
            exit_condition="you can explain why loop order and tiling change performance without changing the matrix product and verify a tiled implementation.",
            next_lesson="06 — Measuring performance correctly.",
        ),
    )


def build_system_benchmarking() -> None:
    cells = [
        md(
            """
            ## 1. A benchmark is an experiment with a declared question

            State the operation, shapes, dtype, device, setup boundary, warm-up, synchronization,
            repeats, statistic, and correctness tolerance. Latency, throughput, peak memory, and
            energy answer different questions. Profiling explains where time goes; benchmarking
            compares an outcome under controlled conditions.
            """
        ),
        code(
            """
            import platform
            import time
            import numpy as np
            import torch

            def synchronize(device: torch.device) -> None:
                if device.type == "cuda": torch.cuda.synchronize(device)
                elif device.type == "mps": torch.mps.synchronize()

            def benchmark(function, *, warmup=5, repeats=30, device=torch.device("cpu")):
                for _ in range(warmup): function()
                synchronize(device)
                samples = []
                for _ in range(repeats):
                    synchronize(device)
                    start = time.perf_counter_ns()
                    function()
                    synchronize(device)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                values = np.asarray(samples)
                return {
                    "median_ms": float(np.median(values)),
                    "p90_ms": float(np.percentile(values, 90)),
                    "min_ms": float(values.min()),
                    "max_ms": float(values.max()),
                    "repeats": repeats,
                }
            """
        ),
        md(
            """
            ## 2. Separate setup from steady-state work

            First use can include imports, allocation, graph/kernel compilation, cache filling, and
            autotuning. Excluding setup is correct for steady-state questions but wrong for cold-start
            latency. Report both when the product experiences both.
            """
        ),
        code(
            """
            a = torch.randn(512, 512)
            b = torch.randn(512, 512)
            cold_start = benchmark(lambda: a @ b, warmup=0, repeats=1)
            steady_state = benchmark(lambda: a @ b, warmup=5, repeats=20)
            print("cold:", cold_start)
            print("steady:", steady_state)
            """
        ),
        md(
            """
            ## 3. Convert latency to useful throughput carefully

            Throughput needs a meaningful numerator: examples, tokens, bytes, or approximate FLOPs.
            FLOP/s can compare numerical kernels but may not predict application latency. Tokens/s
            depends on model, batch, prompt/decode phase, context length, sampler, and runtime.
            """
        ),
        code(
            """
            m, k, n = a.shape[0], a.shape[1], b.shape[1]
            operations = 2 * m * k * n
            seconds = steady_state["median_ms"] / 1000
            print("rough matmul throughput:", operations / seconds / 1e9, "GFLOP/s")

            vector = torch.randn(2_000_000)
            vector_stats = benchmark(lambda: vector + 1, repeats=30)
            optimistic_bytes = vector.numel() * vector.element_size() * 2  # read + write
            bandwidth = optimistic_bytes / (vector_stats["median_ms"] / 1000) / 1e9
            print("vector add:", vector_stats, "optimistic effective GB/s:", bandwidth)
            """
        ),
        md(
            """
            ## 4. Allocation and reuse need equivalent semantics

            Benchmark variants must compute the same result and expose equivalent synchronization.
            Reused buffers change ownership and may be unsafe in a real graph; include correctness and
            peak-live-memory checks, not only speed.
            """
        ),
        code(
            """
            left = np.ones(1_000_000, dtype=np.float32)
            right = np.ones_like(left)
            output = np.empty_like(left)
            allocate = benchmark(lambda: left + right, repeats=50)
            reuse = benchmark(lambda: np.add(left, right, out=output), repeats=50)
            np.testing.assert_array_equal(output, left + right)
            print("allocate:", allocate)
            print("reuse:", reuse)
            """
        ),
        md(
            """
            ## 5. A reproducible report records the environment

            At minimum record hardware, OS, Python/framework versions, device, dtype, shapes, command,
            code revision, warm-up, repeats, synchronization, statistics, and correctness criteria.
            Report distributions or percentiles rather than hiding variance behind one value.
            """
        ),
        code(
            """
            report = {
                "question": "steady-state CPU float32 matmul latency",
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "shape": [list(a.shape), list(b.shape)],
                "dtype": str(a.dtype),
                "warmup": 5,
                "metrics": steady_state,
                "correctness": "shape and finite output checked",
            }
            print(report)
            assert report["metrics"]["repeats"] == 20
            """
        ),
    ]
    write(
        "06_benchmarking_profiling.ipynb",
        lesson(
            number=6,
            title="Measuring Performance Correctly",
            coverage="V2 1.6",
            why="Performance claims guide architecture and runtime decisions only when measurements are synchronized, repeated, reproducible, equivalent, and connected to a product-relevant metric.",
            objectives=[
                "Build a warm-up, synchronization, repetition, and percentile benchmark harness.",
                "Separate cold-start from steady-state latency.",
                "Convert time into qualified operation or bandwidth throughput.",
                "Produce a reproducible benchmark report with correctness evidence.",
            ],
            cells=cells,
            failures=[
                "Single-run result: startup or OS noise determines the conclusion.",
                "Async timing: queued work completes outside the measured region.",
                "Different semantics: one variant omits transfers, allocation, or actual output work.",
                "Metric substitution: theoretical FLOPs are reported as user-perceived speed.",
            ],
            exercises=[
                "Extend the harness with p50, p95, mean, standard deviation, and raw-sample export.",
                "Benchmark cold and steady matrix multiplication for three shapes and explain variance.",
                "Write a Humor Machine inference benchmark specification covering load, prefill, first token, decode, and peak memory.",
            ],
            exit_condition="another engineer could reproduce your benchmark and understand exactly what was and was not included.",
            next_lesson="07 — Reading a model as a resource budget.",
        ),
    )


def build_system_budget() -> None:
    cells = [
        md(
            r"""
            ## 1. Parameters are only the first memory category

            Inference must store weights plus runtime buffers and KV cache. Training also stores
            gradients, optimizer state, saved activations, and temporary workspaces. For ordinary
            float32 AdamW, a rough persistent budget is $4P$ weight + $4P$ gradient + $8P$ moments
            = $16P$ bytes, before activations and overhead.
            """
        ),
        code(
            """
            from dataclasses import dataclass

            GIB = 2**30

            @dataclass(frozen=True)
            class ModelBudget:
                parameters: int
                bytes_per_weight: float
                layers: int
                context: int
                batch: int
                kv_heads: int
                head_dim: int
                bytes_per_cache_value: int = 2

                def weights_gib(self) -> float:
                    return self.parameters * self.bytes_per_weight / GIB

                def adamw_persistent_gib(self) -> float:
                    # Assumes fp32 weight, gradient, and two fp32 moment buffers.
                    return self.parameters * 16 / GIB

                def kv_cache_gib(self) -> float:
                    values = 2 * self.layers * self.batch * self.context * self.kv_heads * self.head_dim
                    return values * self.bytes_per_cache_value / GIB
            """
        ),
        md(
            """
            ## 2. Precision changes weight storage, not every other category automatically

            Quantized inference weights may use 8 or 4 bits plus scales/metadata. Activations and KV
            cache often use a different dtype. Training may keep master weights or optimizer moments
            in float32. State the precision of each category rather than saying “the model is FP16.”
            """
        ),
        code(
            """
            model_sizes = [10_000_000, 100_000_000, 500_000_000, 1_500_000_000]
            precisions = {"FP32": 4, "FP16/BF16": 2, "INT8": 1, "INT4 ideal": 0.5}
            for parameters in model_sizes:
                row = {name: parameters * bytes_per_weight / GIB for name, bytes_per_weight in precisions.items()}
                print(f"{parameters/1e6:6.0f}M params:", {key: f"{value:.2f} GiB" for key, value in row.items()})
            """
        ),
        md(
            r"""
            ## 3. KV cache grows with batch and context

            Approximate decoder cache elements:

            $$2LBT H_{kv}D$$

            for keys and values. Grouped-query attention reduces $H_{kv}$ without changing the number
            of query heads. Cache allocation policy, padding, fragmentation, and temporary attention
            work add real overhead beyond this clean formula.
            """
        ),
        code(
            """
            for context in (512, 2_048, 8_192, 32_768):
                mha = ModelBudget(125_000_000, 2, 12, context, 1, 12, 64)
                gqa = ModelBudget(125_000_000, 2, 12, context, 1, 4, 64)
                print(f"T={context:5}: MHA cache={mha.kv_cache_gib():.3f} GiB, GQA cache={gqa.kv_cache_gib():.3f} GiB")
            """
        ),
        md(
            """
            ## 4. Activation memory depends on graph and implementation

            Autograd saves values required by backward. A rough residual-sized estimate
            `B×T×C×L×bytes` is only a floor: attention intermediates, MLP expansion, normalization,
            dropout masks, temporary kernels, and allocator fragmentation add memory. Activation
            checkpointing trades recomputation for fewer saved tensors.
            """
        ),
        code(
            """
            def residual_floor_gib(batch, context, width, layers, bytes_per_value=2):
                return batch * context * width * layers * bytes_per_value / GIB

            for batch in (1, 8, 32):
                floor = residual_floor_gib(batch, 1024, 768, 12)
                print(f"batch={batch:2}: one residual-sized fp16 tensor/layer={floor:.3f} GiB")
            """
        ),
        md(
            """
            ## 5. LoRA changes trainable state, not the frozen base requirement

            A rank-r adapter for `[out,in]` trains `r(out+in)` values instead of `out×in`, reducing
            gradient and optimizer state for that update. The frozen base weights still occupy device
            memory unless quantized or offloaded. QLoRA combines quantized frozen weights with trainable
            low-rank adapters.
            """
        ),
        code(
            """
            def lora_parameters(input_width: int, output_width: int, rank: int) -> int:
                return rank * (input_width + output_width)

            full = 4096 * 4096
            for rank in (4, 8, 64):
                adapter = lora_parameters(4096, 4096, rank)
                print(f"rank={rank:2}: adapter={adapter:,}, full={full:,}, ratio={adapter/full:.4%}")
            """
        ),
        md(
            """
            ## 6. Feasibility includes headroom

            Never compare a theoretical total directly with all reported device memory. Reserve space
            for the OS/browser, runtime, allocator fragmentation, tokenizer/application, and workload
            spikes. A useful estimator reports assumptions and ranges, then validates them with peak
            memory measurements on the target runtime.
            """
        ),
        code(
            """
            example = ModelBudget(500_000_000, 2, 24, 4096, 1, 8, 128)
            estimate = {
                "weights_gib": example.weights_gib(),
                "kv_cache_gib": example.kv_cache_gib(),
                "known_total_gib": example.weights_gib() + example.kv_cache_gib(),
                "not_included": ["quantization metadata", "activations", "workspaces", "fragmentation", "runtime"],
            }
            print(estimate)
            assert estimate["known_total_gib"] > estimate["weights_gib"]
            """
        ),
    ]
    write(
        "07_model_resource_budget.ipynb",
        lesson(
            number=7,
            title="Reading a Model as a Resource Budget",
            coverage="V2 1.7",
            why="Before selecting a model or context length, an engineer should estimate whether weights, training state, activations, caches, workspaces, and runtime headroom plausibly fit the target device.",
            objectives=[
                "Estimate weight, gradient, optimizer, activation-floor, and KV-cache memory.",
                "Track precision separately for each memory category.",
                "Compare full training, inference, LoRA, and grouped-query cache costs.",
                "State estimator assumptions and missing overhead explicitly.",
            ],
            cells=cells,
            failures=[
                "Checkpoint-only budget: runtime or training state causes out-of-memory.",
                "One-dtype assumption: weights, moments, activations, and cache are miscounted.",
                "Context/batch omission: KV cache and activations grow beyond the estimate.",
                "No headroom: allocator or application overhead consumes the final available memory.",
            ],
            exercises=[
                "Build a table for 10M–1.5B parameter models across FP32, FP16, INT8, and INT4 storage.",
                "Add configurable SGD, AdamW, and 8-bit optimizer-state policies to the estimator.",
                "Choose a target Mac/browser memory budget and defend one plausible Humor Machine configuration.",
            ],
            exit_condition="you can estimate whether a model can load, train, or decode at a given context while naming every important excluded cost.",
            next_lesson="08 — Systems profiler capstone.",
        ),
    )


def build_system_capstone() -> None:
    cells = [
        md(
            """
            ## Capstone contract

            Build a compact systems profiler that records environment, shape, dtype, median latency,
            throughput, and a bottleneck hypothesis for representative numerical workloads. The goal
            is not a perfect profiler; it is a disciplined workflow that replaces “the GPU is slow”
            with testable hypotheses about compute, movement, precision, allocation, or launch cost.
            """
        ),
        code(
            """
            import platform
            import time
            import numpy as np
            import torch

            def synchronize(device):
                if device.type == "cuda": torch.cuda.synchronize(device)
                elif device.type == "mps": torch.mps.synchronize()

            device = (torch.device("cuda") if torch.cuda.is_available() else
                      torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))

            def median_ms(function, *, warmup=3, repeats=15, target=device):
                for _ in range(warmup): function()
                synchronize(target)
                samples = []
                for _ in range(repeats):
                    synchronize(target); start = time.perf_counter_ns(); function(); synchronize(target)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                return float(np.median(samples))

            environment = {
                "machine": platform.machine(), "OS": platform.platform(),
                "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            }
            print(environment)
            """
        ),
        md(
            """
            ## 1. Run controlled workloads

            Each report row names its shape and dtype, includes an approximate amount of work, and
            records the measurement method. The bottleneck label is a hypothesis to test with a real
            profiler—not a conclusion derived automatically from one ratio.
            """
        ),
        code(
            """
            rows = []

            vector = torch.randn(1_000_000, device=device)
            vector_ms = median_ms(lambda: vector + 1)
            rows.append({"workload": "vector add", "shape": list(vector.shape), "dtype": str(vector.dtype),
                         "median_ms": vector_ms, "hypothesis": "bandwidth or launch overhead"})

            left = torch.randn(512, 512, device=device)
            right = torch.randn(512, 512, device=device)
            matmul_ms = median_ms(lambda: left @ right)
            rows.append({"workload": "matmul", "shape": [list(left.shape), list(right.shape)],
                         "dtype": str(left.dtype), "median_ms": matmul_ms, "hypothesis": "compute/tiling"})

            for row in rows: print(row)
            """
        ),
        md(
            """
            ## 2. Compare allocation with reuse and contiguous with strided

            These demonstrations isolate memory-management and layout hypotheses. They use CPU NumPy
            so their array semantics are visible; repeat on the future C++/WebGPU runtime before using
            the results for engine design.
            """
        ),
        code(
            """
            rng = np.random.default_rng(42)
            a = rng.normal(size=(1024, 1024)).astype(np.float32)
            b = rng.normal(size=(1024, 1024)).astype(np.float32)
            out = np.empty_like(a)
            allocate_ms = median_ms(lambda: a + b, target=torch.device("cpu"))
            reuse_ms = median_ms(lambda: np.add(a, b, out=out), target=torch.device("cpu"))

            contiguous = a
            strided = a.T
            contiguous_ms = median_ms(lambda: contiguous.sum(axis=1), target=torch.device("cpu"))
            strided_ms = median_ms(lambda: strided.sum(axis=1), target=torch.device("cpu"))
            print({"allocate_ms": allocate_ms, "reuse_ms": reuse_ms,
                   "contiguous_reduce_ms": contiguous_ms, "strided_reduce_ms": strided_ms})
            np.testing.assert_array_equal(out, a + b)
            """
        ),
        md(
            """
            ## 3. Compare storage precision and reconstruction

            Performance without numerical quality is incomplete. Record bytes and error when testing
            lower precision or quantization, and identify which operations still accumulate in higher
            precision.
            """
        ),
        code(
            """
            signal = rng.normal(size=100_000).astype(np.float32)
            precision_rows = []
            for dtype in (np.float32, np.float16):
                stored = signal.astype(dtype)
                restored = stored.astype(np.float32)
                precision_rows.append({"dtype": str(dtype), "bytes": stored.nbytes,
                                       "RMSE": float(np.sqrt(np.mean((signal - restored) ** 2)))})
            print(precision_rows)
            assert precision_rows[1]["bytes"] == precision_rows[0]["bytes"] // 2
            """
        ),
        md(
            """
            ## 4. Produce the report and hypotheses

            A useful capstone report includes what was measured, what was held constant, correctness
            checks, and the next experiment. Avoid assigning “compute-bound” or “memory-bound” from
            intuition alone; a profiler, bandwidth counters, or scaling experiment should test it.
            """
        ),
        code(
            """
            report = {
                "environment": environment,
                "workloads": rows,
                "precision": precision_rows,
                "method": {"warmup": 3, "repeats": 15, "statistic": "median", "synchronized": True},
                "next_experiments": [
                    "sweep tensor size to find overhead/throughput transition",
                    "measure effective bandwidth for elementwise operations",
                    "profile matmul shapes used by the planned decoder",
                    "measure peak memory rather than estimating only live arrays",
                ],
            }
            print(report)
            assert len(report["workloads"]) >= 2 and report["method"]["synchronized"]
            """
        ),
    ]
    write(
        "08_systems_profiler_capstone.ipynb",
        lesson(
            number=8,
            title="Systems Profiler Capstone",
            coverage="V2 Part I capstone",
            why="The capstone integrates hardware, movement, precision, parallel work, matmul, benchmarking, and budgeting into a reproducible report and a set of falsifiable performance hypotheses.",
            objectives=[
                "Build a reusable synchronized benchmark kernel.",
                "Report vector, matmul, layout, allocation, and precision experiments.",
                "Separate measurements from bottleneck hypotheses.",
                "Specify the next profiler or scaling experiment needed to validate each hypothesis.",
            ],
            cells=cells,
            failures=[
                "Hardware folklore: a bottleneck label is assigned without a discriminating measurement.",
                "No correctness evidence: a fast variant may compute different values.",
                "Environment omitted: results cannot be reproduced or compared.",
                "Microbenchmark overreach: one synthetic workload dictates product architecture.",
            ],
            exercises=[
                "Add small and large CPU/accelerator comparisons including transfer time.",
                "Add effective bandwidth and approximate FLOP/s fields with clearly stated formulas.",
                "Write a one-page report classifying each workload as an evidence-backed or still-untested hypothesis.",
            ],
            exit_condition="you can inspect a slow tensor program and propose separate tests for compute, memory, transfer, allocation, precision, and launch overhead.",
            next_lesson="09 — Tensors and numerical Python.",
        ),
    )


def build_math_tensors() -> None:
    cells = [
        md(
            r"""
            ## 1. Tensor vocabulary

            A tensor is an n-dimensional rectangular array plus metadata. The core metadata is:

            - **shape:** length of every axis;
            - **rank / ndim:** number of axes (not matrix rank);
            - **dtype:** numeric representation and precision;
            - **device:** where storage and operations live;
            - **stride:** how indices map to memory offsets.

            SLM convention used throughout the course:

            - $B$: batch size, $T$: sequence length, $C$: model width;
            - $V$: vocabulary size, $H$: attention heads, $D=C/H$: head width.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            np.random.seed(42)
            torch.manual_seed(42)

            scalar = torch.tensor(7.0)                         # []
            vector = torch.tensor([1.0, 2.0, 3.0])             # [C]
            matrix = torch.arange(12).reshape(3, 4)             # [T, C]
            sequence_batch = torch.randn(2, 3, 4)              # [B, T, C]

            for name, value in {"scalar": scalar, "vector": vector, "matrix": matrix,
                                "sequence_batch": sequence_batch}.items():
                print(f"{name:15} shape={tuple(value.shape)!s:10} ndim={value.ndim} dtype={value.dtype}")

            assert scalar.shape == torch.Size([])
            assert sequence_batch.shape == (2, 3, 4)
            """
        ),
        md(
            """
            ## 2. Axes carry meaning

            Shape equality is not semantic equality. `[B, T, C]` and `[T, B, C]` contain the
            same number of values but mean different things. Name shapes in comments and assert
            boundaries. In TypeScript terms, a tensor shape behaves like crucial runtime schema
            information that the basic tensor type does not encode.
            """
        ),
        code(
            """
            B, T, C = sequence_batch.shape
            per_token_mean = sequence_batch.mean(dim=-1)       # [B, T]
            per_sequence_mean = sequence_batch.mean(dim=1)     # [B, C]
            transposed = sequence_batch.transpose(0, 1)        # [T, B, C]

            print("per token:", per_token_mean.shape)
            print("per sequence:", per_sequence_mean.shape)
            print("transposed:", transposed.shape)
            assert per_token_mean.shape == (B, T)
            assert per_sequence_mean.shape == (B, C)
            """
        ),
        md(
            r"""
            ## 3. Dtypes and numerical meaning

            Integer tensors usually hold token IDs or labels. Floating tensors hold parameters,
            activations, gradients, and probabilities. Boolean tensors hold masks. The dtype
            changes range, precision, memory, and which operations are legal.

            A float with $n$ elements uses approximately `n × element_size` bytes. Casting can
            be lossy; integer division, overflow, and low-precision accumulation deserve explicit
            attention.
            """
        ),
        code(
            """
            token_ids = torch.tensor([[1, 4, 2], [3, 0, 0]], dtype=torch.long)  # [B, T]
            padding_mask = token_ids != 0                                      # [B, T]
            activations = torch.randn(2, 3, 8, dtype=torch.float32)             # [B, T, C]

            for tensor in (token_ids, padding_mask, activations, activations.half()):
                mib = tensor.numel() * tensor.element_size() / 2**20
                print(tensor.dtype, tensor.shape, f"{mib:.6f} MiB")

            rounded = torch.tensor([16_777_216.0], dtype=torch.float32)
            print("float32 loses this +1:", rounded + 1 == rounded)
            assert token_ids.dtype == torch.int64 and padding_mask.dtype == torch.bool
            """
        ),
        md(
            """
            ## 4. NumPy ↔ PyTorch and device placement

            NumPy is ideal for visible numerical algorithms. PyTorch adds devices, automatic
            differentiation, neural-network layers, and optimized kernels. On CPU, conversion can
            share memory; use `.copy()` or `.clone()` when aliasing would be surprising.
            """
        ),
        code(
            """
            array = np.arange(6, dtype=np.float32).reshape(2, 3)
            tensor = torch.from_numpy(array)  # shares CPU memory
            array[0, 0] = 99.0
            assert tensor[0, 0].item() == 99.0

            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            moved = tensor.to(device)
            print("selected device:", device, "tensor device:", moved.device)
            print("NumPy shares CPU storage:", tensor.data_ptr() == torch.from_numpy(array).data_ptr())
            """
        ),
        md(
            """
            ## 5. Indexing, reshaping, and reductions

            Indexing removes or selects axes. Reshaping reorganizes axis boundaries while keeping
            the number of elements constant. Reductions remove axes unless `keepdim=True`. Keeping
            a reduced dimension often makes later broadcasting easier to reason about.
            """
        ),
        code(
            """
            x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)  # [B=2, T=3, C=4]
            first_sequence = x[0]                         # [T, C]
            last_feature = x[:, :, -1]                    # [B, T]
            flattened_tokens = x.reshape(-1, 4)           # [B*T, C]
            centered = x.float() - x.float().mean(dim=-1, keepdim=True)

            assert first_sequence.shape == (3, 4)
            assert flattened_tokens.shape == (6, 4)
            torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0)
            print("all shape and centering invariants passed")
            """
        ),
    ]
    write(
        "09_tensors.ipynb",
        lesson(
            number=9,
            title="Tensors and Numerical Python",
            coverage="V2 2.1",
            why="Nearly every SLM bug eventually appears as a wrong shape, dtype, device, or interpretation of an axis. Tensor literacy is the runtime type system of ML.",
            objectives=[
                "Read and write scalar, vector, matrix, sequence, and batch shapes.",
                "Choose appropriate integer, floating-point, and boolean dtypes.",
                "Use indexing, reshaping, reductions, and device movement deliberately.",
                "Recognize memory sharing between NumPy and CPU PyTorch tensors.",
            ],
            cells=cells,
            failures=[
                "Axis confusion: a valid operation produces semantically wrong results.",
                "Dtype mismatch: token IDs become floats or a floating operation truncates to integers.",
                "Device mismatch: operands on CPU and MPS/CUDA cannot participate in one operation.",
                "Unexpected aliasing: modifying a NumPy array silently changes a shared tensor.",
            ],
            exercises=[
                "Create logits with shape `[B=2, T=5, V=11]` and extract the final time step as `[B, V]`.",
                "Compute the memory in MiB for a hypothetical `[8, 1024, 768]` float16 activation.",
                "Intentionally swap batch and time, then write an assertion that catches the mistake.",
            ],
            exit_condition="you can annotate every tensor in a small sequence computation and predict the result of indexing or reducing each axis.",
            next_lesson="10 — Broadcasting and vectorization.",
        ),
    )


def build_math_broadcasting() -> None:
    cells = [
        md(
            r"""
            ## 1. Broadcasting is alignment from the right

            Compare dimensions from right to left. Two dimensions are compatible when they are
            equal or either one is 1; missing leading dimensions act like 1. Broadcasting creates
            a logical expanded view rather than copying values in most cases.

            Example: `[B,T,C] + [C] → [B,T,C]`. A feature bias is reused for every token in every
            batch item. But `[B,T,C] + [T]` fails because `T` is compared with `C`.
            """
        ),
        code(
            """
            import time
            import numpy as np
            import torch

            torch.manual_seed(42)
            B, T, C = 2, 3, 4
            x = torch.arange(B * T * C, dtype=torch.float32).reshape(B, T, C)
            feature_bias = torch.tensor([0.1, 0.2, 0.3, 0.4])  # [C]
            token_bias = torch.tensor([1.0, 2.0, 3.0]).view(1, T, 1)  # [1,T,1]

            y = x + feature_bias + token_bias
            assert y.shape == (B, T, C)
            print(y[0])
            """
        ),
        md(
            """
            ## 2. Replace loops, preserve meaning

            Vectorization expresses a whole operation in array algebra so optimized native kernels
            can process it. First write the obvious loop, establish a correct reference, then
            vectorize and compare. Faster wrong code is still wrong.
            """
        ),
        code(
            """
            rng = np.random.default_rng(42)
            inputs = rng.normal(size=(32, 64)).astype(np.float32)   # [B,Cin]
            weights = rng.normal(size=(128, 64)).astype(np.float32) # [Cout,Cin]
            bias = rng.normal(size=(128,)).astype(np.float32)       # [Cout]

            loop_output = np.empty((32, 128), dtype=np.float32)
            for b in range(inputs.shape[0]):
                for o in range(weights.shape[0]):
                    total = bias[o]
                    for i in range(weights.shape[1]):
                        total += inputs[b, i] * weights[o, i]
                    loop_output[b, o] = total

            vectorized_output = inputs @ weights.T + bias
            np.testing.assert_allclose(loop_output, vectorized_output, rtol=2e-5, atol=2e-5)
            print("loop and vectorized affine transforms agree")
            """
        ),
        md(
            """
            ## 3. Views, copies, strides, and contiguity

            A tensor is storage plus shape, stride, and offset. `transpose` usually changes strides
            without moving data. `view` requires a compatible contiguous layout; `reshape` may
            return a view or allocate a copy. `contiguous()` explicitly materializes standard layout.
            """
        ),
        code(
            """
            base = torch.arange(12).reshape(3, 4)
            transposed = base.T
            contiguous = transposed.contiguous()

            print("base stride:", base.stride(), "contiguous:", base.is_contiguous())
            print("transpose stride:", transposed.stride(), "contiguous:", transposed.is_contiguous())
            print("copied stride:", contiguous.stride(), "contiguous:", contiguous.is_contiguous())
            assert base.untyped_storage().data_ptr() == transposed.untyped_storage().data_ptr()
            assert contiguous.untyped_storage().data_ptr() != transposed.untyped_storage().data_ptr()
            """
        ),
        md(
            r"""
            ## 4. Batched matrix multiplication and Einstein notation

            Attention is dominated by batched matrix products. `einsum` names axis relationships:

            $$Y_{bto}=\sum_c X_{btc}W_{oc}$$

            The string `btc,oc->bto` is both executable and a compact shape derivation.
            """
        ),
        code(
            """
            x = torch.randn(2, 5, 4)     # [B,T,C]
            w = torch.randn(7, 4)        # [O,C]
            via_matmul = x @ w.T         # [B,T,O]
            via_einsum = torch.einsum("btc,oc->bto", x, w)
            torch.testing.assert_close(via_matmul, via_einsum)

            expanded = feature_bias.expand(B, T, C)
            print("expanded stride:", expanded.stride(), "storage elements:", expanded.untyped_storage().nbytes() // 4)
            assert expanded.stride()[0] == 0 and expanded.stride()[1] == 0
            """
        ),
        md(
            """
            ## 5. A careful microbenchmark

            Timings are noisy and device operations can be asynchronous. Warm up first, repeat,
            synchronize accelerators, and compare equivalent work. This tiny CPU timing is only a
            demonstration—not a publishable benchmark.
            """
        ),
        code(
            """
            a = np.random.default_rng(0).normal(size=(200_000,)).astype(np.float32)
            b = np.random.default_rng(1).normal(size=(200_000,)).astype(np.float32)

            start = time.perf_counter()
            loop_sum = sum(float(left * right) for left, right in zip(a, b))
            loop_seconds = time.perf_counter() - start
            start = time.perf_counter()
            vector_sum = float(a @ b)
            vector_seconds = time.perf_counter() - start

            print(f"loop={loop_seconds:.4f}s vectorized={vector_seconds:.6f}s")
            np.testing.assert_allclose(loop_sum, vector_sum, rtol=2e-5)
            """
        ),
    ]
    write(
        "10_broadcasting_vectorization.ipynb",
        lesson(
            number=10,
            title="Broadcasting and Vectorization",
            coverage="V2 1.2, 2.1, 2.3",
            why="Vectorized tensor algebra is how mathematical intent reaches fast kernels. Broadcasting and layout knowledge prevent silent shape errors and unnecessary memory copies.",
            objectives=[
                "Derive broadcasted output shapes from right to left.",
                "Translate nested loops into matrix, batched, and Einstein notation.",
                "Explain views, copies, strides, expansion, and contiguity.",
                "Design a modest benchmark without drawing conclusions from one noisy timing.",
            ],
            cells=cells,
            failures=[
                "Accidental broadcasting: compatible shapes produce the wrong semantic pairing.",
                "Hidden copy: a reshape or contiguous conversion increases peak memory.",
                "Non-contiguous view: `view` fails or a kernel takes a slower layout path.",
                "Bad benchmark: startup, asynchronous execution, or unequal work dominates timing.",
            ],
            exercises=[
                "Derive and test `[B,T,1] * [C]`, `[B,1,C] + [T,1]`, and one intentionally invalid pair.",
                "Implement batched cosine similarity first with loops and then without Python loops.",
                "Transpose a four-axis tensor, inspect its stride, and predict whether `view(-1)` succeeds.",
            ],
            exit_condition="you can vectorize a loop, prove numerical equivalence, and explain the output shape and storage behavior.",
            next_lesson="11 — Vectors and geometric intuition.",
        ),
    )


def build_math_vectors() -> None:
    cells = [
        md(
            r"""
            ## 1. Vectors are coordinates and directions

            A vector $x\in\mathbb{R}^d$ is an ordered list relative to a basis. In an embedding
            space, coordinates are learned; individual dimensions rarely have a predefined human
            meaning. Geometry still gives useful relationships.

            $$\|x\|_2=\sqrt{\sum_i x_i^2},\qquad x\cdot y=\sum_i x_i y_i$$

            The dot product combines magnitude and alignment. Cosine similarity divides out the
            magnitudes and lies in $[-1,1]$ for nonzero real vectors.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            def dot(x: np.ndarray, y: np.ndarray) -> float:
                assert x.shape == y.shape and x.ndim == 1
                return float(sum(float(a * b) for a, b in zip(x, y)))

            def l2_norm(x: np.ndarray) -> float:
                return float(np.sqrt(dot(x, x)))

            def cosine(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
                return dot(x, y) / max(l2_norm(x) * l2_norm(y), eps)

            x = np.array([3.0, 4.0])
            y = np.array([4.0, -3.0])
            print("norm:", l2_norm(x), "dot:", dot(x, y), "cosine:", cosine(x, y))
            assert l2_norm(x) == 5.0 and dot(x, y) == 0.0
            """
        ),
        md(
            r"""
            ## 2. Projection decomposes a vector

            Projection of $x$ onto nonzero $u$:

            $$\operatorname{proj}_u(x)=\frac{x\cdot u}{u\cdot u}u$$

            Then $x=\operatorname{proj}_u(x)+r$, and residual $r$ is orthogonal to $u$. This pattern
            reappears in least squares, basis changes, and analysis of learned representations.
            """
        ),
        code(
            """
            def project(x: np.ndarray, onto: np.ndarray) -> np.ndarray:
                denominator = dot(onto, onto)
                if denominator == 0:
                    raise ValueError("cannot project onto the zero vector")
                return (dot(x, onto) / denominator) * onto

            x = np.array([3.0, 4.0])
            axis = np.array([1.0, 1.0])
            parallel = project(x, axis)
            residual = x - parallel
            print("parallel:", parallel, "residual:", residual)
            np.testing.assert_allclose(dot(residual, axis), 0.0, atol=1e-12)
            """
        ),
        md(
            r"""
            ## 3. Batch geometry

            For embeddings `E=[N,C]`, all pairwise dot products are `E @ E.T = [N,N]`. Normalize
            each row first to obtain a cosine-similarity matrix. The diagonal should be one unless
            a row is zero.
            """
        ),
        code(
            """
            torch.manual_seed(42)
            embeddings = torch.randn(5, 8)  # [tokens,C]
            unit = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            similarities = unit @ unit.T    # [tokens,tokens]

            print(similarities.round(decimals=2))
            torch.testing.assert_close(similarities.diag(), torch.ones(5))
            torch.testing.assert_close(similarities, similarities.T)
            """
        ),
        md(
            r"""
            ## 4. Distance, similarity, and scaling

            Euclidean distance is sensitive to scale; cosine similarity is scale-invariant. Neither
            is universally correct. Attention uses scaled dot products because magnitudes carry
            information but growing dimension otherwise increases logit variance.
            """
        ),
        code(
            """
            anchor = np.array([1.0, 1.0])
            candidates = {
                "same direction, far": np.array([10.0, 10.0]),
                "nearby, different angle": np.array([1.0, 0.5]),
                "opposite": np.array([-1.0, -1.0]),
            }
            for name, candidate in candidates.items():
                distance = np.linalg.norm(anchor - candidate)
                print(f"{name:24} distance={distance:6.2f} cosine={cosine(anchor, candidate):6.2f}")
            """
        ),
    ]
    write(
        "11_vectors_geometry.ipynb",
        lesson(
            number=11,
            title="Vectors and Geometric Intuition",
            coverage="V2 2.2",
            why="Tokens, hidden states, gradients, and parameter updates are vectors. Norm, angle, and projection give a language for their size, alignment, and decomposition.",
            objectives=[
                "Implement dot products and norms from scalar operations.",
                "Distinguish Euclidean distance from cosine similarity.",
                "Project a vector and verify the orthogonal residual.",
                "Compute pairwise similarity for a batch of embeddings.",
            ],
            cells=cells,
            failures=[
                "Zero-vector normalization: division produces NaN without an epsilon or explicit policy.",
                "Cosine as distance: direction-only similarity ignores useful magnitude information.",
                "Wrong reduction axis: similarities aggregate tokens instead of features.",
                "Embedding literalism: one learned coordinate is treated as a stable human concept.",
            ],
            exercises=[
                "Implement L1 and infinity norms and compare their unit-distance rankings.",
                "Use Gram–Schmidt to turn two independent vectors into an orthonormal pair.",
                "Find the closest pair among ten random embeddings using cosine similarity without loops over pairs.",
            ],
            exit_condition="you can derive and verify norm, cosine similarity, orthogonality, and projection for individual and batched vectors.",
            next_lesson="12 — Matrices and linear transformations.",
        ),
    )


def build_math_matrices() -> None:
    cells = [
        md(
            r"""
            ## 1. A matrix is a linear map

            For $A\in\mathbb{R}^{m\times n}$ and $x\in\mathbb{R}^n$, $Ax\in\mathbb{R}^m$.
            Columns show where input basis vectors go. Rows define output coordinates as dot
            products with the input.

            Matrix multiplication composes maps: if $B:\mathbb{R}^k\to\mathbb{R}^n$ and
            $A:\mathbb{R}^n\to\mathbb{R}^m$, then $AB:\mathbb{R}^k\to\mathbb{R}^m$.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
                out = np.zeros((a.shape[0], b.shape[1]), dtype=np.result_type(a, b))
                for i in range(a.shape[0]):
                    for j in range(b.shape[1]):
                        for k in range(a.shape[1]):
                            out[i, j] += a[i, k] * b[k, j]
                return out

            A = np.array([[2.0, 0.0], [0.0, 0.5]])
            B = np.array([[1.0, -1.0], [1.0, 1.0]])
            np.testing.assert_allclose(matmul(A, B), A @ B)
            print("A @ B =\\n", A @ B)
            print("B @ A =\\n", B @ A)
            assert not np.allclose(A @ B, B @ A)
            """
        ),
        md(
            r"""
            ## 2. Affine layers

            Neural-network “linear” layers usually include a bias and are therefore affine:

            $$Y=XW^\top+b$$

            Shapes: $X=[B,C_{in}]$, $W=[C_{out},C_{in}]$, $b=[C_{out}]$,
            $Y=[B,C_{out}]$. PyTorch stores `Linear.weight` in the displayed $W$ orientation.
            """
        ),
        code(
            """
            torch.manual_seed(42)
            X = torch.randn(3, 4)       # [B,Cin]
            W = torch.randn(5, 4)       # [Cout,Cin]
            b = torch.randn(5)          # [Cout]
            manual = X @ W.T + b        # [B,Cout]

            layer = torch.nn.Linear(4, 5)
            with torch.no_grad():
                layer.weight.copy_(W)
                layer.bias.copy_(b)
            torch.testing.assert_close(manual, layer(X))
            print("output shape:", manual.shape)
            """
        ),
        md(
            r"""
            ## 3. Rank and information capacity

            Matrix rank is the number of independent row/column directions. A map with rank less
            than `min(m,n)` collapses at least one direction and cannot be inverted on the whole
            input space. This matrix rank is unrelated to tensor `ndim`.
            """
        ),
        code(
            """
            full = np.array([[1.0, 0.0], [0.0, 1.0]])
            deficient = np.array([[1.0, 2.0], [2.0, 4.0]])
            print("full rank:", np.linalg.matrix_rank(full))
            print("deficient rank:", np.linalg.matrix_rank(deficient))

            x1 = np.array([2.0, 0.0])
            null_direction = np.array([-2.0, 1.0])
            np.testing.assert_allclose(deficient @ x1, deficient @ (x1 + null_direction))
            """
        ),
        md(
            r"""
            ## 4. Basis changes and conditioning

            A basis is a coordinate system. If columns of invertible $P$ are new basis vectors,
            new coordinates are $P^{-1}x$. Nearly dependent basis vectors make $P$ ill-conditioned:
            small numeric/input changes cause large coordinate changes.
            """
        ),
        code(
            """
            good_basis = np.array([[1.0, 0.0], [0.0, 1.0]])
            bad_basis = np.array([[1.0, 1.0], [1.0, 1.000001]])
            vector = np.array([2.0, 1.0])

            for name, basis in {"good": good_basis, "bad": bad_basis}.items():
                coordinates = np.linalg.solve(basis, vector)
                reconstruction = basis @ coordinates
                print(name, "condition=", np.linalg.cond(basis), "coordinates=", coordinates)
                np.testing.assert_allclose(reconstruction, vector, atol=1e-9)
            """
        ),
    ]
    write(
        "12_matrices_linear_maps.ipynb",
        lesson(
            number=12,
            title="Matrices and Linear Transformations",
            coverage="V2 2.3–2.4",
            why="Every embedding lookup, projection, attention transform, and MLP layer depends on linear algebra. Shape derivation and rank reasoning reveal what these layers can preserve or discard.",
            objectives=[
                "Implement matrix multiplication and interpret it as composition.",
                "Derive the shapes and orientation of an affine neural-network layer.",
                "Distinguish matrix rank from tensor rank/ndim.",
                "Connect basis choice and conditioning to numerical sensitivity.",
            ],
            cells=cells,
            failures=[
                "Inner-dimension mismatch: the proposed maps cannot be composed.",
                "Weight transpose error: an affine layer projects the wrong axes or fails its shape check.",
                "Rank collapse: distinct input directions become indistinguishable.",
                "Ill-conditioning: tiny perturbations create large solution or gradient changes.",
            ],
            exercises=[
                "Derive and implement a batched affine layer for `[B,T,Cin]` input.",
                "Construct a 3×3 rank-2 matrix and find a nonzero vector in its null space.",
                "Show numerically that matrix multiplication is associative but generally not commutative.",
            ],
            exit_condition="you can implement an affine map, predict every dimension, and explain rank loss and conditioning in geometric terms.",
            next_lesson="13 — Eigenvalues, SVD, and low-rank structure.",
        ),
    )


def build_math_optional_svd() -> None:
    cells = [
        md(
            r"""
            ## 1. Eigenvectors reveal invariant directions

            For square $A$, an eigenvector $v\ne0$ satisfies $Av=\lambda v$. The transformation
            changes only its scale (and possibly sign/phase), not its direction. Not every matrix
            has a full real orthogonal eigenbasis; symmetric real matrices do.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            A = np.array([[3.0, 1.0], [1.0, 3.0]])
            eigenvalues, eigenvectors = np.linalg.eigh(A)  # for symmetric matrices
            print("eigenvalues:", eigenvalues)
            for value, vector in zip(eigenvalues, eigenvectors.T):
                np.testing.assert_allclose(A @ vector, value * vector)
            np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(2), atol=1e-12)
            """
        ),
        md(
            r"""
            ## 2. SVD works for every rectangular matrix

            $$A=U\Sigma V^\top$$

            $V^\top$ rotates input coordinates, diagonal $\Sigma$ scales independent directions,
            and $U$ rotates into output coordinates. Singular values are nonnegative and sorted.
            The number of nonzero singular values is the matrix rank.
            """
        ),
        code(
            """
            rng = np.random.default_rng(42)
            matrix = rng.normal(size=(6, 4))
            U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
            reconstructed = U @ np.diag(singular_values) @ Vt
            np.testing.assert_allclose(reconstructed, matrix)
            print("shapes:", U.shape, singular_values.shape, Vt.shape)
            print("singular values:", singular_values.round(3))
            """
        ),
        md(
            r"""
            ## 3. Best low-rank approximation

            Keeping the largest $k$ singular values produces the best rank-$k$ approximation under
            Frobenius and spectral norms. The discarded singular values exactly characterize error:

            $$\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2$$

            This explains compression and why learned low-rank adapters can express useful updates
            with far fewer trainable parameters.
            """
        ),
        code(
            """
            for k in range(1, len(singular_values) + 1):
                approx = U[:, :k] @ np.diag(singular_values[:k]) @ Vt[:k]
                measured = np.linalg.norm(matrix - approx, ord="fro")
                predicted = np.sqrt(np.sum(singular_values[k:] ** 2))
                np.testing.assert_allclose(measured, predicted, atol=1e-12)
                params = k * (matrix.shape[0] + matrix.shape[1])
                print(f"rank={k}: error={measured:.3f}, factor values={params}")
            """
        ),
        md(
            r"""
            ## 4. LoRA connection

            A frozen weight $W\in\mathbb{R}^{d_{out}\times d_{in}}$ can receive a rank-$r$ update
            $\Delta W=BA$, where $B=[d_{out},r]$ and $A=[r,d_{in}]$. Instead of training
            $d_{out}d_{in}$ values, train $r(d_{out}+d_{in})$ values. Low rank is a constraint on
            the update, not necessarily on the base weight.
            """
        ),
        code(
            """
            torch.manual_seed(42)
            d_in, d_out, rank = 16, 12, 2
            W = torch.randn(d_out, d_in)
            A_factor = torch.randn(rank, d_in)
            B_factor = torch.randn(d_out, rank)
            delta = B_factor @ A_factor
            x = torch.randn(5, d_in)

            direct = x @ (W + delta).T
            factored = x @ W.T + (x @ A_factor.T) @ B_factor.T
            torch.testing.assert_close(direct, factored)
            print("full update values:", W.numel(), "low-rank values:", A_factor.numel() + B_factor.numel())
            """
        ),
    ]
    write(
        "13_optional_eigen_svd_low_rank.ipynb",
        lesson(
            number=13,
            title="Optional: Eigenvalues, SVD, and Low-Rank Structure",
            coverage="Added mathematical foundation for V2 compression and LoRA",
            why="Spectral structure explains dominant directions, matrix conditioning, compression, and low-rank adaptation. SVD provides a concrete bridge from linear algebra to modern model efficiency.",
            objectives=[
                "Verify an eigendecomposition for a symmetric matrix.",
                "Interpret and reconstruct a rectangular matrix with SVD.",
                "Measure rank-k approximation error from discarded singular values.",
                "Relate factorized low-rank updates to LoRA parameter savings.",
            ],
            cells=cells,
            failures=[
                "General eigensolver assumptions: nonsymmetric matrices can have complex or defective decompositions.",
                "Rank chosen only by parameter count: approximation error or downstream quality becomes unacceptable.",
                "Factor overhead ignored: low-rank factors can exceed the original size when rank is not small.",
                "Base/update confusion: LoRA constrains the adaptation, not the pretrained matrix itself.",
            ],
            exercises=[
                "Create a known rank-2 matrix from two factors and confirm only two singular values are nonzero.",
                "Plot reconstruction error versus rank for a noisy low-rank matrix.",
                "Calculate the LoRA parameter ratio for a 4096×4096 weight at ranks 4, 8, and 64.",
            ],
            exit_condition="you can reconstruct with SVD, choose a rank using an error tradeoff, and derive the cost of a low-rank update.",
            next_lesson="14 — Derivatives and local sensitivity.",
        ),
    )


def build_math_derivatives() -> None:
    cells = [
        md(
            r"""
            ## 1. A derivative is local sensitivity

            $$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

            It answers: for a tiny input change, what first-order output change should we expect?
            $f(x+\Delta x)\approx f(x)+f'(x)\Delta x$. Training uses derivatives of loss with
            respect to millions of parameters to choose a local improvement direction.
            """
        ),
        code(
            """
            import numpy as np
            import torch
            import matplotlib.pyplot as plt

            def f(x):
                return x**3 - 2 * x + 1

            def analytic_derivative(x):
                return 3 * x**2 - 2

            def central_difference(fn, x, h):
                return (fn(x + h) - fn(x - h)) / (2 * h)

            point = 1.5
            for h in (1e-1, 1e-3, 1e-5, 1e-8):
                estimate = central_difference(f, point, h)
                print(f"h={h:.0e}, derivative={estimate:.10f}, error={abs(estimate-analytic_derivative(point)):.2e}")
            """
        ),
        md(
            """
            ## 2. Finite differences have two competing errors

            Large `h` has truncation error: the secant spans too much curvature. Extremely small
            `h` has floating-point cancellation: nearly equal values are subtracted. Gradient
            checking therefore uses a small but not microscopic step, often with float64.
            """
        ),
        code(
            """
            hs = np.logspace(-14, -1, 80)
            errors = np.array([abs(central_difference(f, point, h) - analytic_derivative(point)) for h in hs])
            best = hs[errors.argmin()]
            print(f"best tested h={best:.2e}, error={errors.min():.2e}")

            plt.figure(figsize=(6, 3))
            plt.loglog(hs, errors)
            plt.xlabel("finite-difference step h")
            plt.ylabel("absolute error")
            plt.title("Truncation versus floating-point cancellation")
            plt.grid(True, which="both", alpha=0.3)
            plt.show()
            """
        ),
        md(
            r"""
            ## 3. Partial derivatives and gradients

            For scalar $L(\theta_1,\ldots,\theta_n)$, the gradient stacks partial derivatives:

            $$\nabla_\theta L = [\partial L/\partial\theta_1,\ldots,\partial L/\partial\theta_n]$$

            It points in the direction of steepest local increase under Euclidean geometry;
            `-gradient` is the steepest local decrease direction.
            """
        ),
        code(
            """
            def loss(theta: np.ndarray) -> float:
                x, y = theta
                return (x - 2.0) ** 2 + 3.0 * (y + 1.0) ** 2

            theta = np.array([4.0, 2.0])
            analytic = np.array([2 * (theta[0] - 2), 6 * (theta[1] + 1)])
            numeric = np.array([
                central_difference(lambda value: loss(np.array([value, theta[1]])), theta[0], 1e-5),
                central_difference(lambda value: loss(np.array([theta[0], value])), theta[1], 1e-5),
            ])
            np.testing.assert_allclose(analytic, numeric, rtol=1e-6)
            direction = -analytic / np.linalg.norm(analytic)
            assert loss(theta + 1e-3 * direction) < loss(theta)
            print("gradient:", analytic, "verified numerically")
            """
        ),
        md(
            r"""
            ## 4. Curvature and PyTorch verification

            The second derivative measures how slope changes. Positive curvature near a stationary
            point suggests a local minimum in one dimension; negative suggests a local maximum.
            In many dimensions the Hessian contains all second partial derivatives, but optimizers
            usually avoid constructing it explicitly.
            """
        ),
        code(
            """
            x = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
            y = x**3 - 2 * x + 1
            first, = torch.autograd.grad(y, x, create_graph=True)
            second, = torch.autograd.grad(first, x)
            print("value:", y.item(), "first:", first.item(), "second:", second.item())
            assert abs(first.item() - analytic_derivative(1.5)) < 1e-12
            assert abs(second.item() - 6 * 1.5) < 1e-12
            """
        ),
    ]
    write(
        "14_derivatives_sensitivity.ipynb",
        lesson(
            number=14,
            title="Derivatives and Local Sensitivity",
            coverage="V2 2.5",
            why="Gradient-based learning is repeated sensitivity analysis. Understanding approximation, direction, and curvature makes backpropagation and optimization less mysterious.",
            objectives=[
                "Interpret derivatives as local linear approximations.",
                "Estimate derivatives with central finite differences.",
                "Build and numerically verify a multivariable gradient.",
                "Connect second derivatives to curvature and optimization behavior.",
            ],
            cells=cells,
            failures=[
                "Step too large: finite differences measure curvature rather than a local slope.",
                "Step too small: cancellation and precision dominate the estimate.",
                "Stationary-point assumption: zero gradient is called a minimum without checking curvature or neighbors.",
                "Global conclusion from local data: a descent direction does not promise the best final solution.",
            ],
            exercises=[
                "Derive and verify the derivative of `sin(x) * exp(x)` at three points.",
                "Implement a generic finite-difference gradient for a vector input.",
                "Find and classify stationary points of `x**3 - 3*x` using first and second derivatives.",
            ],
            exit_condition="you can derive a gradient, verify it numerically, and explain why finite differences eventually worsen as h shrinks.",
            next_lesson="15 — Chain rule and computation graphs.",
        ),
    )


def build_math_chain_rule() -> None:
    cells = [
        md(
            r"""
            ## 1. The chain rule composes sensitivities

            If $y=f(u)$ and $u=g(x)$, then

            $$\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}.$$

            A computation graph breaks a complex function into primitive local operations. The
            forward pass saves intermediate values. The reverse pass multiplies each local
            derivative by the derivative arriving from downstream—its **upstream gradient**.
            """
        ),
        code(
            """
            import math
            import torch

            # Forward: L = (w*x + b - target)^2
            x, target = 2.0, 5.0
            w, b = 1.5, -0.5
            z = w * x
            prediction = z + b
            error = prediction - target
            loss = error**2

            # Reverse: start with dL/dL = 1 and walk backward.
            dloss_dloss = 1.0
            dloss_derror = dloss_dloss * 2 * error
            dloss_dprediction = dloss_derror * 1.0
            dloss_dz = dloss_dprediction * 1.0
            dloss_db = dloss_dprediction * 1.0
            dloss_dw = dloss_dz * x
            dloss_dx = dloss_dz * w

            print("loss:", loss, "dL/dw:", dloss_dw, "dL/db:", dloss_db, "dL/dx:", dloss_dx)
            """
        ),
        md(
            r"""
            ## 2. Branches add gradient contributions

            When one value influences the loss through multiple paths, the multivariable chain
            rule **sums** contributions. For $L=x^2+3x$, the two paths give $2x$ and $3$, so
            $dL/dx=2x+3$. Autograd engines accumulate into `.grad`; overwriting loses paths.
            """
        ),
        code(
            """
            x_value = 4.0
            path_square = 2 * x_value
            path_linear = 3.0
            total_gradient = path_square + path_linear
            assert total_gradient == 11.0

            x_torch = torch.tensor(x_value, requires_grad=True)
            branched_loss = x_torch**2 + 3 * x_torch
            branched_loss.backward()
            assert x_torch.grad.item() == total_gradient
            print("branch contributions:", path_square, "+", path_linear, "=", total_gradient)
            """
        ),
        md(
            r"""
            ## 3. Vector–Jacobian products

            A vector function $y=f(x)$ has Jacobian $J_{ij}=\partial y_i/\partial x_j$.
            Reverse-mode autodiff does not normally materialize the full Jacobian. Given upstream
            row vector $v^\top=\partial L/\partial y$, it computes $v^\top J$ efficiently. This is
            ideal for ML because the final loss is scalar while parameters are numerous.
            """
        ),
        code(
            """
            x = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)

            def vector_fn(value: torch.Tensor) -> torch.Tensor:
                return torch.stack((value[0] * value[1], value[0] ** 2 + value[1]))

            jacobian = torch.autograd.functional.jacobian(vector_fn, x)
            upstream = torch.tensor([0.5, -2.0], dtype=torch.float64)
            expected_vjp = upstream @ jacobian
            actual_vjp, = torch.autograd.grad(vector_fn(x), x, grad_outputs=upstream)
            torch.testing.assert_close(actual_vjp, expected_vjp)
            print("Jacobian:\\n", jacobian)
            print("VJP:", actual_vjp)
            """
        ),
        md(
            """
            ## 4. Verify the manual graph

            PyTorch records operations involving tensors that require gradients. Calling
            `backward()` performs a reverse topological traversal and accumulates leaf gradients.
            Gradients accumulate across calls by design, so training loops must clear them.
            """
        ),
        code(
            """
            w_t = torch.tensor(w, dtype=torch.float64, requires_grad=True)
            b_t = torch.tensor(b, dtype=torch.float64, requires_grad=True)
            x_t = torch.tensor(x if isinstance(x, float) else 2.0, dtype=torch.float64, requires_grad=True)
            torch_loss = (w_t * x_t + b_t - target) ** 2
            torch_loss.backward()

            assert math.isclose(w_t.grad.item(), dloss_dw)
            assert math.isclose(b_t.grad.item(), dloss_db)
            assert math.isclose(x_t.grad.item(), dloss_dx)
            print("manual reverse pass matches PyTorch")
            """
        ),
    ]
    write(
        "15_chain_rule_graphs.ipynb",
        lesson(
            number=15,
            title="Chain Rule and Computation Graphs",
            coverage="V2 2.6, 3.3",
            why="Backpropagation is the chain rule organized for a graph with reused intermediate values. Seeing upstream gradients and accumulation now makes later autograd implementation mechanical.",
            objectives=[
                "Perform a manual forward and reverse pass through scalar operations.",
                "Accumulate gradient contributions at graph branches.",
                "Explain reverse mode as vector–Jacobian products.",
                "Verify manual derivatives against PyTorch autograd.",
            ],
            cells=cells,
            failures=[
                "Missing branch contribution: a reused value receives only one path's gradient.",
                "Wrong local derivative: every upstream result before that node may look correct while earlier gradients fail.",
                "Stale accumulated gradients: updates contain contributions from earlier batches.",
                "Detached graph: conversion or an in-place operation removes the differentiable path.",
            ],
            exercises=[
                "Manually backpropagate through `sigmoid(w*x+b)` with a squared-error loss.",
                "Draw the graph for `x*x + x` and explain why the two appearances of x contribute separately.",
                "Compute one Jacobian explicitly and verify two different VJPs without constructing it in the VJP calculation.",
            ],
            exit_condition="you can walk a graph backward, state each local derivative, and sum gradients wherever paths meet.",
            next_lesson="16 — Probability and statistics.",
        ),
    )


def build_math_probability() -> None:
    cells = [
        md(
            r"""
            ## 1. Random variables and distributions

            A discrete distribution assigns nonnegative mass $p(x)$ that sums to one. A random
            variable is a numerical function of an outcome. Its expectation is a probability-
            weighted average, not necessarily a value the variable can actually take:

            $$\mathbb{E}[X]=\sum_x x\,p(x),\qquad
            \operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2].$$
            """
        ),
        code(
            """
            import numpy as np
            import torch

            rng = np.random.default_rng(42)
            outcomes = np.array([0.0, 1.0, 2.0])
            probabilities = np.array([0.2, 0.5, 0.3])
            assert np.all(probabilities >= 0) and np.isclose(probabilities.sum(), 1.0)

            expectation = np.sum(outcomes * probabilities)
            variance = np.sum((outcomes - expectation) ** 2 * probabilities)
            samples = rng.choice(outcomes, size=100_000, p=probabilities)
            print("exact mean/variance:", expectation, variance)
            print("sample mean/variance:", samples.mean(), samples.var())
            """
        ),
        md(
            r"""
            ## 2. Conditional probability and Bayes' rule

            $$p(A\mid B)=\frac{p(A,B)}{p(B)},\qquad
            p(A\mid B)=\frac{p(B\mid A)p(A)}{p(B)}.$$

            Conditioning changes the relevant population. Language models are conditional
            distributions: the same next token can receive very different probability under a
            different prefix.
            """
        ),
        code(
            """
            # Rows: context is question / statement. Columns: next token is '?' / '.'.
            joint = np.array([[0.30, 0.10], [0.05, 0.55]])
            assert np.isclose(joint.sum(), 1.0)
            context_mass = joint.sum(axis=1, keepdims=True)
            conditional = joint / context_mass
            print("p(punctuation | context):\\n", conditional)
            np.testing.assert_allclose(conditional.sum(axis=1), 1.0)
            """
        ),
        md(
            r"""
            ## 3. Covariance and correlation

            $$\operatorname{Cov}(X,Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$$

            Positive covariance means variables tend to move together; negative means opposite
            movement. Correlation divides by standard deviations. Neither implies causation, and
            zero covariance does not generally imply independence.
            """
        ),
        code(
            """
            x = rng.normal(size=10_000)
            y = 2.0 * x + rng.normal(scale=0.5, size=10_000)
            z = x**2
            covariance = np.cov(np.stack([x, y]), ddof=0)
            correlation = np.corrcoef(x, y)[0, 1]
            print("covariance matrix:\\n", covariance)
            print("corr(x,y):", correlation, "corr(x,x²):", np.corrcoef(x, z)[0, 1])
            assert correlation > 0.9
            """
        ),
        md(
            r"""
            ## 4. Sampling error and the law of large numbers

            A finite sample statistic varies across datasets. As independent sample size grows,
            the sample mean concentrates around the population expectation, with standard error
            proportional to $1/\sqrt{n}$. One seed is one draw, not proof of a general behavior.
            """
        ),
        code(
            """
            true_mean = expectation
            for n in (10, 100, 1_000, 10_000):
                estimates = np.array([
                    rng.choice(outcomes, size=n, p=probabilities).mean() for _ in range(200)
                ])
                print(f"n={n:5}: estimate mean={estimates.mean():.3f}, std across trials={estimates.std():.4f}")

            torch.manual_seed(42)
            categorical = torch.distributions.Categorical(probs=torch.tensor(probabilities))
            torch_samples = categorical.sample((10_000,))
            assert torch_samples.shape == (10_000,)
            """
        ),
    ]
    write(
        "16_probability_statistics.ipynb",
        lesson(
            number=16,
            title="Probability and Statistics",
            coverage="V2 2.7 plus covariance",
            why="Language models represent conditional probability distributions, while experiments estimate uncertain quantities from finite samples. Probability describes the model; statistics keeps conclusions honest.",
            objectives=[
                "Validate, sample, and summarize a discrete distribution.",
                "Compute conditional probability from a joint table.",
                "Interpret variance, covariance, and correlation.",
                "Measure how sample-size changes estimator variability.",
            ],
            cells=cells,
            failures=[
                "Invalid probability vector: negative mass or a sum different from one breaks sampling semantics.",
                "Conditioning on the wrong axis: rows normalize while the intended event lives in columns.",
                "Correlation as causation: a predictive relationship is mistaken for a mechanism.",
                "Single-run certainty: noise from dataset sampling or initialization is ignored.",
            ],
            exercises=[
                "Compute a posterior with Bayes' rule for a diagnostic test with an imbalanced prior.",
                "Construct dependent random variables with approximately zero correlation.",
                "Empirically verify that standard error shrinks near `1/sqrt(n)`.",
            ],
            exit_condition="you can move between joint, marginal, and conditional distributions and quantify uncertainty in a sample estimate.",
            next_lesson="17 — Likelihood and information.",
        ),
    )


def build_math_information() -> None:
    cells = [
        md(
            r"""
            ## 1. Likelihood scores parameters using observed data

            Probability treats parameters as fixed and asks about outcomes. Likelihood treats the
            observed outcomes as fixed and compares parameter settings. For independent tokens:

            $$\mathcal{L}(\theta)=\prod_t p_\theta(x_t\mid x_{<t}),\qquad
            \log\mathcal{L}(\theta)=\sum_t\log p_\theta(x_t\mid x_{<t}).$$

            Logs turn products into sums and prevent extremely small products from underflowing.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            probabilities = np.array([0.1] * 200, dtype=np.float32)
            product = probabilities.prod()
            log_likelihood = np.log(probabilities.astype(np.float64)).sum()
            print("float32 product:", product)
            print("log likelihood:", log_likelihood)
            assert product == 0.0 and np.isfinite(log_likelihood)
            """
        ),
        md(
            r"""
            ## 2. Entropy is expected surprise

            Self-information is $I(x)=-\log p(x)$. Entropy averages surprise under distribution
            $p$:

            $$H(p)=-\sum_i p_i\log p_i.$$

            Natural logs measure **nats**; base-2 logs measure **bits**. A deterministic categorical
            distribution has zero entropy; a uniform $V$-class distribution has maximum entropy
            $\log V$.
            """
        ),
        code(
            """
            def entropy(p: np.ndarray) -> float:
                p = np.asarray(p, dtype=np.float64)
                positive = p > 0
                return float(-np.sum(p[positive] * np.log(p[positive])))

            for p in (np.array([1.0, 0.0, 0.0]), np.array([0.8, 0.1, 0.1]), np.ones(3) / 3):
                print(p, "entropy=", entropy(p))
            assert np.isclose(entropy(np.ones(3) / 3), np.log(3))
            """
        ),
        md(
            r"""
            ## 3. Cross-entropy and KL divergence

            $$H(p,q)=-\sum_i p_i\log q_i,\qquad
            D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}.$$

            Their identity is $H(p,q)=H(p)+D_{KL}(p\|q)$. Since $H(p)$ does not depend on model
            $q$, minimizing cross-entropy fits $q$ toward data distribution $p$. KL is asymmetric
            and is not a metric.
            """
        ),
        code(
            """
            def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
                if np.any((p > 0) & (q <= 0)):
                    return float("inf")
                positive = p > 0
                return float(-np.sum(p[positive] * np.log(q[positive])))

            p = np.array([0.7, 0.2, 0.1])
            q = np.array([0.5, 0.4, 0.1])
            kl_pq = cross_entropy(p, q) - entropy(p)
            kl_qp = cross_entropy(q, p) - entropy(q)
            print("H(p):", entropy(p), "H(p,q):", cross_entropy(p, q), "KL(p||q):", kl_pq)
            assert kl_pq >= 0 and not np.isclose(kl_pq, kl_qp)
            """
        ),
        md(
            r"""
            ## 4. Token loss and perplexity

            With a one-hot target token $y$, cross-entropy is simply $-\log q_y$. Mean token NLL
            exponentiates to perplexity: $\operatorname{PPL}=\exp(\text{mean NLL})$. Roughly, it is
            the effective number of equally likely choices under the model—but comparisons require
            the same tokenizer and evaluation protocol.
            """
        ),
        code(
            """
            token_probabilities = torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64)
            targets = torch.tensor([0, 1, 2, 3])
            losses = -torch.log(token_probabilities[targets])
            mean_nll = losses.mean()
            perplexity = mean_nll.exp()
            print("token losses:", losses.tolist())
            print("mean NLL:", mean_nll.item(), "perplexity:", perplexity.item())
            assert perplexity.item() >= 1.0
            """
        ),
    ]
    write(
        "17_likelihood_information.ipynb",
        lesson(
            number=17,
            title="Likelihood and Information Theory",
            coverage="V2 2.8",
            why="Next-token training is maximum likelihood expressed as cross-entropy. Entropy, KL divergence, and perplexity provide the language needed to interpret that objective and its evaluation.",
            objectives=[
                "Explain why sequence likelihood is computed in log space.",
                "Calculate self-information and entropy.",
                "Verify the relationship among entropy, cross-entropy, and KL divergence.",
                "Convert mean token NLL into perplexity with proper caveats.",
            ],
            cells=cells,
            failures=[
                "Probability product underflow: a valid sequence receives numeric zero likelihood.",
                "Log of zero: impossible model events produce infinite loss.",
                "KL symmetry assumption: reversing arguments changes the question and result.",
                "Tokenizer-blind perplexity: scores with different token units are compared directly.",
            ],
            exercises=[
                "Compute entropy in bits and nats for the same distribution and explain the unit conversion.",
                "Find a q distribution that gives infinite cross-entropy under a chosen p and explain why.",
                "Show that uniform predictions over V classes produce perplexity V.",
            ],
            exit_condition="you can derive next-token NLL from maximum likelihood and explain entropy, cross-entropy, KL, and perplexity precisely.",
            next_lesson="18 — Softmax, cross-entropy, and numerical stability.",
        ),
    )


def build_math_softmax() -> None:
    cells = [
        md(
            r"""
            ## 1. Softmax converts logits to a categorical distribution

            $$\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.$$

            Adding the same constant to every logit changes neither probability nor ranking.
            This shift invariance permits subtracting the largest logit before exponentiation.
            """
        ),
        code(
            """
            import numpy as np
            import torch
            import torch.nn.functional as F

            def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
                shifted = logits - np.max(logits, axis=axis, keepdims=True)
                exponentials = np.exp(shifted)
                return exponentials / exponentials.sum(axis=axis, keepdims=True)

            logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
            probabilities = stable_softmax(logits)
            print(probabilities)
            np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
            np.testing.assert_allclose(probabilities[0], probabilities[1])
            """
        ),
        md(
            r"""
            ## 2. Log-sum-exp is the stable normalizer

            $$\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m},\quad m=\max_j z_j.$$

            Log-softmax is $z_i-\operatorname{LSE}(z)$. Fused cross-entropy uses log-softmax
            directly instead of computing probabilities and then taking their logs.
            """
        ),
        code(
            """
            def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
                maximum = np.max(x, axis=axis, keepdims=True)
                result = maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
                return result if keepdims else np.squeeze(result, axis=axis)

            def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
                return x - logsumexp(x, axis=axis, keepdims=True)

            np.testing.assert_allclose(np.exp(log_softmax(logits)), probabilities)
            print("stable log probabilities:\\n", log_softmax(logits))
            """
        ),
        md(
            r"""
            ## 3. Batched next-token cross-entropy

            Logits have shape `[B,T,V]`; integer targets have `[B,T]`. Flatten only after confirming
            axes. For each position choose the target log-probability, negate, then average or sum.
            Padding positions require an explicit mask or ignore index.
            """
        ),
        code(
            """
            rng = np.random.default_rng(42)
            B, T, V = 2, 3, 5
            batch_logits = rng.normal(size=(B, T, V))
            targets = rng.integers(0, V, size=(B, T))
            log_probs = log_softmax(batch_logits, axis=-1)
            selected = np.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
            manual_loss = -selected.mean()

            torch_loss = F.cross_entropy(
                torch.tensor(batch_logits, dtype=torch.float64).reshape(-1, V),
                torch.tensor(targets).reshape(-1),
            )
            np.testing.assert_allclose(manual_loss, torch_loss.item(), rtol=1e-12)
            print("cross-entropy:", manual_loss)
            """
        ),
        md(
            r"""
            ## 4. The elegant gradient

            For one example with one-hot target $y$, softmax-cross-entropy has gradient
            $\partial L/\partial z=p-y$. Increasing the target logit lowers loss; non-target
            logits receive positive gradients proportional to their probability.
            """
        ),
        code(
            """
            z = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64, requires_grad=True)
            target = torch.tensor([2])
            loss = F.cross_entropy(z.unsqueeze(0), target)
            loss.backward()
            expected = torch.softmax(z.detach(), dim=-1)
            expected[target] -= 1.0
            torch.testing.assert_close(z.grad, expected)
            print("probability minus one-hot:", z.grad)

            # A finite-difference spot check.
            epsilon = 1e-6
            plus = z.detach().clone(); plus[0] += epsilon
            minus = z.detach().clone(); minus[0] -= epsilon
            numeric = (F.cross_entropy(plus[None], target) - F.cross_entropy(minus[None], target)) / (2 * epsilon)
            torch.testing.assert_close(numeric, z.grad[0], atol=1e-8, rtol=1e-6)
            """
        ),
    ]
    write(
        "18_softmax_cross_entropy.ipynb",
        lesson(
            number=18,
            title="Softmax, Cross-Entropy, and Numerical Stability",
            coverage="V2 2.9–2.10",
            why="This is the core next-token objective. A correct but numerically naive formula can overflow even when the mathematically equivalent stable version is well behaved.",
            objectives=[
                "Implement stable softmax, log-sum-exp, and log-softmax.",
                "Compute next-token cross-entropy for `[B,T,V]` logits.",
                "Verify the fused result against PyTorch.",
                "Derive and gradient-check `softmax(logits) - one_hot(target)`.",
            ],
            cells=cells,
            failures=[
                "Exponent overflow: raw large logits create infinity and NaN.",
                "Wrong softmax axis: probabilities normalize across time or batch rather than vocabulary.",
                "Target/logit misalignment: flattened positions no longer refer to the same tokens.",
                "Padding included: easy padding predictions distort loss and gradients.",
            ],
            exercises=[
                "Add a boolean padding mask and compute a mean over valid tokens only.",
                "Measure probability and gradient changes at temperatures 0.5, 1, and 2.",
                "Implement label smoothing manually and compare it to PyTorch cross-entropy.",
            ],
            exit_condition="you can derive, implement, and gradient-check stable batched next-token cross-entropy without calling softmax first.",
            next_lesson="19 — Optimization.",
        ),
    )


def build_math_optimization() -> None:
    cells = [
        md(
            r"""
            ## 1. Optimization turns gradients into parameter updates

            Full-batch gradient descent uses
            $\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$. Stochastic gradient descent (SGD)
            estimates the gradient from a minibatch. The learning rate $\eta$ controls the step,
            not the final destination directly; too small is slow, too large can oscillate or diverge.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            def objective(theta: np.ndarray) -> float:
                x, y = theta
                return 0.5 * (10 * x**2 + y**2)

            def gradient(theta: np.ndarray) -> np.ndarray:
                x, y = theta
                return np.array([10 * x, y])

            def run_gd(lr: float, steps: int = 30) -> list[float]:
                theta = np.array([3.0, 3.0])
                losses = [objective(theta)]
                for _ in range(steps):
                    theta -= lr * gradient(theta)
                    losses.append(objective(theta))
                return losses

            for lr in (0.01, 0.1, 0.21):
                losses = run_gd(lr)
                print(f"lr={lr:.2f}: first={losses[0]:.2f}, final={losses[-1]:.4g}")
            """
        ),
        md(
            r"""
            ## 2. Momentum smooths and accelerates

            One common convention:

            $$v_t=\beta v_{t-1}+g_t,\qquad\theta_t=\theta_{t-1}-\eta v_t.$$

            Momentum accumulates persistent directions and damps alternating ones. Be careful:
            libraries use slightly different notation and update ordering.
            """
        ),
        code(
            """
            def optimize(kind: str, steps: int = 40, lr: float = 0.05) -> tuple[np.ndarray, list[float]]:
                theta = np.array([3.0, 3.0])
                velocity = np.zeros_like(theta)
                first = np.zeros_like(theta)
                second = np.zeros_like(theta)
                losses = []
                for step in range(1, steps + 1):
                    g = gradient(theta)
                    if kind == "sgd":
                        update = g
                    elif kind == "momentum":
                        velocity = 0.9 * velocity + g
                        update = velocity
                    elif kind == "adam":
                        first = 0.9 * first + 0.1 * g
                        second = 0.999 * second + 0.001 * g**2
                        first_hat = first / (1 - 0.9**step)
                        second_hat = second / (1 - 0.999**step)
                        update = first_hat / (np.sqrt(second_hat) + 1e-8)
                    else:
                        raise ValueError(kind)
                    theta -= lr * update
                    losses.append(objective(theta))
                return theta, losses

            for kind in ("sgd", "momentum", "adam"):
                theta, losses = optimize(kind)
                print(f"{kind:8}: theta={theta.round(4)}, loss={losses[-1]:.5f}")
            """
        ),
        md(
            r"""
            ## 3. Adam adapts per-coordinate steps

            Adam tracks exponential averages of gradients $m_t$ and squared gradients $v_t$,
            corrects their early bias toward zero, and updates with

            $$\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

            Adaptive scaling is useful but does not remove the need to tune learning rate or inspect
            gradient quality.
            """
        ),
        code(
            """
            # Match one manual Adam step with PyTorch.
            parameter = torch.tensor([3.0, 3.0], dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.Adam([parameter], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
            loss = 0.5 * (10 * parameter[0] ** 2 + parameter[1] ** 2)
            loss.backward()
            optimizer.step()

            manual_theta, _ = optimize("adam", steps=1, lr=0.05)
            torch.testing.assert_close(parameter.detach(), torch.tensor(manual_theta))
            print("first Adam step:", parameter.detach())
            """
        ),
        md(
            r"""
            ## 4. AdamW decouples weight decay

            L2 regularization adds $\lambda\theta$ to the loss gradient. AdamW instead decays
            parameters separately from the adaptive gradient update. They are equivalent for plain
            SGD under common conventions, but not generally for Adam because Adam rescales gradients.
            Biases and normalization scale parameters are often excluded from decay.
            """
        ),
        code(
            """
            initial = torch.tensor([2.0], dtype=torch.float64)
            adam_l2 = initial.clone().requires_grad_()
            adamw = initial.clone().requires_grad_()
            opt_l2 = torch.optim.Adam([adam_l2], lr=0.1, weight_decay=0.1)
            opt_w = torch.optim.AdamW([adamw], lr=0.1, weight_decay=0.1)

            # Same task gradient, different decay semantics.
            for parameter_value, optimizer_value in ((adam_l2, opt_l2), (adamw, opt_w)):
                (0.5 * (parameter_value - 1) ** 2).backward()
                optimizer_value.step()
            print("Adam + L2:", adam_l2.item(), "AdamW:", adamw.item())
            assert not torch.allclose(adam_l2, adamw)
            """
        ),
    ]
    write(
        "19_optimization.ipynb",
        lesson(
            number=19,
            title="Optimization: SGD, Momentum, Adam, and AdamW",
            coverage="V2 2.11–2.12",
            why="Backpropagation supplies a gradient; the optimizer determines how that noisy local signal becomes a stable trajectory through parameter space.",
            objectives=[
                "Implement gradient descent, momentum, and Adam updates.",
                "Observe learning-rate stability on an anisotropic objective.",
                "Verify a manual Adam step against PyTorch.",
                "Distinguish L2 regularization from decoupled weight decay.",
            ],
            cells=cells,
            failures=[
                "Learning rate too high: loss oscillates, grows, or becomes non-finite.",
                "Missing bias correction: Adam's earliest steps are systematically mis-scaled.",
                "Epsilon misuse: division becomes unstable or the adaptive effect is overwhelmed.",
                "Universal weight decay: biases and normalization parameters are regularized unintentionally.",
            ],
            exercises=[
                "Plot loss curves for all three optimizers under at least three learning rates.",
                "Add stochastic gradient noise and compare trajectory smoothness.",
                "Implement AdamW manually and match five PyTorch steps with fixed gradients.",
            ],
            exit_condition="you can write each update equation, reproduce it in code, and diagnose a bad trajectory from loss and gradient evidence.",
            next_lesson="20 — Mathematical foundations capstone.",
        ),
    )


def build_math_capstone() -> None:
    cells = [
        md(
            r"""
            ## Capstone contract

            Build multiclass softmax regression from NumPy primitives. Derive every tensor shape,
            implement stable cross-entropy, backpropagate analytically, gradient-check parameters,
            optimize with gradient descent, and interpret uncertainty. This integrates V2 Part II
            without relying on PyTorch autograd.

            Shapes: examples $X=[N,D]$, weights $W=[D,K]$, bias $b=[K]$, logits and probabilities
            $[N,K]$, integer targets $[N]$.
            """
        ),
        code(
            """
            import numpy as np
            import torch
            import torch.nn.functional as F

            rng = np.random.default_rng(42)
            examples_per_class, dimensions, classes = 60, 2, 3
            centers = np.array([[-2.0, -1.5], [2.0, -1.0], [0.0, 2.0]])
            X = np.concatenate([
                rng.normal(loc=center, scale=0.7, size=(examples_per_class, dimensions))
                for center in centers
            ]).astype(np.float64)
            targets = np.repeat(np.arange(classes), examples_per_class)
            permutation = rng.permutation(len(X))
            X, targets = X[permutation], targets[permutation]

            split = 140
            train_x, val_x = X[:split], X[split:]
            train_y, val_y = targets[:split], targets[split:]
            print("train/validation:", train_x.shape, val_x.shape, train_y.shape, val_y.shape)
            """
        ),
        md(
            r"""
            ## 1. Forward pass and stable loss

            $$Z=XW+b,\qquad P_{ik}=\frac{e^{Z_{ik}-m_i}}{\sum_j e^{Z_{ij}-m_i}}$$

            Mean cross-entropy is $L=-\frac1N\sum_i\log P_{i,y_i}$. Subtracting the per-row
            maximum preserves softmax while preventing exponent overflow.
            """
        ),
        code(
            """
            def forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray):
                logits = x @ weight + bias
                shifted = logits - logits.max(axis=1, keepdims=True)
                log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                probabilities = np.exp(log_probabilities)
                return logits, log_probabilities, probabilities

            def loss_and_gradients(x, y, weight, bias):
                logits, log_probabilities, probabilities = forward(x, weight, bias)
                loss = -log_probabilities[np.arange(len(x)), y].mean()

                dlogits = probabilities.copy()
                dlogits[np.arange(len(x)), y] -= 1.0
                dlogits /= len(x)
                dweight = x.T @ dlogits
                dbias = dlogits.sum(axis=0)
                return loss, dweight, dbias, logits, probabilities

            weight = rng.normal(scale=0.01, size=(dimensions, classes))
            bias = np.zeros(classes)
            initial = loss_and_gradients(train_x, train_y, weight, bias)
            print("initial loss:", initial[0], "uniform baseline:", np.log(classes))
            np.testing.assert_allclose(initial[4].sum(axis=1), 1.0)
            """
        ),
        md(
            r"""
            ## 2. Backward derivation

            Softmax-cross-entropy gives $dZ=(P-Y)/N$. Matrix calculus then gives

            $$dW=X^\top dZ,\qquad db=\sum_i dZ_i,\qquad dX=dZW^\top.$$

            Every derivative has the same shape as its differentiated quantity. Bias gradients sum
            across the broadcast example axis.
            """
        ),
        code(
            """
            loss, dweight, dbias, logits, probabilities = initial
            assert dweight.shape == weight.shape and dbias.shape == bias.shape
            dlogits = probabilities.copy()
            dlogits[np.arange(len(train_x)), train_y] -= 1
            dlogits /= len(train_x)
            dx = dlogits @ weight.T
            assert dx.shape == train_x.shape
            print("gradient shapes:", dweight.shape, dbias.shape, dx.shape)
            """
        ),
        md(
            """
            ## 3. Gradient check before optimization

            Central finite differences provide an implementation independent of the analytical
            backward formulas. Use float64, deterministic inputs, and a moderate epsilon. Check both
            weight and bias rather than assuming one passing element proves the entire graph.
            """
        ),
        code(
            """
            def finite_difference(parameter, index, evaluate, epsilon=1e-6):
                original = parameter[index]
                parameter[index] = original + epsilon; plus = evaluate()
                parameter[index] = original - epsilon; minus = evaluate()
                parameter[index] = original
                return (plus - minus) / (2 * epsilon)

            evaluate = lambda: loss_and_gradients(train_x, train_y, weight, bias)[0]
            for index in np.ndindex(weight.shape):
                numeric = finite_difference(weight, index, evaluate)
                np.testing.assert_allclose(numeric, dweight[index], rtol=1e-6, atol=1e-8)
            for index in np.ndindex(bias.shape):
                numeric = finite_difference(bias, index, evaluate)
                np.testing.assert_allclose(numeric, dbias[index], rtol=1e-6, atol=1e-8)
            print("all weight and bias gradients passed finite differences")
            """
        ),
        md(
            """
            ## 4. Optimize and evaluate separately

            The training split updates parameters. The validation split estimates behavior on held-out
            examples and must not influence gradients. Track loss and accuracy, and stop immediately
            on non-finite values.
            """
        ),
        code(
            """
            learning_rate = 0.2
            history = []
            for step in range(201):
                loss, dweight, dbias, _, _ = loss_and_gradients(train_x, train_y, weight, bias)
                assert np.isfinite(loss) and np.isfinite(dweight).all() and np.isfinite(dbias).all()
                weight -= learning_rate * dweight
                bias -= learning_rate * dbias
                history.append(loss)

            def evaluate_split(x, y):
                _, log_probabilities, probabilities = forward(x, weight, bias)
                loss = -log_probabilities[np.arange(len(x)), y].mean()
                accuracy = np.mean(probabilities.argmax(axis=1) == y)
                return float(loss), float(accuracy)

            train_metrics = evaluate_split(train_x, train_y)
            validation_metrics = evaluate_split(val_x, val_y)
            print("train loss/accuracy:", train_metrics)
            print("validation loss/accuracy:", validation_metrics)
            assert history[-1] < history[0] * 0.2 and validation_metrics[1] > 0.9
            """
        ),
        md(
            """
            ## 5. Framework verification and uncertainty

            PyTorch should reproduce logits and loss for the same parameters. Predicted probability is
            model confidence under its fitted assumptions, not guaranteed real-world correctness or
            calibration. Inspect ambiguous points rather than treating argmax as the whole model output.
            """
        ),
        code(
            """
            torch_logits = torch.tensor(val_x) @ torch.tensor(weight) + torch.tensor(bias)
            torch_loss = F.cross_entropy(torch_logits, torch.tensor(val_y))
            numpy_logits, _, numpy_probabilities = forward(val_x, weight, bias)
            np.testing.assert_allclose(torch_logits.numpy(), numpy_logits, rtol=1e-12)
            np.testing.assert_allclose(torch_loss.item(), validation_metrics[0], rtol=1e-12)

            confidence = numpy_probabilities.max(axis=1)
            least_confident = np.argsort(confidence)[:5]
            for index in least_confident:
                print("x=", val_x[index].round(2), "target=", val_y[index],
                      "probabilities=", numpy_probabilities[index].round(3))
            """
        ),
    ]
    write(
        "20_mathematical_capstone.ipynb",
        lesson(
            number=20,
            title="Mathematical Foundations Capstone",
            coverage="V2 2.15",
            why="A complete differentiable classifier integrates tensors, matrix multiplication, affine maps, probability, stable cross-entropy, chain-rule gradients, numerical verification, and optimization in one visible system.",
            objectives=[
                "Implement stable multiclass softmax regression entirely in NumPy.",
                "Derive all forward and backward tensor shapes.",
                "Gradient-check every trainable value against finite differences.",
                "Optimize on training data, evaluate held-out data, and verify against PyTorch.",
            ],
            cells=cells,
            failures=[
                "Unstable exponentials: otherwise valid large logits produce NaN.",
                "Missing mean or broadcast reduction: analytical and numerical gradients disagree.",
                "Training/validation leakage: held-out metrics influence parameter updates.",
                "Confidence overclaim: softmax probability is presented as guaranteed correctness.",
            ],
            exercises=[
                "Re-derive the gradients without reading the notebook and annotate every matrix product.",
                "Add L2 regularization and gradient-check the changed objective.",
                "Replace plain gradient descent with your manual momentum and Adam implementations.",
                "Write a short report connecting each Part II module to one line of this capstone.",
            ],
            exit_condition="you can rebuild, derive, gradient-check, optimize, and evaluate this classifier without autograd or copied formulas.",
            next_lesson="21 — Learning from data.",
        ),
    )


def build_nn_learning_from_data() -> None:
    cells = [
        md(
            """
            ## 1. Supervised learning separates fitting from evaluation

            A dataset contains examples from an unknown process. The training split changes model
            parameters. The validation split selects hyperparameters and stopping decisions. The
            test split estimates final generalization once. Reusing test feedback turns it into a
            validation set and biases the estimate.
            """
        ),
        code(
            """
            import numpy as np
            import torch
            import torch.nn.functional as F

            rng = np.random.default_rng(42)
            n = 300
            x = rng.normal(size=(n, 2)).astype(np.float32)
            y_reg = (2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5 + rng.normal(scale=0.3, size=n)).astype(np.float32)
            y_cls = (y_reg > np.median(y_reg)).astype(np.int64)

            indices = rng.permutation(n)
            train_idx, val_idx, test_idx = indices[:200], indices[200:250], indices[250:]
            assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx))
            print("split sizes:", len(train_idx), len(val_idx), len(test_idx))
            """
        ),
        md(
            r"""
            ## 2. Linear regression

            The model $\hat y=Xw+b$ minimizes mean squared error
            $\frac1N\sum_i(\hat y_i-y_i)^2$. The closed-form least-squares solution is useful for
            understanding; gradient methods scale to models where no closed form exists.
            """
        ),
        code(
            """
            X_train = torch.tensor(x[train_idx])
            y_train = torch.tensor(y_reg[train_idx])
            X_val = torch.tensor(x[val_idx])
            y_val = torch.tensor(y_reg[val_idx])

            regressor = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(regressor.parameters(), lr=0.05)
            for _ in range(200):
                prediction = regressor(X_train).squeeze(-1)
                loss = F.mse_loss(prediction, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                val_mse = F.mse_loss(regressor(X_val).squeeze(-1), y_val)
            print("learned weight:", regressor.weight.detach().numpy().round(2),
                  "bias:", regressor.bias.item(), "val MSE:", val_mse.item())
            assert val_mse < 0.2
            """
        ),
        md(
            r"""
            ## 3. Logistic regression

            Binary logistic regression emits logit $z=x\cdot w+b$. Sigmoid maps it to
            $p(y=1\mid x)$. Binary cross-entropy fits probabilities; thresholding at 0.5 is a later
            decision rule and may be inappropriate under class imbalance or asymmetric costs.
            """
        ),
        code(
            """
            classifier = torch.nn.Linear(2, 1)
            targets = torch.tensor(y_cls[train_idx], dtype=torch.float32)
            optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
            for _ in range(250):
                logits = classifier(X_train).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                val_logits = classifier(X_val).squeeze(-1)
                val_predictions = (val_logits.sigmoid() >= 0.5).long()
                accuracy = (val_predictions == torch.tensor(y_cls[val_idx])).float().mean()
            print("validation accuracy:", accuracy.item())
            assert accuracy > 0.85
            """
        ),
        md(
            """
            ## 4. Leakage creates performance you cannot deploy

            Leakage occurs when training inputs or preprocessing contain information unavailable at
            prediction time. Common examples: normalizing with all splits, near-duplicate documents
            across splits, future information in time-series features, and packing adjacent slices
            of the same source into different splits.
            """
        ),
        code(
            """
            # A deliberately leaked feature: it directly contains the label.
            clean_features = torch.tensor(x[test_idx])
            test_targets = torch.tensor(y_cls[test_idx])
            leaked_features = torch.column_stack((clean_features, test_targets.float()))
            leaked_rule = (leaked_features[:, -1] > 0.5).long()
            leaked_accuracy = (leaked_rule == test_targets).float().mean()
            print("impossible-looking leaked accuracy:", leaked_accuracy.item())
            assert leaked_accuracy == 1.0

            # Fit preprocessing statistics only on training data.
            mean = X_train.mean(dim=0, keepdim=True)
            std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
            normalized_test = (clean_features - mean) / std
            assert normalized_test.shape == clean_features.shape
            """
        ),
    ]
    write(
        "21_learning_from_data.ipynb",
        lesson(
            number=21,
            title="Learning from Data",
            coverage="Added foundation and V2 3.10",
            why="Before deep networks, a learner needs the entire empirical workflow: define a task, split data correctly, fit a baseline, choose metrics, and recognize leakage.",
            objectives=[
                "Assign distinct roles to train, validation, and test splits.",
                "Train linear regression with mean-squared error.",
                "Train logistic regression from logits with binary cross-entropy.",
                "Identify feature, preprocessing, duplicate, and temporal leakage.",
            ],
            cells=cells,
            failures=[
                "Test-set tuning: reported test performance is optimistically biased.",
                "Preprocessing leakage: validation/test statistics influence training transforms.",
                "Accuracy-only evaluation: class imbalance hides a useless classifier.",
                "No baseline: complexity is added without proving a simpler model is insufficient.",
            ],
            exercises=[
                "Derive the linear-regression gradients for weight and bias and implement them without autograd.",
                "Create an imbalanced classification dataset and compare accuracy, precision, recall, and a confusion matrix.",
                "List three ways document-level leakage could enter an SLM dataset and how to prevent each.",
            ],
            exit_condition="you can build and evaluate a baseline with leakage-safe splits and explain what information influenced every reported metric.",
            next_lesson="22 — Neurons, activations, and MLPs.",
        ),
    )


def build_nn_mlps() -> None:
    cells = [
        md(
            r"""
            ## 1. A neuron is affine transformation plus nonlinearity

            $$z=w\cdot x+b,\qquad h=\phi(z).$$

            Stacking only affine maps is still one affine map. Nonlinear activations allow a network
            to represent curved and piecewise decision boundaries. An MLP applies this per token in
            a Transformer; attention handles communication between tokens.
            """
        ),
        code(
            """
            import numpy as np
            import torch
            import torch.nn.functional as F

            def relu(x: np.ndarray) -> np.ndarray:
                return np.maximum(x, 0)

            def sigmoid(x: np.ndarray) -> np.ndarray:
                # Stable branch avoids overflow for large negative x.
                positive = x >= 0
                result = np.empty_like(x, dtype=np.float64)
                result[positive] = 1 / (1 + np.exp(-x[positive]))
                exp_x = np.exp(x[~positive])
                result[~positive] = exp_x / (1 + exp_x)
                return result

            values = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
            print("ReLU:", relu(values))
            print("sigmoid:", sigmoid(values))
            assert np.isfinite(sigmoid(values)).all()
            """
        ),
        md(
            """
            ## 2. Activation tradeoffs

            - **Sigmoid/tanh:** smooth and bounded, but saturate and shrink gradients.
            - **ReLU:** cheap and nonsaturating on positives, but zero gradient on negatives.
            - **GELU:** smooth gating used in classic Transformers.
            - **SiLU/Swish:** smooth self-gating used inside SwiGLU in many modern models.

            There is no best activation independent of architecture, initialization, and budget.
            """
        ),
        code(
            """
            x = torch.linspace(-5, 5, 11, requires_grad=True)
            activations = {
                "sigmoid": torch.sigmoid(x),
                "tanh": torch.tanh(x),
                "relu": F.relu(x),
                "gelu": F.gelu(x),
                "silu": F.silu(x),
            }
            for name, output in activations.items():
                gradient, = torch.autograd.grad(output.sum(), x, retain_graph=True)
                print(f"{name:7} output endpoints=({output[0]:.3f},{output[-1]:.3f}) "
                      f"gradient endpoints=({gradient[0]:.3g},{gradient[-1]:.3g})")
            """
        ),
        md(
            r"""
            ## 3. Manual two-layer MLP

            $$H=\phi(XW_1^\top+b_1),\qquad Y=HW_2^\top+b_2.$$

            For token states $X=[B,T,C]$, a Transformer feed-forward network preserves `[B,T]`
            while expanding features from $C$ to $C_{hidden}$ and projecting back to $C$.
            """
        ),
        code(
            """
            torch.manual_seed(42)
            B, T, C, hidden = 2, 3, 4, 12
            X = torch.randn(B, T, C)
            W1, b1 = torch.randn(hidden, C), torch.randn(hidden)
            W2, b2 = torch.randn(C, hidden), torch.randn(C)

            H = F.gelu(X @ W1.T + b1)  # [B,T,hidden]
            Y = H @ W2.T + b2          # [B,T,C]
            assert H.shape == (B, T, hidden) and Y.shape == X.shape

            module = torch.nn.Sequential(torch.nn.Linear(C, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, C))
            with torch.no_grad():
                module[0].weight.copy_(W1); module[0].bias.copy_(b1)
                module[2].weight.copy_(W2); module[2].bias.copy_(b2)
            torch.testing.assert_close(Y, module(X))
            """
        ),
        md(
            """
            ## 4. Nonlinearity solves XOR

            XOR is not linearly separable. A small MLP can bend the representation so a final linear
            output separates it. This is a minimal demonstration of representation learning—not a
            claim that every dataset needs a deep model.
            """
        ),
        code(
            """
            xor_x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
            xor_y = torch.tensor([0., 1., 1., 0.])
            torch.manual_seed(1)
            xor_model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
            optimizer = torch.optim.Adam(xor_model.parameters(), lr=0.05)
            for _ in range(500):
                logits = xor_model(xor_x).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, xor_y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            predictions = (xor_model(xor_x).squeeze(-1) > 0).float()
            print("XOR predictions:", predictions.tolist(), "loss:", loss.item())
            assert torch.equal(predictions, xor_y)
            """
        ),
    ]
    write(
        "22_neurons_activations_mlps.ipynb",
        lesson(
            number=22,
            title="Neurons, Activations, and MLPs",
            coverage="V2 3.1–3.2",
            why="The Transformer feed-forward sublayer is an MLP. Understanding affine maps, nonlinear gates, saturation, and hidden width is essential before assembling deeper networks.",
            objectives=[
                "Implement stable common activation functions and inspect their derivatives.",
                "Explain why stacked affine layers need nonlinearity.",
                "Implement a two-layer MLP with explicit shape annotations.",
                "Use a hidden representation to solve a nonlinearly separable task.",
            ],
            cells=cells,
            failures=[
                "Saturated activation: useful gradients approach zero.",
                "Dead ReLU units: preactivations remain negative and receive no gradient.",
                "Missing nonlinearity: multiple layers collapse algebraically to one affine map.",
                "Output/target mismatch: the final activation and loss duplicate or omit needed transformations.",
            ],
            exercises=[
                "Implement GELU's exact formula and compare it numerically with PyTorch.",
                "Train XOR with no activation and document the irreducible failure.",
                "Count MLP parameters for widths C and 4C, including biases, and apply it to C=768.",
            ],
            exit_condition="you can implement an MLP forward pass, derive its shapes and parameter count, and explain the role of each activation.",
            next_lesson="23 — Scalar autograd.",
        ),
    )


def build_nn_scalar_autograd() -> None:
    cells = [
        md(
            """
            ## 1. Build a scalar reverse-mode autograd engine

            Each `Value` stores a scalar result, accumulated gradient, parent nodes, operation label,
            and a local backward closure. Operators construct a directed acyclic graph during the
            forward pass. Backward sorts parents before children, then visits that order in reverse.
            """
        ),
        code(
            """
            import math
            import torch

            class Value:
                def __init__(self, data, children=(), op="", label=""):
                    self.data = float(data)
                    self.grad = 0.0
                    self._prev = set(children)
                    self._op = op
                    self.label = label
                    self._backward = lambda: None

                def __repr__(self):
                    return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

                def __add__(self, other):
                    other = other if isinstance(other, Value) else Value(other)
                    out = Value(self.data + other.data, (self, other), "+")
                    def _backward():
                        self.grad += out.grad
                        other.grad += out.grad
                    out._backward = _backward
                    return out

                __radd__ = __add__

                def __mul__(self, other):
                    other = other if isinstance(other, Value) else Value(other)
                    out = Value(self.data * other.data, (self, other), "*")
                    def _backward():
                        self.grad += other.data * out.grad
                        other.grad += self.data * out.grad
                    out._backward = _backward
                    return out

                __rmul__ = __mul__

                def __pow__(self, exponent):
                    assert isinstance(exponent, (int, float))
                    out = Value(self.data**exponent, (self,), f"**{exponent}")
                    def _backward():
                        self.grad += exponent * self.data ** (exponent - 1) * out.grad
                    out._backward = _backward
                    return out

                def __neg__(self): return self * -1
                def __sub__(self, other): return self + (-other)
                def __rsub__(self, other): return other + (-self)
                def __truediv__(self, other): return self * other**-1

                def tanh(self):
                    value = math.tanh(self.data)
                    out = Value(value, (self,), "tanh")
                    def _backward():
                        self.grad += (1 - value**2) * out.grad
                    out._backward = _backward
                    return out

                def exp(self):
                    value = math.exp(self.data)
                    out = Value(value, (self,), "exp")
                    def _backward():
                        self.grad += value * out.grad
                    out._backward = _backward
                    return out

                def backward(self):
                    order, visited = [], set()
                    def visit(node):
                        if node not in visited:
                            visited.add(node)
                            for parent in node._prev:
                                visit(parent)
                            order.append(node)
                    visit(self)
                    self.grad = 1.0
                    for node in reversed(order):
                        node._backward()
            """
        ),
        md(
            """
            ## 2. Branching requires `+=`

            The same node can appear through multiple paths. Every local backward function must
            accumulate rather than assign. The topological traversal ensures a node receives all
            downstream contributions before it propagates them further.
            """
        ),
        code(
            """
            x = Value(3.0, label="x")
            y = x * x + 2 * x + 1
            y.backward()
            print("y:", y, "x:", x)
            assert y.data == 16.0 and x.grad == 8.0  # derivative 2x + 2
            """
        ),
        md(
            """
            ## 3. A tiny neuron and squared loss

            The engine is enough to construct a neuron, nonlinear activation, loss, and parameter
            update. Real frameworks add tensors, broadcasting, efficient native kernels, graph
            lifetime management, mixed precision, and many operations; the reverse-mode idea stays.
            """
        ),
        code(
            """
            inputs = [Value(2.0), Value(-1.0)]
            weights = [Value(0.5, label="w0"), Value(-1.0, label="w1")]
            bias = Value(0.25, label="b")
            target = 0.5

            preactivation = sum((weight * value for weight, value in zip(weights, inputs)), bias)
            prediction = preactivation.tanh()
            loss = (prediction - target) ** 2
            loss.backward()
            print("prediction:", prediction, "loss:", loss)
            print("parameter gradients:", [weight.grad for weight in weights], bias.grad)
            """
        ),
        md(
            """
            ## 4. Verify every leaf against PyTorch

            Independent implementations should agree in float64. This comparison catches local
            derivative, graph ordering, and accumulation bugs more effectively than checking that
            loss merely decreases once.
            """
        ),
        code(
            """
            tx = torch.tensor([2.0, -1.0], dtype=torch.float64)
            tw = torch.tensor([0.5, -1.0], dtype=torch.float64, requires_grad=True)
            tb = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
            tp = torch.tanh((tx * tw).sum() + tb)
            tloss = (tp - target) ** 2
            tloss.backward()

            assert math.isclose(loss.data, tloss.item(), rel_tol=1e-12)
            for manual, reference in zip(weights, tw.grad):
                assert math.isclose(manual.grad, reference.item(), rel_tol=1e-12)
            assert math.isclose(bias.grad, tb.grad.item(), rel_tol=1e-12)
            print("forward values and all parameter gradients match PyTorch")
            """
        ),
    ]
    write(
        "23_scalar_autograd.ipynb",
        lesson(
            number=23,
            title="Build a Scalar Autograd Engine",
            coverage="V2 3.3–3.4",
            why="Implementing reverse-mode autodiff removes the magic from `.backward()`. The important design ideas—dynamic graphs, topological order, local rules, and accumulation—carry directly to tensor frameworks.",
            objectives=[
                "Represent scalar operations as a dynamic computation graph.",
                "Topologically order that graph for reverse traversal.",
                "Implement local backward rules with gradient accumulation.",
                "Verify a nonlinear neuron against PyTorch autograd.",
            ],
            cells=cells,
            failures=[
                "Gradient assignment instead of accumulation: branch contributions disappear.",
                "Forward-order backward pass: parents propagate before receiving all contributions.",
                "Unseeded output: every gradient remains zero.",
                "Stale graph gradients: repeated backward calls accumulate unless deliberately cleared.",
            ],
            exercises=[
                "Add `log`, `relu`, and sigmoid operations with local derivative tests.",
                "Implement a method that clears gradients for all nodes reachable from an output.",
                "Train the tiny neuron for 20 steps and compare the loss trajectory with PyTorch.",
            ],
            exit_condition="you can implement a new differentiable scalar operation, state its local rule, and verify branched gradients against PyTorch.",
            next_lesson="24 — Tensor autograd and gradient checking.",
        ),
    )


def build_nn_tensor_autograd() -> None:
    cells = [
        md(
            r"""
            ## 1. Tensor reverse mode is scalar reverse mode plus shape rules

            For $Y=XW+b$ and scalar loss $L$, let upstream $G=\partial L/\partial Y$:

            $$\frac{\partial L}{\partial X}=GW^\top,\qquad
            \frac{\partial L}{\partial W}=X^\top G,\qquad
            \frac{\partial L}{\partial b}=\sum_{\text{broadcast axes}}G.$$

            The matrix formulas depend on the chosen orientation. Shape annotations are part of
            the derivation, not decoration.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            rng = np.random.default_rng(42)
            B, I, O = 4, 3, 2
            X = rng.normal(size=(B, I))       # [B,I]
            W = rng.normal(size=(I, O))       # [I,O]
            b = rng.normal(size=(O,))         # [O]
            target = rng.normal(size=(B, O))  # [B,O]

            Y = X @ W + b
            error = Y - target
            loss = np.mean(error**2)
            dY = 2 * error / error.size
            dX = dY @ W.T
            dW = X.T @ dY
            db = dY.sum(axis=0)
            print("loss:", loss, "gradient shapes:", dX.shape, dW.shape, db.shape)
            """
        ),
        md(
            """
            ## 2. Broadcasting backward means summing expanded axes

            If `[O]` bias broadcasts to `[B,O]`, each bias value influences all `B` rows, so its
            gradient sums the batch axis. A general `unbroadcast` removes extra leading axes and
            sums axes whose original dimension was 1.
            """
        ),
        code(
            """
            def unbroadcast(gradient: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
                while gradient.ndim > len(original_shape):
                    gradient = gradient.sum(axis=0)
                for axis, size in enumerate(original_shape):
                    if size == 1 and gradient.shape[axis] != 1:
                        gradient = gradient.sum(axis=axis, keepdims=True)
                return gradient

            upstream = np.ones((2, 5, 3))
            assert unbroadcast(upstream, (3,)).shape == (3,)
            assert np.all(unbroadcast(upstream, (3,)) == 10)
            assert unbroadcast(upstream, (1, 5, 1)).shape == (1, 5, 1)
            print("unbroadcast invariants passed")
            """
        ),
        md(
            r"""
            ## 3. Finite-difference gradient checking

            For each parameter element, perturb by $\pm\epsilon$ and compare the central difference
            with the analytical gradient. Use float64, a deterministic function, and relative error:

            $$\frac{|g_a-g_n|}{\max(1,|g_a|,|g_n|)}.$$

            Nondifferentiable points such as ReLU at zero require care because a numerical secant
            need not match the framework's chosen subgradient.
            """
        ),
        code(
            """
            def finite_difference(array: np.ndarray, loss_fn, epsilon: float = 1e-6) -> np.ndarray:
                estimate = np.zeros_like(array)
                for index in np.ndindex(array.shape):
                    old = array[index]
                    array[index] = old + epsilon
                    plus = loss_fn()
                    array[index] = old - epsilon
                    minus = loss_fn()
                    array[index] = old
                    estimate[index] = (plus - minus) / (2 * epsilon)
                return estimate

            numeric_dW = finite_difference(W, lambda: np.mean((X @ W + b - target) ** 2))
            relative_error = np.max(np.abs(dW - numeric_dW) / np.maximum(1, np.maximum(np.abs(dW), np.abs(numeric_dW))))
            print("max relative error:", relative_error)
            assert relative_error < 1e-8
            """
        ),
        md(
            """
            ## 4. PyTorch and `gradcheck`

            PyTorch's `gradcheck` applies finite differences to a function and compares them with
            autograd. It is designed for small float64 inputs. Passing a training run does not prove
            every gradient is correct; direct local checks provide much stronger evidence.
            """
        ),
        code(
            """
            tX = torch.tensor(X, dtype=torch.float64, requires_grad=True)
            tW = torch.tensor(W, dtype=torch.float64, requires_grad=True)
            tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
            ttarget = torch.tensor(target, dtype=torch.float64)
            tloss = ((tX @ tW + tb - ttarget) ** 2).mean()
            tloss.backward()

            np.testing.assert_allclose(tX.grad.numpy(), dX, rtol=1e-10)
            np.testing.assert_allclose(tW.grad.numpy(), dW, rtol=1e-10)
            np.testing.assert_allclose(tb.grad.numpy(), db, rtol=1e-10)

            def affine_loss(x, weight, bias):
                return ((x @ weight + bias - ttarget) ** 2).mean()

            assert torch.autograd.gradcheck(affine_loss, (tX.detach().requires_grad_(),
                                                          tW.detach().requires_grad_(),
                                                          tb.detach().requires_grad_()))
            print("manual, finite-difference, and PyTorch gradients agree")
            """
        ),
    ]
    write(
        "24_tensor_autograd_gradcheck.ipynb",
        lesson(
            number=24,
            title="Tensor Autograd and Gradient Checking",
            coverage="V2 3.5, 3.11",
            why="Tensor derivatives introduce matrix orientation, reductions, and broadcasting. Gradient checks turn those implementation details into testable numerical contracts.",
            objectives=[
                "Derive tensor gradients for an affine layer and mean-squared error.",
                "Reduce gradients correctly through broadcast operations.",
                "Implement elementwise central finite-difference checks.",
                "Compare manual, numerical, and PyTorch gradients in float64.",
            ],
            cells=cells,
            failures=[
                "Missing mean factor: gradients are scaled by batch or element count.",
                "Broadcast gradient has output shape: parameter updates cannot match the original parameter.",
                "Float32 gradient check: rounding hides or invents discrepancies.",
                "Nondifferentiable test point: valid subgradient conventions disagree with a symmetric finite difference.",
            ],
            exercises=[
                "Derive and check gradients for `tanh(X @ W + b)` followed by a sum.",
                "Extend `unbroadcast` tests to scalar, leading-axis, and multiple-singleton cases.",
                "Introduce one deliberate transpose bug and explain which gradient check catches it first.",
            ],
            exit_condition="you can derive each tensor gradient by shape, reduce broadcasts correctly, and make three independent implementations agree.",
            next_lesson="25 — Modules and the training loop.",
        ),
    )


def build_nn_training_loop() -> None:
    cells = [
        md(
            """
            ## 1. A reusable module owns parameters and computation

            `nn.Module` registers submodules and parameters, controls train/eval behavior, moves
            state between devices, and produces a `state_dict`. The forward method defines
            computation; the optimizer owns update state. Keep data iteration, model definition,
            loss, optimization, and evaluation as separate responsibilities.
            """
        ),
        code(
            """
            import io
            import random
            import numpy as np
            import torch
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset

            random.seed(42); np.random.seed(42); torch.manual_seed(42)

            class Classifier(torch.nn.Module):
                def __init__(self, input_dim: int, hidden_dim: int) -> None:
                    super().__init__()
                    self.layers = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(hidden_dim, 2),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.layers(x)

            model = Classifier(2, 16)
            print(model)
            print("parameters:", sum(p.numel() for p in model.parameters()))
            """
        ),
        md(
            """
            ## 2. Minibatches estimate the dataset gradient

            Shuffling changes grouping each epoch. A dedicated generator makes order reproducible.
            The final batch may be smaller unless `drop_last=True`; model code should not assume a
            fixed batch dimension. For language models, batches also group token sequences.
            """
        ),
        code(
            """
            generator = torch.Generator().manual_seed(42)
            points = torch.randn(400, 2, generator=generator)
            labels = ((points[:, 0] ** 2 + points[:, 1] ** 2) > 1.0).long()
            train_x, val_x = points[:320], points[320:]
            train_y, val_y = labels[:320], labels[320:]
            loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32,
                                shuffle=True, generator=torch.Generator().manual_seed(7))
            first_x, first_y = next(iter(loader))
            assert first_x.shape == (32, 2) and first_y.shape == (32,)
            """
        ),
        md(
            """
            ## 3. The canonical training step

            1. switch to training mode;
            2. compute predictions and scalar loss;
            3. clear old gradients;
            4. backpropagate;
            5. optionally inspect or clip gradients;
            6. update parameters.

            Clearing before or after the step can both work if done consistently. `set_to_none=True`
            avoids unnecessary zero fills and makes missing gradients easier to notice.
            """
        ),
        code(
            """
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
            history = []
            for epoch in range(25):
                model.train()
                total_loss, total_items = 0.0, 0
                for batch_x, batch_y in loader:
                    logits = model(batch_x)
                    loss = F.cross_entropy(logits, batch_y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    assert torch.isfinite(gradient_norm)
                    optimizer.step()
                    total_loss += loss.item() * batch_x.shape[0]
                    total_items += batch_x.shape[0]
                history.append(total_loss / total_items)

            print("training loss:", history[0], "→", history[-1])
            assert history[-1] < history[0]
            """
        ),
        md(
            """
            ## 4. Evaluation is a distinct, side-effect-free phase

            `model.eval()` disables dropout randomness and changes batch-normalization behavior.
            `torch.inference_mode()` avoids gradient recording and additional autograd overhead.
            Weighting batch metrics by batch size prevents a small final batch from counting equally.
            """
        ),
        code(
            """
            def evaluate(module, features, targets):
                module.eval()
                with torch.inference_mode():
                    logits = module(features)
                    loss = F.cross_entropy(logits, targets)
                    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
                return loss.item(), accuracy.item()

            val_loss, val_accuracy = evaluate(model, val_x, val_y)
            print(f"validation loss={val_loss:.3f}, accuracy={val_accuracy:.3f}")
            assert val_accuracy > 0.85
            """
        ),
        md(
            """
            ## 5. A checkpoint is more than weights

            Exact resume needs model state, optimizer state, current step/epoch, configuration,
            scheduler/scaler state when used, and random-number-generator states. Production
            experiments also record Git commit, dataset identity, device, metrics, and launch command.
            """
        ),
        code(
            """
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": 24,
                "config": {"input_dim": 2, "hidden_dim": 16, "seed": 42},
                "torch_rng_state": torch.get_rng_state(),
            }
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            restored = torch.load(buffer, weights_only=False)
            assert restored["epoch"] == 24 and restored["config"]["seed"] == 42
            print("in-memory checkpoint bytes:", buffer.getbuffer().nbytes)
            """
        ),
    ]
    write(
        "25_modules_training_loop.ipynb",
        lesson(
            number=25,
            title="Modules and the Training Loop",
            coverage="V2 3.6–3.7",
            why="Correct experiments require more than a forward formula. Modules, batching, mode changes, gradient lifecycle, evaluation, and checkpoints form the reusable skeleton of every later training run.",
            objectives=[
                "Define a parameter-owning PyTorch module.",
                "Build reproducible minibatches with variable final batch size.",
                "Implement a complete gradient-based training step.",
                "Evaluate without training side effects and serialize resume state.",
            ],
            cells=cells,
            failures=[
                "Gradients never cleared: every batch changes the effective update.",
                "Eval mode omitted: dropout makes metrics stochastic and batch norm mutates/uses the wrong state.",
                "Loss averaged by batches: a small last batch receives too much weight.",
                "Weights-only checkpoint: optimizer trajectory and exact resume are lost.",
            ],
            exercises=[
                "Refactor the loop into `train_epoch` and test that every parameter receives a finite gradient.",
                "Resume from the in-memory checkpoint into a new model and verify identical evaluation logits.",
                "Record a minimal experiment manifest with commit, seed, data, config, metric, device, and command.",
            ],
            exit_condition="you can train, evaluate, checkpoint, and exactly describe the mutable state involved in an experiment.",
            next_lesson="26 — Initialization and gradient flow.",
        ),
    )


def build_nn_initialization() -> None:
    cells = [
        md(
            r"""
            ## 1. Initialization controls signal scale before learning

            For independent zero-mean inputs and weights in $y_j=\sum_i w_{ji}x_i$,
            $\operatorname{Var}(y_j)\approx fan_{in}\operatorname{Var}(w)\operatorname{Var}(x)$.
            Choosing weight variance near $1/fan_{in}$ preserves forward variance for linear/tanh-like
            regimes (Xavier). ReLU discards roughly half the signal, motivating variance near
            $2/fan_{in}$ (He/Kaiming).
            """
        ),
        code(
            """
            import math
            import torch
            import torch.nn.functional as F

            torch.manual_seed(42)
            width, depth, batch = 256, 30, 512
            initial = torch.randn(batch, width)

            def propagate(std: float, activation) -> list[float]:
                x = initial.clone()
                variances = [x.var().item()]
                generator = torch.Generator().manual_seed(7)
                for _ in range(depth):
                    weight = torch.randn(width, width, generator=generator) * std
                    x = activation(x @ weight.T)
                    variances.append(x.var().item())
                return variances

            schemes = {
                "too small": propagate(0.01, torch.tanh),
                "too large": propagate(0.3, torch.tanh),
                "xavier": propagate(math.sqrt(1 / width), torch.tanh),
                "he + relu": propagate(math.sqrt(2 / width), F.relu),
            }
            for name, values in schemes.items():
                print(f"{name:10}: first={values[0]:.3f}, layer 10={values[10]:.3g}, final={values[-1]:.3g}")
            """
        ),
        md(
            """
            ## 2. Symmetry must be broken

            If neurons in one layer start with identical weights and biases, they compute identical
            outputs, receive identical gradients, and remain duplicates. Random initialization breaks
            this symmetry. Zero biases are usually fine because random weights already distinguish units.
            """
        ),
        code(
            """
            x = torch.randn(8, 4)
            shared_row = torch.randn(1, 4)
            symmetric_weight = shared_row.repeat(3, 1).requires_grad_()
            output = torch.tanh(x @ symmetric_weight.T)
            loss = output.square().mean()
            loss.backward()
            assert torch.allclose(symmetric_weight.grad[0], symmetric_weight.grad[1])
            print("identical neurons receive identical gradients:", symmetric_weight.grad[0])
            """
        ),
        md(
            r"""
            ## 3. Backward variance matters too

            A deep product of Jacobians can shrink or explode. Activation derivatives, weight scale,
            normalization, and residual paths all affect gradient flow. Inspect distributions across
            depth—not only the final loss.
            """
        ),
        code(
            """
            def gradient_profile(init: str) -> list[float]:
                torch.manual_seed(3)
                layers = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(20)])
                for layer in layers:
                    if init == "small":
                        torch.nn.init.normal_(layer.weight, std=0.01)
                    elif init == "he":
                        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    torch.nn.init.zeros_(layer.bias)
                x = torch.randn(32, 64)
                for layer in layers:
                    x = F.relu(layer(x))
                    x.retain_grad()
                x.square().mean().backward()
                return [layer.weight.grad.norm().item() for layer in layers]

            for init in ("small", "he"):
                norms = gradient_profile(init)
                print(init, "first/middle/last weight grad norms:", norms[0], norms[10], norms[-1])
            """
        ),
        md(
            """
            ## 4. Framework initialization and measurement

            Inspect actual initialized statistics rather than relying only on the function name.
            Fan mode, nonlinearity gain, truncation, bias policy, and parameter orientation change
            the expected scale.
            """
        ),
        code(
            """
            layer = torch.nn.Linear(128, 512, bias=False)
            torch.nn.init.xavier_normal_(layer.weight)
            expected_std = math.sqrt(2 / (128 + 512))
            measured_std = layer.weight.std().item()
            print("expected std:", expected_std, "measured std:", measured_std)
            assert abs(measured_std - expected_std) / expected_std < 0.1
            """
        ),
    ]
    write(
        "26_initialization_gradient_flow.ipynb",
        lesson(
            number=26,
            title="Initialization and Gradient Flow",
            coverage="V2 2.13, 3.8",
            why="A network can be mathematically valid yet untrainable because signals vanish, explode, saturate, or remain symmetric before optimization has a chance to help.",
            objectives=[
                "Derive the fan-in variance heuristic.",
                "Compare bad, Xavier, and He initialization across depth.",
                "Explain and demonstrate symmetry breaking.",
                "Measure activation and gradient statistics layer by layer.",
            ],
            cells=cells,
            failures=[
                "Vanishing activations/gradients: deeper layers receive almost no usable signal.",
                "Exploding scale: activations, loss, or gradients become huge or non-finite.",
                "Symmetric neurons: capacity is wasted on permanently identical units.",
                "Wrong fan convention: parameter orientation makes the selected scale inappropriate.",
            ],
            exercises=[
                "Add sigmoid to the depth experiment and measure saturation frequency.",
                "Compare Xavier uniform with Xavier normal across ten seeds.",
                "Record activation mean, standard deviation, and zero fraction for every ReLU layer.",
            ],
            exit_condition="you can choose and justify an initializer, then verify its forward and backward statistics empirically.",
            next_lesson="27 — Stable depth and generalization.",
        ),
    )


def build_nn_stable_depth() -> None:
    cells = [
        md(
            r"""
            ## 1. Normalization standardizes a representation

            LayerNorm operates independently at each token across its feature dimension:

            $$\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\hat x+\beta.$$

            It does not mix batch items or time steps. RMSNorm removes mean subtraction and normalizes
            root mean square. Both learn a scale; LayerNorm also learns a shift.
            """
        ),
        code(
            """
            import copy
            import torch
            import torch.nn.functional as F

            torch.manual_seed(42)
            x = torch.randn(2, 3, 8) * 4 + 7  # [B,T,C]

            def manual_layer_norm(value, eps=1e-5):
                mean = value.mean(dim=-1, keepdim=True)
                variance = value.var(dim=-1, unbiased=False, keepdim=True)
                return (value - mean) / torch.sqrt(variance + eps)

            normalized = manual_layer_norm(x)
            reference = F.layer_norm(x, (x.shape[-1],))
            torch.testing.assert_close(normalized, reference)
            torch.testing.assert_close(normalized.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0)
            print("per-token mean and variance normalized")
            """
        ),
        md(
            """
            ## 2. Residual paths create an identity route

            A residual block computes `x + F(x)`. Its Jacobian is `I + J_F`, so gradients have a
            direct identity contribution. Residuals do not guarantee perfect optimization, but they
            make learning incremental corrections easier. Transformer pre-norm blocks normalize
            before each sublayer and add its result to the residual stream.
            """
        ),
        code(
            """
            def input_gradient(depth: int, residual: bool) -> float:
                torch.manual_seed(0)
                value = torch.randn(16, 32, requires_grad=True)
                original = value
                for _ in range(depth):
                    weight = torch.randn(32, 32) * 0.03
                    transformed = torch.tanh(value @ weight)
                    value = value + transformed if residual else transformed
                value.square().mean().backward()
                return original.grad.norm().item()

            for residual in (False, True):
                print("residual=", residual, "input grad norm=", input_gradient(20, residual))
            """
        ),
        md(
            """
            ## 3. Generalization tools solve different problems

            - **Weight decay** prefers smaller parameter values.
            - **Dropout** injects multiplicative noise during training and rescales survivors.
            - **Early stopping** selects a checkpoint before validation performance degrades.
            - **More/better data** changes the evidence and is often the strongest intervention.

            These tools do not repair leakage, mislabeled data, an inappropriate metric, or an
            optimization failure.
            """
        ),
        code(
            """
            dropout = torch.nn.Dropout(p=0.5)
            ones = torch.ones(20_000)
            dropout.train()
            train_output = dropout(ones)
            dropout.eval()
            eval_output = dropout(ones)
            print("train mean:", train_output.mean().item(), "zero fraction:", (train_output == 0).float().mean().item())
            print("eval unique:", eval_output.unique().tolist())
            assert abs(train_output.mean().item() - 1.0) < 0.05
            assert torch.equal(eval_output, ones)
            """
        ),
        md(
            """
            ## 4. Diagnose with train/validation curves

            Underfitting: both losses remain high. Overfitting: training improves while validation
            worsens. Optimization instability: both curves spike or become non-finite. Distribution
            shift: validation may stay persistently worse even when regularization is strong.
            """
        ),
        code(
            """
            def diagnose(train_loss, val_loss):
                if not (torch.isfinite(torch.tensor(train_loss)).all() and torch.isfinite(torch.tensor(val_loss)).all()):
                    return "numerical/optimization instability"
                if train_loss[-1] > 0.8 and val_loss[-1] > 0.8:
                    return "underfitting or optimization failure"
                if train_loss[-1] < train_loss[0] and val_loss[-1] > min(val_loss):
                    return "overfitting after best checkpoint"
                return "no obvious pathology from loss curves alone"

            examples = {
                "underfit": ([1.2, 1.1, 1.0], [1.3, 1.2, 1.1]),
                "overfit": ([1.0, 0.5, 0.1], [1.1, 0.7, 0.9]),
                "stable": ([1.0, 0.6, 0.4], [1.1, 0.7, 0.5]),
            }
            for name, curves in examples.items():
                print(name, "→", diagnose(*curves))
            """
        ),
    ]
    write(
        "27_residuals_normalization_regularization.ipynb",
        lesson(
            number=27,
            title="Residuals, Normalization, and Regularization",
            coverage="V2 2.14, 3.9–3.10",
            why="Normalization and residuals make deep optimization practical; regularization and honest validation determine whether fitted behavior survives beyond the training examples.",
            objectives=[
                "Implement LayerNorm over the correct feature axis.",
                "Explain residual gradient flow through an identity path.",
                "Distinguish dropout, weight decay, and early stopping.",
                "Diagnose underfitting, overfitting, and instability from evidence.",
            ],
            cells=cells,
            failures=[
                "Normalization over batch/time: examples or tokens leak into each other's statistics.",
                "Residual shape mismatch: the identity and update paths cannot be added.",
                "Dropout active during evaluation: metrics and generation remain stochastic for the wrong reason.",
                "Regularizing an optimization bug: lower capacity cannot fix broken gradients or learning rates.",
            ],
            exercises=[
                "Implement RMSNorm and compare its output statistics and parameter count with LayerNorm.",
                "Measure input-gradient norms across depth with plain, residual, and pre-norm residual blocks.",
                "Given three pairs of learning curves, propose one diagnostic experiment before selecting a fix.",
            ],
            exit_condition="you can explain how depth remains trainable and choose a generalization intervention based on train/validation evidence.",
            next_lesson="28 — Text, Unicode, characters, words, and bytes.",
        ),
    )


def build_text_unicode() -> None:
    cells = [
        md(
            """
            ## 1. Text is Unicode, storage is bytes, models consume IDs

            A Python string is a sequence of Unicode code points. UTF-8 encodes those code points
            into one to four bytes. What a human sees as one grapheme can contain multiple code
            points (base character, combining mark, emoji joiner sequence). Tokenization must define
            exactly which representation it accepts and preserve a reliable decode path.
            """
        ),
        code(
            """
            import re
            import unicodedata

            examples = ["café", "cafe\u0301", "😂", "👩‍💻", "東京"]
            for text in examples:
                print(repr(text), "code points=", len(text), "utf8 bytes=", len(text.encode("utf-8")),
                      "hex=", text.encode("utf-8").hex())
            assert examples[0] != examples[1]
            assert unicodedata.normalize("NFC", examples[0]) == unicodedata.normalize("NFC", examples[1])
            """
        ),
        md(
            """
            ## 2. Normalization is a product decision

            NFC composes canonical sequences; NFD decomposes them. NFKC/NFKD additionally collapse
            compatibility distinctions and can change meaning or style. Lowercasing, whitespace
            normalization, and punctuation replacement likewise trade vocabulary size against lost
            information. Store the exact policy with the tokenizer.
            """
        ),
        code(
            """
            samples = ["ＡＩ", "①", "Straße", "  timing…  matters  "]
            for text in samples:
                print("original:", repr(text), "NFC:", repr(unicodedata.normalize("NFC", text)),
                      "NFKC:", repr(unicodedata.normalize("NFKC", text)))

            def conservative_normalize(text: str) -> str:
                return unicodedata.normalize("NFC", text).replace("\\r\\n", "\\n")

            assert conservative_normalize("cafe\u0301") == "café"
            """
        ),
        md(
            """
            ## 3. Character, word, and byte vocabularies

            Character tokens are simple but Unicode makes coverage large and graphemes complicated.
            Word tokens are interpretable but explode vocabulary and cannot naturally represent new
            words. Bytes guarantee coverage with at most 256 base values, but sequences are longer
            and individual bytes may not decode independently.
            """
        ),
        code(
            """
            corpus = ["jokes need timing", "timing needs pauses", "😂 needs context"]

            characters = sorted(set("".join(corpus)))
            words = sorted(set(token for line in corpus for token in re.findall(r"\\w+|[^\\w\\s]", line)))
            byte_values = sorted(set("\\n".join(corpus).encode("utf-8")))
            print("character vocab/length:", len(characters), sum(map(len, corpus)))
            print("word vocab/length:", len(words), sum(len(re.findall(r"\\w+|[^\\w\\s]", line)) for line in corpus))
            print("observed byte vocab/length:", len(byte_values), len("\\n".join(corpus).encode("utf-8")))
            """
        ),
        md(
            """
            ## 4. Round-trip invariants

            For supported input, `decode(encode(text))` should equal the tokenizer's documented
            normalized form. Unknown-token fallback is lossy; byte fallback preserves coverage.
            Test empty strings, every special token, multilingual text, combining marks, newlines,
            and malformed byte sequences according to explicit policy.
            """
        ),
        code(
            """
            def byte_encode(text: str) -> list[int]:
                return list(conservative_normalize(text).encode("utf-8"))

            def byte_decode(ids: list[int]) -> str:
                if any(not 0 <= value <= 255 for value in ids):
                    raise ValueError("byte token outside [0, 255]")
                return bytes(ids).decode("utf-8", errors="strict")

            tests = ["", "hello", "cafe\u0301", "東京 😂\\nnext line", "👩‍💻"]
            for text in tests:
                expected = conservative_normalize(text)
                actual = byte_decode(byte_encode(text))
                assert actual == expected
            print("all byte round trips passed")
            """
        ),
    ]
    write(
        "28_text_unicode_bytes.ipynb",
        lesson(
            number=28,
            title="Text, Unicode, Characters, Words, and Bytes",
            coverage="V2 4.1–4.4",
            why="A tokenizer cannot be correct without an explicit text representation. Unicode normalization and vocabulary units determine reversibility, coverage, sequence length, and what distinctions the SLM can learn.",
            objectives=[
                "Distinguish graphemes, code points, encoded bytes, tokens, and token IDs.",
                "Compare canonical and compatibility normalization policies.",
                "Measure character, word, and byte vocabulary tradeoffs.",
                "Specify and test strict round-trip behavior.",
            ],
            cells=cells,
            failures=[
                "Normalization drift: training and inference transform equivalent text differently.",
                "Unknown-token loss: unsupported text cannot be reconstructed.",
                "Byte boundary confusion: arbitrary token slices decode as invalid UTF-8.",
                "Grapheme assumption: code-point length is presented as user-visible character length.",
            ],
            exercises=[
                "Inspect code points and bytes for five examples from languages you expect Humor Machine to handle.",
                "Write a normalization policy and list one useful distinction every transformation could erase.",
                "Create a round-trip test table containing empty, whitespace, special-looking, multilingual, and emoji text.",
            ],
            exit_condition="you can state exactly how raw text becomes bytes or symbols and prove supported text round-trips under a documented policy.",
            next_lesson="29 — Learned n-gram language models.",
        ),
    )


def build_text_ngram() -> None:
    cells = [
        md(
            r"""
            ## 1. Learn a bigram table from data

            A bigram model assumes the next token depends only on the current token:
            $p(x_t\mid x_{<t})\approx p(x_t\mid x_{t-1})$. Count observed pairs, add smoothing,
            and normalize each row. Unlike notebook 00, these parameters are estimated from text.
            """
        ),
        code(
            """
            import numpy as np

            corpus = ["timing is funny", "timing is everything", "funny timing wins", "timing wins"]
            words = sorted({word for line in corpus for word in line.split()})
            vocab = ["<BOS>", "<EOS>", *words]
            stoi = {token: index for index, token in enumerate(vocab)}
            itos = dict(enumerate(vocab))
            V = len(vocab)
            counts = np.zeros((V, V), dtype=np.int64)

            for line in corpus:
                sequence = ["<BOS>", *line.split(), "<EOS>"]
                for left, right in zip(sequence, sequence[1:]):
                    counts[stoi[left], stoi[right]] += 1
            print("vocabulary:", vocab)
            print("observed transitions:", int((counts > 0).sum()))
            """
        ),
        md(
            r"""
            ## 2. Smoothing reserves probability for unseen pairs

            Maximum-likelihood rows are count divided by row total, but unseen events get zero
            probability and therefore infinite evaluation loss. Add-$\alpha$ smoothing uses
            $(c_{ij}+\alpha)/(\sum_j c_{ij}+\alpha V)$. It is simple, not state of the art.
            """
        ),
        code(
            """
            def bigram_probabilities(count_matrix: np.ndarray, alpha: float) -> np.ndarray:
                smoothed = count_matrix.astype(np.float64) + alpha
                return smoothed / smoothed.sum(axis=1, keepdims=True)

            probabilities = bigram_probabilities(counts, alpha=0.1)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
            timing_row = probabilities[stoi["timing"]]
            print("p(next | timing):", {itos[i]: round(p, 3) for i, p in enumerate(timing_row) if p > 0.02})
            """
        ),
        md(
            """
            ## 3. Evaluate held-out negative log-likelihood

            Token NLL measures every predicted transition. Perplexity is its exponent. Split source
            documents before extracting n-grams; otherwise near-identical local transitions leak
            across splits. An n-gram baseline reveals whether a neural model actually learns longer
            dependencies rather than only local frequency.
            """
        ),
        code(
            """
            def sequence_nll(text: str, table: np.ndarray) -> float:
                sequence = ["<BOS>", *text.split(), "<EOS>"]
                log_probabilities = []
                for left, right in zip(sequence, sequence[1:]):
                    if left not in stoi or right not in stoi:
                        raise ValueError("held-out text contains out-of-vocabulary word")
                    log_probabilities.append(np.log(table[stoi[left], stoi[right]]))
                return float(-np.mean(log_probabilities))

            for text in ("timing is funny", "funny timing wins", "everything timing is"):
                nll = sequence_nll(text, probabilities)
                print(f"{text!r}: NLL={nll:.3f}, PPL={np.exp(nll):.2f}")
            """
        ),
        md(
            """
            ## 4. Autoregressive generation

            Generation uses the same row repeatedly, appending the sampled token until EOS or a
            length cap. A bigram cannot distinguish two contexts ending in the same token; this is
            its Markov limitation and the motivation for richer context representations.
            """
        ),
        code(
            """
            rng = np.random.default_rng(7)

            def generate(max_tokens: int = 12) -> str:
                current = "<BOS>"
                output = []
                for _ in range(max_tokens):
                    next_id = int(rng.choice(V, p=probabilities[stoi[current]]))
                    current = itos[next_id]
                    if current == "<EOS>":
                        break
                    if current != "<BOS>":
                        output.append(current)
                return " ".join(output)

            for _ in range(8):
                print(generate())
            """
        ),
    ]
    write(
        "29_ngram_language_model.ipynb",
        lesson(
            number=29,
            title="Learned N-Gram Language Models",
            coverage="Added foundation between V2 Parts IV and V",
            why="An n-gram model is the simplest learned next-token model. It exposes counting, smoothing, likelihood, generation, and context limits before neural-network complexity arrives.",
            objectives=[
                "Estimate bigram counts and conditional probabilities from documents.",
                "Use smoothing to handle unseen transitions.",
                "Evaluate mean NLL and perplexity on complete sequences.",
                "Generate autoregressively and state the model's context limitation.",
            ],
            cells=cells,
            failures=[
                "Zero-probability event: one unseen transition makes sequence NLL infinite.",
                "Row with no observations: unsmoothed normalization divides by zero.",
                "Document leakage: transitions from one source appear in both train and evaluation.",
                "Context overclaim: a bigram is credited with dependencies it cannot represent.",
            ],
            exercises=[
                "Implement a trigram model and define its fallback for an unseen two-token context.",
                "Sweep smoothing alpha and report train versus held-out NLL.",
                "Compare generated samples with shuffled unigram samples and explain the difference.",
            ],
            exit_condition="you can learn, smooth, evaluate, and sample an n-gram model while describing exactly what context it forgets.",
            next_lesson="30 — BPE, unigram tokenization, and vocabulary design.",
        ),
    )


def build_text_subwords() -> None:
    cells = [
        md(
            """
            ## 1. Subwords trade vocabulary size for sequence length

            Byte Pair Encoding begins with small symbols (here characters for readability), counts
            adjacent pairs, and repeatedly merges the most frequent pair. Frequent sequences become
            single tokens while rare words remain decomposable. Production tokenizers typically use
            byte-level bases to guarantee coverage.
            """
        ),
        code(
            """
            from collections import Counter

            word_frequencies = {
                "funny": 5,
                "funnier": 2,
                "funniest": 2,
                "timing": 4,
                "timed": 2,
            }
            vocabulary = {tuple(word) + ("</w>",): frequency for word, frequency in word_frequencies.items()}

            def pair_counts(vocab):
                counts = Counter()
                for symbols, frequency in vocab.items():
                    counts.update({pair: frequency for pair in zip(symbols, symbols[1:])})
                return counts

            print(pair_counts(vocabulary).most_common(8))
            """
        ),
        md(
            """
            ## 2. Learn and apply BPE merges

            Merge order is part of the tokenizer artifact. Encoding applies learned merges in rank
            order, not by greedily choosing any currently longest substring. Deterministic tie-breaking
            is necessary for reproducible vocabulary training.
            """
        ),
        code(
            """
            def merge_pair(symbols, pair):
                merged, index = [], 0
                while index < len(symbols):
                    if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                        merged.append(symbols[index] + symbols[index + 1])
                        index += 2
                    else:
                        merged.append(symbols[index])
                        index += 1
                return tuple(merged)

            merges = []
            for _ in range(10):
                counts = pair_counts(vocabulary)
                best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                merges.append(best)
                vocabulary = {merge_pair(symbols, best): frequency for symbols, frequency in vocabulary.items()}

            print("merge rules:", merges)
            print("learned word segmentations:", list(vocabulary))
            """
        ),
        code(
            """
            def bpe_encode_word(word: str, learned_merges) -> tuple[str, ...]:
                symbols = tuple(word) + ("</w>",)
                for pair in learned_merges:
                    symbols = merge_pair(symbols, pair)
                return symbols

            for word in ("funny", "funnier", "timing", "untimed"):
                print(word, "→", bpe_encode_word(word, merges))
            """
        ),
        md(
            """
            ## 3. Unigram language-model tokenization

            Unigram tokenization starts with a large candidate vocabulary and assigns each token a
            probability. Dynamic programming finds the segmentation with highest probability; low-
            utility candidates are pruned iteratively. Unlike BPE's deterministic merge history,
            unigram defines a probabilistic segmentation model and can sample segmentations during
            training. The small Viterbi example below illustrates inference, not full vocabulary EM.
            """
        ),
        code(
            """
            import math

            token_prob = {"f": .04, "u": .04, "n": .05, "y": .03,
                          "fu": .08, "fun": .20, "funny": .25, "ny": .07}

            def best_unigram_segmentation(text: str):
                best = [(-math.inf, []) for _ in range(len(text) + 1)]
                best[0] = (0.0, [])
                for end in range(1, len(text) + 1):
                    for start in range(end):
                        token = text[start:end]
                        if token in token_prob and best[start][0] > -math.inf:
                            score = best[start][0] + math.log(token_prob[token])
                            if score > best[end][0]:
                                best[end] = (score, [*best[start][1], token])
                if best[-1][0] == -math.inf:
                    raise ValueError("text is not covered by candidate vocabulary")
                return best[-1]

            print("unigram best:", best_unigram_segmentation("funny"))
            """
        ),
        md(
            """
            ## 4. Vocabulary evaluation

            Compare coverage, average tokens per character/word, vocabulary storage, training speed,
            and downstream quality. Larger vocabularies shorten sequences but enlarge embedding and
            LM-head matrices (`V × C`) and may give rare tokens poorly estimated embeddings. The
            tokenizer must be frozen and versioned with model checkpoints.
            """
        ),
        code(
            """
            evaluation_words = list(word_frequencies) + ["unfunny", "retiming"]
            lengths = [len(bpe_encode_word(word, merges)) for word in evaluation_words]
            character_lengths = [len(word) + 1 for word in evaluation_words]  # include boundary marker
            print("mean BPE symbols:", sum(lengths) / len(lengths))
            print("mean character symbols:", sum(character_lengths) / len(character_lengths))
            assert all(length <= chars for length, chars in zip(lengths, character_lengths))
            """
        ),
    ]
    write(
        "30_bpe_unigram_vocab.ipynb",
        lesson(
            number=30,
            title="BPE, Unigram Tokenization, and Vocabulary Design",
            coverage="V2 4.5–4.7",
            why="Subword design sets sequence lengths, model matrix sizes, multilingual coverage, and what patterns become atomic. A tokenizer is learned data infrastructure, not a harmless preprocessing detail.",
            objectives=[
                "Train and apply a small deterministic BPE merge list.",
                "Explain unigram tokenization and find a best segmentation with dynamic programming.",
                "Compare coverage, sequence length, and vocabulary-size tradeoffs.",
                "State why tokenizer versioning is inseparable from model weights.",
            ],
            cells=cells,
            failures=[
                "Different merge order: the same vocabulary strings produce different tokenizations.",
                "No base coverage: unseen text cannot be encoded losslessly.",
                "Vocabulary too large: embeddings dominate parameters and rare tokens are weakly trained.",
                "Tokenizer changed after training: token IDs no longer address the intended embeddings.",
            ],
            exercises=[
                "Run 0, 5, 10, and 20 BPE merges and plot vocabulary size against corpus token count.",
                "Add byte-level base symbols conceptually and explain how raw bytes would display and decode.",
                "Change unigram token probabilities until `funny` receives three different best segmentations.",
            ],
            exit_condition="you can train and apply a subword scheme, test its coverage, and justify vocabulary size with model and data costs.",
            next_lesson="31 — Special tokens, controls, and prompt formats.",
        ),
    )


def build_text_protocol() -> None:
    cells = [
        md(
            """
            ## 1. Special tokens are protocol symbols

            Common roles include BOS (start), EOS (stop), PAD (batch alignment), UNK (only when
            coverage is incomplete), separators, roles, and task/control tokens. Each must have one
            stable ID, documented insertion rules, and training examples that teach its meaning.
            Reserve them before ordinary vocabulary construction so IDs cannot collide.
            """
        ),
        code(
            """
            SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>",
                        "<STYLE_DRY>", "<STYLE_PUN>"]
            ordinary = ["tell", "me", "a", "joke", "timing", "matters", "."]
            vocabulary = [*SPECIALS, *ordinary]
            stoi = {token: index for index, token in enumerate(vocabulary)}
            itos = dict(enumerate(vocabulary))
            assert len(stoi) == len(vocabulary)
            print(stoi)
            """
        ),
        md(
            """
            ## 2. Prompt formatting is part of the training distribution

            The model sees tokens, not abstract chat messages. Training and inference must serialize
            roles, controls, separators, and EOS identically. A format should be unambiguous even when
            user text contains strings that visually resemble control tokens.
            """
        ),
        code(
            """
            def format_example(user_tokens: list[str], answer_tokens: list[str], style: str) -> list[str]:
                style_token = {"dry": "<STYLE_DRY>", "pun": "<STYLE_PUN>"}[style]
                return ["<BOS>", style_token, "<USER>", *user_tokens,
                        "<ASSISTANT>", *answer_tokens, "<EOS>"]

            example = format_example(["tell", "me", "a", "joke"], ["timing", "matters", "."], "dry")
            ids = [stoi[token] for token in example]
            decoded = [itos[index] for index in ids]
            assert decoded == example and ids[-1] == stoi["<EOS>"]
            print(example)
            print(ids)
            """
        ),
        md(
            """
            ## 3. Loss masks decide which tokens teach the model

            In instruction tuning, the input prompt often supplies context but only assistant tokens
            contribute to the supervised loss. A boolean loss mask must align exactly with shifted
            targets. Pretraining generally predicts every non-padding token; do not reuse one masking
            policy blindly across objectives.
            """
        ),
        code(
            """
            assistant_position = example.index("<ASSISTANT>")
            # Target at position t is example[t+1]. Include tokens after the assistant marker.
            inputs = ids[:-1]
            targets = ids[1:]
            loss_mask = [position >= assistant_position for position in range(len(targets))]
            supervised_targets = [itos[target] for target, include in zip(targets, loss_mask) if include]
            print("supervised targets:", supervised_targets)
            assert supervised_targets == ["timing", "matters", ".", "<EOS>"]
            """
        ),
        md(
            """
            ## 4. Literal-text collisions and escaping

            If raw user text can become the same ID as `<ASSISTANT>`, it can alter structure. Robust
            systems tokenize structured messages through an API rather than string concatenation,
            escape reserved sequences, or use special IDs that ordinary text encoding cannot emit.
            Test malformed role order, missing EOS, duplicate BOS, and controls absent from training.
            """
        ),
        code(
            """
            def validate_protocol(tokens: list[str]) -> None:
                assert tokens[0] == "<BOS>" and tokens[-1] == "<EOS>"
                assert tokens.count("<BOS>") == 1 and tokens.count("<EOS>") == 1
                assert tokens.count("<USER>") == 1 and tokens.count("<ASSISTANT>") == 1
                assert tokens.index("<USER>") < tokens.index("<ASSISTANT>")

            validate_protocol(example)
            malformed = ["<BOS>", "<ASSISTANT>", "joke", "<USER>", "me", "<EOS>"]
            try:
                validate_protocol(malformed)
            except AssertionError:
                print("malformed role order correctly rejected")
            """
        ),
    ]
    write(
        "31_special_tokens_padding_prompts.ipynb",
        lesson(
            number=31,
            title="Special Tokens, Padding, Controls, and Prompt Formats",
            coverage="V2 4.8–4.9, 4.13",
            why="BOS/EOS, roles, and style controls define a machine-readable protocol. If serialization or loss masking differs between training and inference, model behavior fails even when architecture and weights are correct.",
            objectives=[
                "Assign noncolliding stable IDs to protocol tokens.",
                "Serialize a structured prompt deterministically.",
                "Align input IDs, shifted targets, and assistant-only loss masks.",
                "Validate malformed formats and literal-text collision policies.",
            ],
            cells=cells,
            failures=[
                "Special/ordinary ID collision: decoded meaning depends on code path rather than the ID.",
                "Train/serve format drift: inference context is out of distribution.",
                "Off-by-one loss mask: prompt or marker tokens receive unintended supervision.",
                "Untrained control: a reserved token is expected to steer behavior without relevant examples.",
            ],
            exercises=[
                "Design a multi-turn prompt grammar and write validators for legal role order.",
                "Create input, target, attention, and loss masks for two padded instruction examples.",
                "Specify how ordinary user text containing `<EOS>` should encode without ending the sequence.",
            ],
            exit_condition="you can define, serialize, mask, round-trip, and validate the complete token protocol used by both training and inference.",
            next_lesson="32 — Context windows, masks, packing, and storage.",
        ),
    )


def build_text_sequences() -> None:
    cells = [
        md(
            r"""
            ## 1. Shift a token sequence into inputs and targets

            Given tokens $[x_0,x_1,\ldots,x_T]$, inputs are $[x_0,\ldots,x_{T-1}]$ and targets
            are $[x_1,\ldots,x_T]$. The model at position $t$ predicts target $x_{t+1}$. A context
            window of length $T$ therefore requires $T+1$ source tokens to make $T$ training pairs.
            """
        ),
        code(
            """
            import numpy as np
            import torch

            PAD, BOS, EOS = 0, 1, 2
            document = [BOS, 10, 11, 12, EOS]
            inputs = torch.tensor(document[:-1])
            targets = torch.tensor(document[1:])
            print("inputs: ", inputs.tolist())
            print("targets:", targets.tolist())
            assert torch.equal(inputs[1:], targets[:-1])
            """
        ),
        md(
            r"""
            ## 2. Causal and padding masks answer different questions

            A causal mask prevents query position $t$ from reading key positions $>t$. A padding
            mask prevents attention to nonexistent padded keys. A loss mask decides which target
            positions contribute to the objective. They have different shapes and must not be
            confused.

            Causal allowed mask: `allowed[t_query, t_key] = (t_key <= t_query)` with `[T,T]`.
            """
        ),
        code(
            """
            T = 5
            causal_allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))  # [Tq,Tk]
            padding_valid = torch.tensor([[True, True, True, True, False],
                                          [True, True, True, False, False]])  # [B,Tk]
            combined_allowed = causal_allowed[None, :, :] & padding_valid[:, None, :]  # [B,Tq,Tk]

            print("causal:\\n", causal_allowed.int())
            print("combined batch 1:\\n", combined_allowed[1].int())
            assert combined_allowed.shape == (2, T, T)
            assert not combined_allowed[:, 0, 1:].any()
            """
        ),
        md(
            """
            ## 3. Pack documents without cross-document leakage

            Concatenating documents improves utilization, but ordinary causal attention would let
            later documents read earlier documents. EOS boundaries may be acceptable for continuous
            pretraining when intentionally modeled; strict document isolation requires a block-
            diagonal attention mask or resetting sequence segments. The policy must be explicit.
            """
        ),
        code(
            """
            documents = [[BOS, 10, 11, EOS], [BOS, 20, 21, 22, EOS]]
            packed = [token for doc in documents for token in doc]
            segment_ids = [segment for segment, doc in enumerate(documents) for _ in doc]
            packed_t = torch.tensor(packed)
            segments_t = torch.tensor(segment_ids)
            length = len(packed)

            causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
            same_document = segments_t[:, None] == segments_t[None, :]
            isolated_attention = causal & same_document
            assert not isolated_attention[4:, :4].any()
            print("packed IDs:", packed)
            print("segment IDs:", segment_ids)
            """
        ),
        md(
            """
            ## 4. Fixed windows and boundary-safe labels

            For long streams, choose windows by a documented stride. Overlapping windows expose
            tokens multiple times; non-overlapping windows are cheaper but waste tails. If windows
            are independent, never create a target that belongs outside the retained source boundary.
            Pad short windows and set padded targets to an ignore index.
            """
        ),
        code(
            """
            def make_windows(tokens: list[int], context: int, stride: int, ignore_index: int = -100):
                examples = []
                for start in range(0, max(1, len(tokens) - 1), stride):
                    chunk = tokens[start:start + context + 1]
                    if len(chunk) < 2:
                        break
                    x, y = chunk[:-1], chunk[1:]
                    valid = len(x)
                    x = x + [PAD] * (context - valid)
                    y = y + [ignore_index] * (context - valid)
                    examples.append((x, y, valid))
                    if start + context + 1 >= len(tokens):
                        break
                return examples

            windows = make_windows(packed, context=5, stride=5)
            for x, y, valid in windows:
                print("x=", x, "y=", y, "valid=", valid)
                assert len(x) == len(y) == 5 and all(value == -100 for value in y[valid:])
            """
        ),
        md(
            """
            ## 5. Storage needs schema and provenance

            Token arrays may use compact integer dtypes when vocabulary size permits. Alongside IDs,
            store tokenizer version/hash, normalization policy, document boundaries, split identity,
            filtering/deduplication version, and checksums. Memory mapping avoids loading an entire
            corpus but does not replace shuffled, boundary-aware sampling.
            """
        ),
        code(
            """
            vocab_size = 50_000
            max_id = vocab_size - 1
            dtype = np.uint16 if max_id <= np.iinfo(np.uint16).max else np.uint32
            token_array = np.asarray(packed, dtype=dtype)
            metadata = {
                "dtype": str(token_array.dtype),
                "token_count": int(token_array.size),
                "vocab_size": vocab_size,
                "tokenizer_version": "lesson-demo-v1",
                "split": "train",
            }
            print(metadata, "bytes:", token_array.nbytes)
            assert token_array.max() < vocab_size
            """
        ),
    ]
    write(
        "32_context_windows_attention_masks_packing.ipynb",
        lesson(
            number=32,
            title="Context Windows, Masks, Packing, and Storage",
            coverage="V2 4.10–4.12",
            why="Correct token IDs can still create corrupt training data through off-by-one targets, future leakage, padding loss, or document boundary leakage. Sequence construction defines the actual learning task.",
            objectives=[
                "Create aligned next-token inputs and targets.",
                "Distinguish causal, padding, document, and loss masks.",
                "Pack documents with an explicit cross-document attention policy.",
                "Window and store token data with boundary and provenance metadata.",
            ],
            cells=cells,
            failures=[
                "Unshifted targets: the model learns to copy the current token.",
                "Future-visible attention: training loss is excellent but autoregressive generation fails.",
                "Padding loss: batch shape artifacts dominate the objective.",
                "Cross-document leakage: unrelated or evaluation content becomes accessible context.",
            ],
            exercises=[
                "Build a padded batch and produce attention, target, and loss masks with explicit shapes.",
                "Compare token utilization for padded, packed, and overlapping-window strategies.",
                "Write assertions proving no query can attend to a future or different-document key.",
            ],
            exit_condition="you can turn documents into fixed training tensors and prove there is no unintended future, padding, or document leakage.",
            next_lesson="33 — Embeddings, the residual stream, and position.",
        ),
    )


def build_transformer_embeddings() -> None:
    cells = [
        md(
            r"""
            ## 1. Embeddings are learned rows selected by token ID

            An embedding table $E\in\mathbb{R}^{V\times C}$ maps integer ID $i$ to row $E_i$.
            Lookup is equivalent to multiplying a one-hot vector by $E$, but avoids constructing
            `[B,T,V]` one-hot tensors. Repeated token IDs share one row and accumulate gradient into it.
            """
        ),
        code(
            """
            import math
            import torch
            import torch.nn.functional as F

            torch.manual_seed(42)
            B, T, V, C = 2, 5, 11, 8
            token_ids = torch.tensor([[1, 4, 4, 2, 0], [1, 7, 3, 2, 0]])  # [B,T]
            table = torch.randn(V, C, requires_grad=True)                   # [V,C]
            looked_up = table[token_ids]                                   # [B,T,C]
            one_hot = F.one_hot(token_ids, num_classes=V).float()           # [B,T,V]
            via_one_hot = one_hot @ table                                  # [B,T,C]
            torch.testing.assert_close(looked_up, via_one_hot)
            print("embedding shape:", looked_up.shape)
            """
        ),
        md(
            """
            ## 2. Gradients collect at selected rows

            A token appearing twice contributes twice to its embedding-row gradient. Tokens absent
            from a batch receive zero embedding gradient for that step. Sparse lookup semantics help
            interpret frequency effects and why rare token embeddings learn slowly.
            """
        ),
        code(
            """
            looked_up.sum().backward()
            row_counts = torch.bincount(token_ids.flatten(), minlength=V).float()
            expected_gradient = row_counts[:, None].expand(V, C)
            torch.testing.assert_close(table.grad, expected_gradient)
            print("token counts:", row_counts.tolist())
            """
        ),
        md(
            r"""
            ## 3. Without position, attention cannot know order

            Token embeddings represent identity but not absolute or relative position. A classic
            learned absolute scheme adds $P_t$ to every token at position $t$. Sinusoidal encodings
            use fixed frequencies:

            $$PE_{t,2i}=\sin(t/10000^{2i/C}),\quad
            PE_{t,2i+1}=\cos(t/10000^{2i/C}).$$
            """
        ),
        code(
            """
            def sinusoidal_positions(length: int, width: int) -> torch.Tensor:
                positions = torch.arange(length, dtype=torch.float32)[:, None]
                frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10_000.0) / width))
                encoding = torch.zeros(length, width)
                encoding[:, 0::2] = torch.sin(positions * frequencies)
                encoding[:, 1::2] = torch.cos(positions * frequencies)
                return encoding

            positions = sinusoidal_positions(T, C)                  # [T,C]
            residual_stream = table.detach()[token_ids] + positions # [B,T,C]
            assert positions.shape == (T, C) and residual_stream.shape == (B, T, C)
            print(positions.round(decimals=3))
            """
        ),
        md(
            """
            ## 4. The residual stream is a shared communication workspace

            The residual stream has shape `[B,T,C]` throughout the decoder. Embeddings initialize it;
            attention and MLP sublayers add updates. It is not one interpretable feature vector per
            concept, nor is it copied separately for every layer. Normalization reads a transformed
            view while residual additions maintain a consistent width.
            """
        ),
        code(
            """
            learned_positions = torch.nn.Embedding(T, C)
            position_ids = torch.arange(T)
            x = torch.nn.Embedding.from_pretrained(table.detach())(token_ids)
            x = x + learned_positions(position_ids)[None, :, :]
            update = torch.randn_like(x) * 0.01
            x_next = x + update
            assert x_next.shape == (B, T, C)
            print("residual RMS before/after:", x.square().mean().sqrt().item(), x_next.square().mean().sqrt().item())
            """
        ),
        md(
            r"""
            ## 5. RoPE rotates queries and keys by position

            Rotary position embeddings group head features into 2D pairs and rotate each pair by a
            position-dependent angle. Unlike learned or sinusoidal addition to the residual stream,
            RoPE is applied to queries and keys inside attention. Rotation preserves each vector norm,
            while the query-key dot product encodes their relative position.

            $$R_m q \cdot R_n k = q^\top R_{n-m}k$$
            """
        ),
        code(
            """
            def apply_rope(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
                # values: [B,H,T,D], D must be even; positions: [T]
                head_dim = values.shape[-1]
                if head_dim % 2:
                    raise ValueError("RoPE head dimension must be even")
                frequencies = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2, device=values.device) / head_dim))
                angles = positions[:, None] * frequencies[None, :]  # [T,D/2]
                cosine = angles.cos()[None, None, :, :]
                sine = angles.sin()[None, None, :, :]
                even, odd = values[..., 0::2], values[..., 1::2]
                rotated = torch.stack((even * cosine - odd * sine,
                                       even * sine + odd * cosine), dim=-1)
                return rotated.flatten(start_dim=-2)

            q = torch.randn(B, 2, T, C // 2)  # [B,H,T,D], D=4
            k = torch.randn_like(q)
            rope_positions = torch.arange(T)
            rotated_q = apply_rope(q, rope_positions)
            rotated_k = apply_rope(k, rope_positions)
            torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
            torch.testing.assert_close(rotated_k.norm(dim=-1), k.norm(dim=-1))
            print("RoPE Q/K shapes:", rotated_q.shape, rotated_k.shape)
            """
        ),
        code(
            """
            # The same q/k pair at position pairs with equal relative offset has the same dot product.
            base_q = torch.randn(1, 1, 1, 4).expand(1, 1, 4, 4).clone()
            base_k = torch.randn(1, 1, 1, 4).expand(1, 1, 4, 4).clone()
            rq = apply_rope(base_q, torch.arange(4))
            rk = apply_rope(base_k, torch.arange(4))
            dot_01 = (rq[:, :, 0] * rk[:, :, 1]).sum(dim=-1)
            dot_12 = (rq[:, :, 1] * rk[:, :, 2]).sum(dim=-1)
            dot_23 = (rq[:, :, 2] * rk[:, :, 3]).sum(dim=-1)
            torch.testing.assert_close(dot_01, dot_12, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(dot_12, dot_23, atol=1e-6, rtol=1e-5)
            print("equal relative offsets produce equal rotated dot products")
            """
        ),
    ]
    write(
        "33_embeddings_positions_rope.ipynb",
        lesson(
            number=33,
            title="Embeddings, Residual Stream, Position, and RoPE",
            coverage="V2 5.1–5.3",
            why="A decoder begins by turning token IDs into continuous states and injecting order. These `[B,T,C]` states become the residual workspace used by every later block.",
            objectives=[
                "Implement embedding lookup and prove its one-hot equivalence.",
                "Explain how repeated IDs accumulate embedding gradients.",
                "Construct learned and sinusoidal position representations.",
                "Implement RoPE on query/key head pairs and verify its invariants.",
                "Describe the residual stream's stable shape and additive updates.",
            ],
            cells=cells,
            failures=[
                "Out-of-range token ID: lookup addresses no embedding row.",
                "No position signal: reordered equal-token multisets are indistinguishable to permutation-equivariant attention.",
                "Context beyond learned table: absolute position lookup exceeds trained rows.",
                "RoPE applied to values or residual states: the architecture no longer implements rotary Q/K attention.",
                "Residual width mismatch: a sublayer update cannot join the shared stream.",
            ],
            exercises=[
                "Calculate embedding parameter counts for three vocabulary/width choices.",
                "Verify which embedding rows receive nonzero gradients for a batch with repeated tokens.",
                "Plot dot-product similarity among sinusoidal position vectors and interpret the pattern cautiously.",
                "Add RoPE to the next attention notebook and compare output with and without rotation.",
            ],
            exit_condition="you can produce `[B,T,C]` states from IDs and implement both additive absolute position and rotary Q/K position while explaining their different insertion points.",
            next_lesson="34 — Queries, keys, values, and scaled attention.",
        ),
    )


def build_transformer_attention() -> None:
    cells = [
        md(
            r"""
            ## 1. Attention is content-addressed weighted retrieval

            Each token produces a query (what am I looking for?), key (what do I contain?), and
            value (what information do I send?). For one head:

            $$Q=XW_Q,\quad K=XW_K,\quad V=XW_V,$$
            $$A=\operatorname{softmax}(QK^\top/\sqrt D),\quad Y=AV.$$

            Shapes: `X=[B,T,C]`, projections `[C,D]`, `Q,K,V=[B,T,D]`, scores/weights
            `[B,Tq,Tk]`, output `[B,T,D]`.
            """
        ),
        code(
            """
            import math
            import torch

            torch.manual_seed(42)
            B, T, C, D = 2, 4, 6, 3
            X = torch.randn(B, T, C)
            Wq, Wk, Wv = (torch.randn(C, D) for _ in range(3))
            Q, K, V = X @ Wq, X @ Wk, X @ Wv
            scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
            weights = torch.softmax(scores, dim=-1)
            output = weights @ V
            print("Q/K/V:", Q.shape, "scores:", scores.shape, "output:", output.shape)
            torch.testing.assert_close(weights.sum(dim=-1), torch.ones(B, T))
            """
        ),
        md(
            r"""
            ## 2. Why divide by $\sqrt D$?

            If query/key components are independent with unit variance, their dot product has
            variance $D$. Growing $D$ makes logits large, softmax overly sharp, and gradients small.
            Dividing by $\sqrt D$ keeps score variance near one at initialization.
            """
        ),
        code(
            """
            for head_dim in (8, 32, 128, 512):
                q = torch.randn(20_000, head_dim)
                k = torch.randn(20_000, head_dim)
                raw = (q * k).sum(dim=-1)
                scaled = raw / math.sqrt(head_dim)
                print(f"D={head_dim:3}: raw std={raw.std():6.2f}, scaled std={scaled.std():5.2f}")
            """
        ),
        md(
            """
            ## 3. Attention output is a convex combination per query

            Softmax weights are nonnegative and sum to one. Each output row lies in the convex hull
            of the value rows for that head. Learned output projection and residual addition later
            mix heads and recover a full-width update.
            """
        ),
        code(
            """
            simple_values = torch.tensor([[[0., 0.], [2., 0.], [0., 4.]]])  # [1,3,2]
            simple_weights = torch.tensor([[[0.25, 0.25, 0.50]]])            # [1,1,3]
            retrieved = simple_weights @ simple_values                       # [1,1,2]
            print("retrieved:", retrieved)
            torch.testing.assert_close(retrieved, torch.tensor([[[0.5, 2.0]]]))
            """
        ),
        md(
            """
            ## 4. Mask before softmax

            Disallowed scores receive `-inf` (or the minimum safe representable value) before
            softmax, yielding zero probability. Multiplying probabilities by a mask afterward breaks
            row normalization unless renormalized and may allow information to affect earlier steps.
            A fully masked row has no valid categorical distribution and can produce NaN.
            """
        ),
        code(
            """
            causal_allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
            masked_scores = scores.masked_fill(~causal_allowed, float("-inf"))
            causal_weights = torch.softmax(masked_scores, dim=-1)
            causal_output = causal_weights @ V
            assert not causal_weights.triu(diagonal=1).any()
            torch.testing.assert_close(causal_weights.sum(dim=-1), torch.ones(B, T))
            print("first query weights:", causal_weights[0, 0])
            """
        ),
        md(
            """
            ## 5. Match PyTorch's scaled dot-product attention

            Framework kernels may fuse masking, softmax, dropout, and value aggregation. First prove
            semantic equivalence on a small deterministic example; performance work comes later.
            """
        ),
        code(
            """
            reference = torch.nn.functional.scaled_dot_product_attention(
                Q[:, None], K[:, None], V[:, None], is_causal=True, dropout_p=0.0
            ).squeeze(1)
            torch.testing.assert_close(causal_output, reference, atol=1e-6, rtol=1e-5)
            print("manual causal attention matches PyTorch SDPA")
            """
        ),
    ]
    write(
        "34_scaled_attention.ipynb",
        lesson(
            number=34,
            title="Queries, Keys, Values, and Scaled Attention",
            coverage="V2 5.4–5.5",
            why="Attention is the Transformer's communication mechanism. Deriving every axis and invariant now prevents opaque errors when heads, masks, caching, and optimized kernels are added.",
            objectives=[
                "Project residual states into queries, keys, and values.",
                "Derive score, weight, and output shapes for one attention head.",
                "Explain scaling by square root of head dimension statistically.",
                "Apply a causal mask before softmax and match PyTorch SDPA.",
            ],
            cells=cells,
            failures=[
                "Wrong transpose: dot products mix feature or query axes incorrectly.",
                "Missing scaling: softmax saturates increasingly as head width grows.",
                "Mask after softmax: disallowed positions influenced normalization or keep nonzero mass.",
                "Fully masked row: softmax of all negative infinity becomes undefined.",
            ],
            exercises=[
                "Write all shapes for cross-attention where query and key sequence lengths differ.",
                "Change one key vector and identify which score column and output rows can change.",
                "Gradient-check the manual attention function in float64 on a tiny tensor.",
            ],
            exit_condition="you can implement causal scaled dot-product attention, annotate every contraction, and verify its probabilities and reference output.",
            next_lesson="35 — Causal multi-head attention.",
        ),
    )


def build_transformer_multihead() -> None:
    cells = [
        md(
            r"""
            ## 1. Multiple heads create multiple retrieval subspaces

            Project to combined $Q,K,V$ with width $C$, reshape `C = H × D`, and transpose to
            `[B,H,T,D]`. Each head produces `[B,H,T,D]`; concatenate back to `[B,T,C]`, then apply
            output projection $W_O$. Heads do not permanently own fixed human-interpretable roles.
            """
        ),
        code(
            """
            import math
            import torch

            torch.manual_seed(42)
            B, T, C, H = 2, 5, 12, 3
            assert C % H == 0
            D = C // H
            x = torch.randn(B, T, C)
            qkv_weight = torch.randn(C, 3 * C) / math.sqrt(C)
            output_weight = torch.randn(C, C) / math.sqrt(C)

            qkv = x @ qkv_weight                                  # [B,T,3C]
            q, k, v = qkv.chunk(3, dim=-1)                        # each [B,T,C]

            def split_heads(tensor):
                return tensor.view(B, T, H, D).transpose(1, 2)    # [B,H,T,D]

            q, k, v = map(split_heads, (q, k, v))
            scores = q @ k.transpose(-2, -1) / math.sqrt(D)       # [B,H,T,T]
            allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
            weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
            heads = weights @ v                                   # [B,H,T,D]
            joined = heads.transpose(1, 2).contiguous().view(B, T, C)
            update = joined @ output_weight                       # [B,T,C]
            print("qkv/head/scores/update:", qkv.shape, q.shape, scores.shape, update.shape)
            """
        ),
        md(
            """
            ## 2. Reshape/transpose order is semantic

            `view(B,T,H,D).transpose(1,2)` groups adjacent features into heads. A different reshape
            can preserve element count yet assign values to the wrong head/time coordinates. After
            transposition, call `contiguous()` before `view`; `reshape` may silently allocate.
            """
        ),
        code(
            """
            marker = torch.arange(B * T * C).reshape(B, T, C)
            head_marker = marker.view(B, T, H, D).transpose(1, 2)
            round_trip = head_marker.transpose(1, 2).contiguous().view(B, T, C)
            assert torch.equal(marker, round_trip)
            print("head 0, token 0 features:", head_marker[0, 0, 0].tolist())
            """
        ),
        md(
            """
            ## 3. Residual update and dropout

            Multi-head attention returns a full-width update; the decoder adds it to the input
            residual stream. Attention dropout acts on attention probabilities during training;
            residual dropout acts on projected updates. Both must be disabled for deterministic
            evaluation and generation.
            """
        ),
        code(
            """
            residual_output = x + update
            assert residual_output.shape == x.shape
            assert torch.isfinite(residual_output).all()
            torch.testing.assert_close(weights.sum(dim=-1), torch.ones(B, H, T))
            assert not weights.triu(diagonal=1).any()
            print("causal and normalization invariants passed for every batch/head/query")
            """
        ),
        md(
            """
            ## 4. A reusable transparent module

            This implementation deliberately keeps split, mask, softmax, join, and projection visible.
            Later the production brain can replace its internals with fused SDPA after equivalence tests.
            """
        ),
        code(
            """
            class CausalSelfAttention(torch.nn.Module):
                def __init__(self, width: int, heads: int):
                    super().__init__()
                    assert width % heads == 0
                    self.heads = heads
                    self.head_dim = width // heads
                    self.qkv = torch.nn.Linear(width, 3 * width, bias=False)
                    self.out = torch.nn.Linear(width, width, bias=False)

                def forward(self, x):
                    batch, time, width = x.shape
                    q, k, v = self.qkv(x).chunk(3, dim=-1)
                    def heads(value):
                        return value.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
                    q, k, v = map(heads, (q, k, v))
                    scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
                    allowed = torch.ones(time, time, dtype=torch.bool, device=x.device).tril()
                    attention = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
                    values = attention @ v
                    values = values.transpose(1, 2).contiguous().view(batch, time, width)
                    return self.out(values), attention

            module = CausalSelfAttention(C, H)
            result, attention = module(x)
            assert result.shape == x.shape and attention.shape == (B, H, T, T)
            """
        ),
    ]
    write(
        "35_causal_multihead_attention.ipynb",
        lesson(
            number=35,
            title="Causal Multi-Head Attention",
            coverage="V2 5.6–5.8",
            why="Multi-head attention is where shape complexity sharply increases. A transparent implementation makes head splitting, masking, concatenation, projection, and residual integration auditable.",
            objectives=[
                "Split combined projections into `[B,H,T,D]` heads and join them losslessly.",
                "Compute causal attention independently for every head.",
                "Project concatenated heads into a residual update.",
                "Package the operation in a readable PyTorch module.",
            ],
            cells=cells,
            failures=[
                "Width not divisible by heads: head dimension is not integral.",
                "Incorrect reshape order: features silently map to the wrong token/head.",
                "Missing contiguous conversion: view fails after transpose or data is unexpectedly copied elsewhere.",
                "Head output not projected: concatenated subspaces do not receive learned mixing before residual addition.",
            ],
            exercises=[
                "Add attention and residual dropout with correct train/eval behavior.",
                "Copy module weights into an equivalent SDPA implementation and compare outputs and gradients.",
                "Write assertions that detect future attention for every batch and head.",
            ],
            exit_condition="you can derive every multi-head shape, round-trip the layout, and prove causal probability invariants.",
            next_lesson="36 — Normalization, feed-forward networks, and a decoder block.",
        ),
    )


def build_transformer_block() -> None:
    cells = [
        md(
            """
            ## 1. A classic pre-norm decoder block

            ```text
            x = x + causal_attention(layer_norm_1(x))
            x = x + mlp(layer_norm_2(x))
            ```

            Attention communicates between positions. The MLP transforms each position independently.
            Residual additions preserve `[B,T,C]`. Pre-norm gives both sublayers standardized inputs
            and maintains a direct residual route through depth.
            """
        ),
        code(
            """
            import math
            import torch
            import torch.nn.functional as F

            class CausalSelfAttention(torch.nn.Module):
                def __init__(self, width, heads, dropout=0.0):
                    super().__init__()
                    assert width % heads == 0
                    self.heads, self.head_dim = heads, width // heads
                    self.qkv = torch.nn.Linear(width, 3 * width)
                    self.out = torch.nn.Linear(width, width)
                    self.dropout = dropout

                def forward(self, x):
                    B, T, C = x.shape
                    q, k, v = self.qkv(x).chunk(3, dim=-1)
                    def split(z): return z.view(B, T, self.heads, self.head_dim).transpose(1, 2)
                    q, k, v = map(split, (q, k, v))
                    values = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True,
                        dropout_p=self.dropout if self.training else 0.0,
                    )
                    values = values.transpose(1, 2).contiguous().view(B, T, C)
                    return self.out(values)
            """
        ),
        md(
            r"""
            ## 2. LayerNorm, RMSNorm, pre-norm, and post-norm

            LayerNorm subtracts the feature mean and divides by feature standard deviation, then
            learns scale and shift. RMSNorm divides by root mean square and usually learns only scale.
            Both normalize each token independently across $C$.

            Pre-norm uses `x + sublayer(norm(x))`; post-norm uses `norm(x + sublayer(x))`. Their
            gradient paths and checkpoint layouts differ, so the choice is part of architecture.
            """
        ),
        code(
            """
            class RMSNorm(torch.nn.Module):
                def __init__(self, width, eps=1e-6):
                    super().__init__()
                    self.scale = torch.nn.Parameter(torch.ones(width))
                    self.eps = eps

                def forward(self, x):
                    rms = x.square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
                    return self.scale * (x / rms)

            norm_input = torch.randn(2, 5, 32) * 3 + 4
            rms_norm = RMSNorm(32)
            rms_output = rms_norm(norm_input)
            torch.testing.assert_close(
                rms_output.square().mean(dim=-1), torch.ones(2, 5), atol=2e-5, rtol=2e-5
            )
            print("LayerNorm parameters:", sum(p.numel() for p in torch.nn.LayerNorm(32).parameters()))
            print("RMSNorm parameters:", sum(p.numel() for p in rms_norm.parameters()))
            """
        ),
        md(
            r"""
            ## 3. Position-wise GELU and SwiGLU feed-forward networks

            A classic GPT-style FFN expands $C\to4C$, applies GELU, then projects $4C\to C$.
            It shares weights across positions but processes each position separately. Its parameter
            count is approximately $8C^2$, often more than the attention projections' $4C^2$.

            Modern decoders often use SwiGLU: two input projections produce a gate and value, SiLU
            gates one branch, their elementwise product is projected back to $C$. A hidden width near
            $8C/3$ gives a parameter count comparable to the classic 4C FFN.
            """
        ),
        code(
            """
            class FeedForward(torch.nn.Module):
                def __init__(self, width, multiplier=4, dropout=0.0):
                    super().__init__()
                    hidden = multiplier * width
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(width, hidden),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden, width),
                        torch.nn.Dropout(dropout),
                    )

                def forward(self, x):
                    return self.net(x)

            width = 32
            ffn = FeedForward(width)

            class SwiGLU(torch.nn.Module):
                def __init__(self, width, hidden=None):
                    super().__init__()
                    hidden = hidden or round(8 * width / 3)
                    self.gate = torch.nn.Linear(width, hidden, bias=False)
                    self.value = torch.nn.Linear(width, hidden, bias=False)
                    self.down = torch.nn.Linear(hidden, width, bias=False)

                def forward(self, x):
                    return self.down(F.silu(self.gate(x)) * self.value(x))

            swiglu = SwiGLU(width)
            sample = torch.randn(2, 5, width)
            assert ffn(sample).shape == swiglu(sample).shape == sample.shape
            print("GELU FFN parameters:", sum(p.numel() for p in ffn.parameters()))
            print("SwiGLU parameters:", sum(p.numel() for p in swiglu.parameters()))
            """
        ),
        md(
            """
            ## 4. Assemble and inspect the block

            Each LayerNorm has independent learned scale and bias. Do not reuse one norm unless that
            sharing is an intentional architecture decision. Dropout belongs inside update branches,
            before their residual addition.
            """
        ),
        code(
            """
            class DecoderBlock(torch.nn.Module):
                def __init__(self, width, heads, dropout=0.0):
                    super().__init__()
                    self.norm1 = torch.nn.LayerNorm(width)
                    self.attention = CausalSelfAttention(width, heads, dropout)
                    self.norm2 = torch.nn.LayerNorm(width)
                    self.ffn = FeedForward(width, dropout=dropout)

                def forward(self, x):
                    x = x + self.attention(self.norm1(x))
                    x = x + self.ffn(self.norm2(x))
                    return x

            torch.manual_seed(42)
            block = DecoderBlock(width=32, heads=4, dropout=0.1)
            x = torch.randn(2, 8, 32, requires_grad=True)
            y = block(x)
            assert y.shape == x.shape
            y.square().mean().backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in block.parameters())
            print("block output:", y.shape, "parameters:", sum(p.numel() for p in block.parameters()))
            """
        ),
        md(
            """
            ## 5. Causality is an end-to-end property

            A causal mask inside attention should guarantee that changing future tokens does not
            change earlier block outputs in evaluation mode. This black-box test catches mask,
            transpose, and accidental time-mixing bugs without trusting internal attention weights.
            """
        ),
        code(
            """
            block.eval()
            prefix = torch.randn(1, 4, 32)
            future_a = torch.randn(1, 3, 32)
            future_b = torch.randn(1, 3, 32) * 100
            sequence_a = torch.cat((prefix, future_a), dim=1)
            sequence_b = torch.cat((prefix, future_b), dim=1)
            with torch.inference_mode():
                output_a = block(sequence_a)
                output_b = block(sequence_b)
            torch.testing.assert_close(output_a[:, :4], output_b[:, :4], atol=1e-5, rtol=1e-5)
            assert not torch.allclose(output_a[:, 4:], output_b[:, 4:])
            print("future perturbation cannot change prefix outputs")
            """
        ),
    ]
    write(
        "36_norms_feedforward_decoder_block.ipynb",
        lesson(
            number=36,
            title="Normalization, Gated Feed-Forward Networks, and a Decoder Block",
            coverage="V2 5.9–5.12",
            why="The decoder block combines communication, per-token computation, normalization, and identity paths. It is the repeating unit whose correctness and stability determine the whole model.",
            objectives=[
                "Implement and compare LayerNorm and RMSNorm semantics.",
                "Distinguish pre-norm and post-norm residual ordering.",
                "Build and count GELU and SwiGLU feed-forward sublayers.",
                "Implement a classic pre-norm residual block.",
                "Verify shapes and finite gradients through the complete block.",
                "Test causal behavior from block inputs to outputs.",
            ],
            cells=cells,
            failures=[
                "Post/pre-norm confusion: checkpoint architecture differs despite matching names.",
                "Shared normalization by accident: attention and MLP updates use tied scale/shift parameters.",
                "Wrong normalization axis: tokens or batch items influence each other's statistics.",
                "Dropout after residual sum: the identity path is corrupted.",
                "Internal-only causality check: a later time-mixing operation reintroduces leakage.",
            ],
            exercises=[
                "Manually derive the block parameter count as a function of C and expansion multiplier.",
                "Replace LayerNorm/GELU with RMSNorm/SwiGLU while preserving the public block contract.",
                "Compare input-gradient norms across 12 pre-norm and post-norm blocks.",
            ],
            exit_condition="you can compare normalization and FFN variants, assemble a block, account for every parameter, and prove finite gradients and end-to-end causality.",
            next_lesson="37 — Decoder stack and language-model head.",
        ),
    )


def build_transformer_stack() -> None:
    cells = [
        md(
            """
            ## 1. Stack blocks into a decoder-only language model

            Token and position embeddings initialize `[B,T,C]`. `L` decoder blocks repeatedly update
            it. A final normalization produces states for an LM head mapping `C → V`; logits are
            `[B,T,V]`. The model predicts each next token in parallel during training because a causal
            mask prevents forbidden context.
            """
        ),
        code(
            """
            import math
            from dataclasses import dataclass
            import torch
            import torch.nn.functional as F

            @dataclass(frozen=True)
            class Config:
                vocab_size: int = 23
                context: int = 12
                width: int = 32
                heads: int = 4
                layers: int = 2
                dropout: float = 0.0

            class Attention(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    assert config.width % config.heads == 0
                    self.heads = config.heads
                    self.head_dim = config.width // config.heads
                    self.qkv = torch.nn.Linear(config.width, 3 * config.width)
                    self.out = torch.nn.Linear(config.width, config.width)

                def forward(self, x):
                    B, T, C = x.shape
                    q, k, v = self.qkv(x).chunk(3, dim=-1)
                    def split(z): return z.view(B, T, self.heads, self.head_dim).transpose(1, 2)
                    q, k, v = map(split, (q, k, v))
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    return self.out(y.transpose(1, 2).contiguous().view(B, T, C))

            class Block(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.norm1 = torch.nn.LayerNorm(config.width)
                    self.attention = Attention(config)
                    self.norm2 = torch.nn.LayerNorm(config.width)
                    self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(config.width, 4 * config.width), torch.nn.GELU(),
                        torch.nn.Linear(4 * config.width, config.width),
                    )
                def forward(self, x):
                    x = x + self.attention(self.norm1(x))
                    return x + self.mlp(self.norm2(x))
            """
        ),
        code(
            """
            class TinyDecoder(torch.nn.Module):
                def __init__(self, config: Config, tie_weights: bool = True):
                    super().__init__()
                    self.config = config
                    self.token_embedding = torch.nn.Embedding(config.vocab_size, config.width)
                    self.position_embedding = torch.nn.Embedding(config.context, config.width)
                    self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.layers)])
                    self.final_norm = torch.nn.LayerNorm(config.width)
                    self.lm_head = torch.nn.Linear(config.width, config.vocab_size, bias=False)
                    if tie_weights:
                        self.lm_head.weight = self.token_embedding.weight

                def forward(self, token_ids, targets=None):
                    B, T = token_ids.shape
                    if T > self.config.context:
                        raise ValueError(f"sequence length {T} exceeds context {self.config.context}")
                    positions = torch.arange(T, device=token_ids.device)
                    x = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
                    for block in self.blocks:
                        x = block(x)
                    logits = self.lm_head(self.final_norm(x))
                    loss = None
                    if targets is not None:
                        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    return logits, loss

            torch.manual_seed(42)
            config = Config()
            model = TinyDecoder(config)
            token_ids = torch.randint(0, config.vocab_size, (3, 8))
            targets = torch.randint(0, config.vocab_size, (3, 8))
            logits, loss = model(token_ids, targets)
            print("logits:", logits.shape, "loss:", loss.item())
            assert logits.shape == (3, 8, config.vocab_size)
            """
        ),
        md(
            """
            ## 2. Weight tying

            Input embeddings map token IDs into features; the LM head compares features with output
            token rows. Tying uses the same `[V,C]` parameter for both, reducing parameter count and
            often improving statistical efficiency. Assignment must preserve true parameter identity,
            not copy values once.
            """
        ),
        code(
            """
            assert model.token_embedding.weight is model.lm_head.weight
            untied = TinyDecoder(config, tie_weights=False)
            tied_params = sum(p.numel() for p in model.parameters())
            untied_params = sum(p.numel() for p in untied.parameters())
            print("tied:", tied_params, "untied:", untied_params, "saved:", untied_params - tied_params)
            assert untied_params - tied_params == config.vocab_size * config.width
            """
        ),
        md(
            """
            ## 3. Initialization and parameter accounting

            Parameter count is a testable architectural property. Embeddings scale with `V×C` and
            `T×C`; each classic block is dominated by about `12C²` weights (4C² attention and 8C²
            MLP). Exact counts include biases and normalization parameters.
            """
        ),
        code(
            """
            for name, parameter in model.named_parameters():
                assert torch.isfinite(parameter).all()
                print(f"{name:40} {tuple(parameter.shape)!s:16} {parameter.numel():6}")
            print("total:", tied_params)

            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters())
            print("all trainable parameters received finite gradients")
            """
        ),
    ]
    write(
        "37_decoder_stack_lm_head.ipynb",
        lesson(
            number=37,
            title="Decoder Stack and Language-Model Head",
            coverage="V2 5.13–5.14",
            why="This is the first complete decoder forward graph: IDs become embeddings, blocks exchange and transform information, and the LM head emits one vocabulary distribution per position.",
            objectives=[
                "Assemble embeddings, repeated blocks, final norm, and LM head.",
                "Compute `[B,T,V]` logits and next-token loss.",
                "Implement true input/output weight tying.",
                "Account for parameters and verify all gradients are finite.",
            ],
            cells=cells,
            failures=[
                "Context overflow: position embeddings or masks do not cover the requested sequence.",
                "Copied rather than tied weights: embedding and head diverge after one update.",
                "Missing final norm: architecture no longer matches the intended pre-norm design.",
                "Wrong flatten order: logits and targets refer to different positions.",
            ],
            exercises=[
                "Derive an exact parameter-count formula and verify it for three configurations.",
                "Add an `ignore_index` for padded targets and test that padding changes neither loss nor gradients.",
                "Test end-to-end causality by changing suffix token IDs and comparing prefix logits.",
            ],
            exit_condition="you can construct the full logits graph, explain every state-dict tensor, and prove parameter, shape, gradient, and causality invariants.",
            next_lesson="38 — Transformer complexity and failure analysis.",
        ),
    )


def build_transformer_complexity() -> None:
    cells = [
        md(
            r"""
            ## 1. Separate parameter, compute, and activation scaling

            For batch $B$, length $T$, width $C$, heads $H$, layers $L$:

            - QKV + output projections: roughly $4BTC^2$ multiply-adds per layer;
            - attention score and value products: roughly $2BT^2C$ per layer;
            - classic 4C MLP: roughly $8BTC^2$ per layer;
            - attention matrix storage: $BHT^2$ values per layer when materialized.

            Long context makes the quadratic term dominant; wide models make projection/MLP terms dominant.
            """
        ),
        code(
            """
            import math
            import time
            import torch

            def rough_layer_work(batch, time_steps, width):
                linear = 12 * batch * time_steps * width**2
                attention = 2 * batch * time_steps**2 * width
                return linear, attention

            for T in (128, 512, 2048, 8192):
                linear, attention = rough_layer_work(1, T, 768)
                print(f"T={T:5}: linear={linear/1e9:8.2f}G, attention={attention/1e9:8.2f}G, ratio={attention/linear:.3f}")
            """
        ),
        md(
            """
            ## 2. Attention memory grows quadratically

            Optimized exact attention such as FlashAttention reduces intermediate memory traffic and
            avoids materializing the full matrix, but does not change the mathematical dense-attention
            pair count. Fused kernels improve constants and memory behavior; sparse/local schemes
            change which pairs exist and therefore the computation.
            """
        ),
        code(
            """
            def attention_matrix_gib(batch, heads, context, bytes_per_value=2):
                return batch * heads * context * context * bytes_per_value / 2**30

            for T in (512, 2048, 8192, 32768):
                print(f"T={T:5}: one 12-head fp16 score/prob matrix = {attention_matrix_gib(1,12,T):.3f} GiB")
            """
        ),
        md(
            """
            ## 3. Build failure tests before trusting training

            High-value invariants include shape/dtype/device checks, finite logits/loss/gradients,
            probability-row sums, no future influence, weight-tie identity, deterministic eval,
            overfit-one-batch, save/load equivalence, and exact-resume tests. A decreasing full training
            curve is too indirect to localize most failures.
            """
        ),
        code(
            """
            def causal_leak_test(model, prefix, suffix_a, suffix_b, atol=1e-5):
                model.eval()
                with torch.inference_mode():
                    first = model(torch.cat((prefix, suffix_a), dim=1))[0]
                    second = model(torch.cat((prefix, suffix_b), dim=1))[0]
                torch.testing.assert_close(first[:, :prefix.shape[1]], second[:, :prefix.shape[1]],
                                           atol=atol, rtol=atol)

            # Demonstrate the invariant directly on a correct causal weight matrix.
            T = 6
            allowed = torch.ones(T, T, dtype=torch.bool).tril()
            logits = torch.randn(2, 3, T, T).masked_fill(~allowed, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            assert not probabilities.triu(diagonal=1).any()
            torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, 3, T))
            """
        ),
        md(
            """
            ## 4. Attention weights are evidence, not explanations

            A large weight says a query used a value strongly in that head at that layer. The value
            vector, output projection, residual stream, later layers, and nonlinear transformations
            determine effect. Attention can help debug masks and patterns, but it is not automatically
            a faithful explanation of a final prediction.
            """
        ),
        code(
            """
            # Same attention weights, different values → different outputs.
            weights = torch.tensor([[0.1, 0.9]])
            values_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            values_b = torch.tensor([[100.0, 0.0], [0.0, -1.0]])
            print("same weights, output A:", weights @ values_a)
            print("same weights, output B:", weights @ values_b)
            assert not torch.allclose(weights @ values_a, weights @ values_b)
            """
        ),
        md(
            """
            ## 5. Benchmarking needs a question

            Report hardware/software, dtype, shapes, warm-up, repeats, synchronization, memory metric,
            and whether backward is included. Compare outputs within a precision-appropriate tolerance.
            Throughput, latency, and peak memory are different objectives; batch size can improve one
            while harming another.
            """
        ),
        code(
            """
            # A CPU-only, small semantic benchmark example—not a universal performance claim.
            q = torch.randn(1, 4, 256, 32)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            for _ in range(3):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            start = time.perf_counter()
            repeats = 10
            for _ in range(repeats):
                result = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            elapsed = time.perf_counter() - start
            print(f"shape={tuple(q.shape)}, dtype={q.dtype}, mean latency={elapsed/repeats*1e3:.3f} ms")
            assert torch.isfinite(result).all()
            """
        ),
    ]
    write(
        "38_transformer_complexity_failures.ipynb",
        lesson(
            number=38,
            title="Transformer Complexity and Failure Analysis",
            coverage="V2 5.15–5.16",
            why="A correct model must also fit memory, meet latency goals, and fail loudly when invariants break. Complexity estimates and targeted tests turn debugging from guesswork into engineering.",
            objectives=[
                "Estimate linear-layer and attention compute scaling.",
                "Estimate materialized attention memory against context length.",
                "Design targeted shape, causality, finiteness, state, and resume tests.",
                "Interpret attention weights without treating them as complete explanations.",
            ],
            cells=cells,
            failures=[
                "Quadratic term ignored: a context increase exceeds compute or memory budget.",
                "Optimized-kernel mythology: fused exact attention is claimed to change dense asymptotic pair count.",
                "Loss-only debugging: silent leakage or state bugs remain unlocalized.",
                "Benchmark without equivalence: faster outputs solve a different or numerically broken problem.",
            ],
            exercises=[
                "Find the context length where attention work equals linear work for width C under the rough formulas.",
                "Write a failure-test matrix mapping ten symptoms to the smallest invariant test.",
                "Design a benchmark report comparing manual attention and SDPA on your available device.",
            ],
            exit_condition="you can budget a configuration, define its performance question, and select a minimal test for each major Transformer failure class.",
            next_lesson="39 — Tiny decoder readiness capstone.",
        ),
    )


def build_transformer_capstone() -> None:
    cells = [
        md(
            """
            ## Capstone contract

            Build a tiny character-level decoder without `nn.Transformer`, overfit one fixed batch,
            generate text, and produce architecture/verification evidence. This is a proof of
            understanding, not the production SLM. Keep it deliberately small and deterministic.

            Success criteria in this notebook:

            - input `[B,T]` → logits `[B,T,V]`;
            - every prefix logit is independent of future input tokens;
            - all trainable parameters receive finite gradients;
            - one fixed batch loss drops far below the uniform baseline `log(V)`;
            - greedy generation uses only the available context window.
            """
        ),
        code(
            """
            import math
            import random
            from dataclasses import asdict, dataclass
            import numpy as np
            import torch
            import torch.nn.functional as F

            seed = 42
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            torch.set_num_threads(1)

            text = ("timing makes the joke. " * 20).strip()
            vocabulary = sorted(set(text))
            stoi = {character: index for index, character in enumerate(vocabulary)}
            itos = dict(enumerate(vocabulary))
            encoded = torch.tensor([stoi[character] for character in text], dtype=torch.long)
            print("characters:", len(text), "vocab:", len(vocabulary), vocabulary)
            """
        ),
        md(
            """
            ## 1. Configuration and model

            Configuration is immutable, serializable experiment state. The model uses learned absolute
            positions, pre-norm blocks, causal SDPA, GELU MLPs, final LayerNorm, and tied embeddings.
            """
        ),
        code(
            """
            @dataclass(frozen=True)
            class Config:
                vocab_size: int
                context: int = 16
                width: int = 32
                heads: int = 4
                layers: int = 2

            class Attention(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    assert config.width % config.heads == 0
                    self.heads = config.heads
                    self.head_dim = config.width // config.heads
                    self.qkv = torch.nn.Linear(config.width, 3 * config.width)
                    self.out = torch.nn.Linear(config.width, config.width)
                def forward(self, x):
                    B, T, C = x.shape
                    q, k, v = self.qkv(x).chunk(3, dim=-1)
                    def split(z): return z.view(B, T, self.heads, self.head_dim).transpose(1, 2)
                    q, k, v = map(split, (q, k, v))
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    return self.out(y.transpose(1, 2).contiguous().view(B, T, C))

            class Block(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.norm1 = torch.nn.LayerNorm(config.width)
                    self.attention = Attention(config)
                    self.norm2 = torch.nn.LayerNorm(config.width)
                    self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(config.width, 4 * config.width), torch.nn.GELU(),
                        torch.nn.Linear(4 * config.width, config.width),
                    )
                def forward(self, x):
                    x = x + self.attention(self.norm1(x))
                    return x + self.mlp(self.norm2(x))

            class TinyDecoder(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.token_embedding = torch.nn.Embedding(config.vocab_size, config.width)
                    self.position_embedding = torch.nn.Embedding(config.context, config.width)
                    self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.layers)])
                    self.final_norm = torch.nn.LayerNorm(config.width)
                    self.lm_head = torch.nn.Linear(config.width, config.vocab_size, bias=False)
                    self.lm_head.weight = self.token_embedding.weight
                def forward(self, ids, targets=None):
                    B, T = ids.shape
                    if T > self.config.context:
                        raise ValueError("input exceeds configured context")
                    positions = torch.arange(T, device=ids.device)
                    x = self.token_embedding(ids) + self.position_embedding(positions)[None]
                    for block in self.blocks:
                        x = block(x)
                    logits = self.lm_head(self.final_norm(x))
                    loss = None if targets is None else F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                    )
                    return logits, loss
            """
        ),
        md(
            """
            ## 2. Build one deterministic batch

            Overfitting one batch is a diagnostic: it proves the graph can represent and optimize
            these examples. It does not prove generalization. Use varied start positions so the batch
            contains multiple local contexts while remaining exactly repeatable.
            """
        ),
        code(
            """
            config = Config(vocab_size=len(vocabulary))
            starts = torch.tensor([0, 3, 7, 11, 17, 23, 31, 41])
            batch_x = torch.stack([encoded[start:start + config.context] for start in starts])
            batch_y = torch.stack([encoded[start + 1:start + config.context + 1] for start in starts])
            model = TinyDecoder(config)
            logits, initial_loss = model(batch_x, batch_y)
            print("config:", asdict(config))
            print("input/target/logits:", batch_x.shape, batch_y.shape, logits.shape)
            print("parameters:", sum(p.numel() for p in model.parameters()))
            print("initial loss / uniform baseline:", initial_loss.item(), math.log(config.vocab_size))
            assert logits.shape == (*batch_x.shape, config.vocab_size)
            """
        ),
        md(
            """
            ## 3. Verify the graph before training

            Check weight identity, end-to-end causality, finite forward values, and gradient coverage.
            The causality test changes only the suffix and compares prefix logits. Run in eval mode so
            no stochastic layer could confound the result.
            """
        ),
        code(
            """
            assert model.token_embedding.weight is model.lm_head.weight
            model.eval()
            prefix_length = 8
            changed = batch_x[:1].clone()
            changed[:, prefix_length:] = torch.flip(changed[:, prefix_length:], dims=(1,))
            with torch.inference_mode():
                original_logits, _ = model(batch_x[:1])
                changed_logits, _ = model(changed)
            torch.testing.assert_close(original_logits[:, :prefix_length], changed_logits[:, :prefix_length],
                                       atol=1e-5, rtol=1e-5)

            model.train()
            _, check_loss = model(batch_x, batch_y)
            check_loss.backward()
            missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
            nonfinite = [name for name, parameter in model.named_parameters()
                         if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            assert not missing and not nonfinite
            model.zero_grad(set_to_none=True)
            print("causality, weight tying, gradient coverage, and finiteness passed")
            """
        ),
        md(
            """
            ## 4. Overfit the fixed batch

            AdamW is used without weight decay because this is a memorization diagnostic. Record loss
            periodically, assert a strong decrease, and stop if values become non-finite. In a real run,
            train and validation data, schedules, clipping, checkpoints, and experiment metadata return.
            """
        ),
        code(
            """
            torch.manual_seed(seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
            losses = []
            for step in range(301):
                _, loss = model(batch_x, batch_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert torch.isfinite(loss) and torch.isfinite(gradient_norm)
                optimizer.step()
                losses.append(loss.item())
                if step % 75 == 0:
                    print(f"step={step:3}, loss={loss.item():.4f}, grad_norm={gradient_norm.item():.3f}")

            assert losses[-1] < 0.08 and losses[-1] < losses[0] * 0.05
            """
        ),
        md(
            """
            ## 5. Autoregressive generation

            Training computes all positions in parallel; generation produces one token at a time.
            At each step crop to the last `context` IDs, take final-position logits, choose one next
            ID, append, and repeat. Greedy decoding is deterministic and useful for this diagnostic.
            """
        ),
        code(
            """
            @torch.inference_mode()
            def generate(model, start_ids, new_tokens):
                model.eval()
                ids = start_ids.clone()
                for _ in range(new_tokens):
                    context_ids = ids[:, -model.config.context:]
                    logits, _ = model(context_ids)
                    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                    ids = torch.cat((ids, next_id), dim=1)
                return ids

            prompt = "timing"
            prompt_ids = torch.tensor([[stoi[character] for character in prompt]])
            generated_ids = generate(model, prompt_ids, 50)[0].tolist()
            generated_text = "".join(itos[index] for index in generated_ids)
            print(generated_text)
            assert generated_text.startswith(prompt)
            """
        ),
        md(
            """
            ## 6. Architecture and experiment report

            Fill this out in your own words before leaving the notebook:

            | Item | This run |
            | --- | --- |
            | Objective | character next-token cross-entropy |
            | Shapes | input `[8,16]`, logits `[8,16,V]` |
            | Architecture | learned token/position embedding, 2 pre-norm decoder blocks, tied head |
            | Verification | shapes, causality, finite values/gradients, weight identity, one-batch overfit |
            | Reproducibility | seed 42, fixed in-notebook corpus/config/command; add Git commit and device in a real run |
            | Limitation | memorizes a repeated tiny corpus; no evidence of general language ability |

            The production `brain/` project begins by translating this understood baseline into tested,
            reusable modules and a real experiment/data pipeline. Modern components come after baseline
            equivalence, so every change has a measured reason.
            """
        ),
    ]
    write(
        "39_tiny_decoder_capstone.ipynb",
        lesson(
            number=39,
            title="Tiny Decoder Readiness Capstone",
            coverage="V2 Part V capstone and bridge to Part VI",
            why="This capstone integrates the entire prerequisite path into one falsifiable result: a transparent decoder that passes invariants, overfits a fixed batch, and generates autoregressively.",
            objectives=[
                "Assemble a complete character-level decoder from explicit components.",
                "Verify shape, causal, finite-gradient, and weight-tying contracts.",
                "Intentionally overfit one deterministic batch.",
                "Generate autoregressively and document architecture, experiment state, and limitations.",
            ],
            cells=cells,
            failures=[
                "Cannot overfit one batch: suspect target shift, causal mask, gradients, capacity, or optimizer before collecting more data.",
                "Prefix changes with suffix: future-token leakage invalidates training loss.",
                "Training/generation context mismatch: positions or crop policy differ.",
                "Memorization called generalization: the diagnostic result is overstated.",
            ],
            exercises=[
                "Reimplement the attention calculation manually and prove output/gradient equivalence with SDPA.",
                "Add stochastic temperature/top-k sampling and separate model probabilities from sampling policy.",
                "Save and resume at step 150, then verify the resumed loss trajectory is identical to uninterrupted training.",
                "Write a one-page readiness report answering every final gate in `SYLLABUS.md` with notebook evidence.",
            ],
            exit_condition="you can rebuild the decoder without copying, explain every shape and parameter, pass all invariants, overfit one batch, and state honestly what the result does not prove.",
            next_lesson="Move to `brain/` milestone B1 — Train MiniGPT as a reusable, reproducible project.",
        ),
    )


BUILDERS = [
    build_00,
    build_system_cpu_gpu,
    build_system_memory,
    build_system_numbers,
    build_system_parallelism,
    build_system_matmul,
    build_system_benchmarking,
    build_system_budget,
    build_system_capstone,
    build_math_tensors,
    build_math_broadcasting,
    build_math_vectors,
    build_math_matrices,
    build_math_optional_svd,
    build_math_derivatives,
    build_math_chain_rule,
    build_math_probability,
    build_math_information,
    build_math_softmax,
    build_math_optimization,
    build_math_capstone,
    build_nn_learning_from_data,
    build_nn_mlps,
    build_nn_scalar_autograd,
    build_nn_tensor_autograd,
    build_nn_training_loop,
    build_nn_initialization,
    build_nn_stable_depth,
    build_text_unicode,
    build_text_ngram,
    build_text_subwords,
    build_text_protocol,
    build_text_sequences,
    build_transformer_embeddings,
    build_transformer_attention,
    build_transformer_multihead,
    build_transformer_block,
    build_transformer_stack,
    build_transformer_complexity,
    build_transformer_capstone,
]


if __name__ == "__main__":
    previous_course_files = set(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    for builder in BUILDERS:
        builder()
    for stale_path in sorted(previous_course_files - WRITTEN):
        stale_path.unlink()
        print(f"removed stale {stale_path.relative_to(ROOT)}")
