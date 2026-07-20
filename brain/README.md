# SLM Brain

The production Python package for defining, training, evaluating, and using the
small language model. Learning notebooks live separately in `../learning/`.

Development of MiniGPT begins here only after completing the readiness gate in
the [foundation syllabus](../learning/SYLLABUS.md). The later brain milestones
cover the V2 curriculum's model training, modernization, data engineering,
post-training, evaluation, and Humor Machine product work.

## Environment

```sh
cd brain
uv sync
```

Runtime dependencies are intentionally small:

| Package | Why it belongs here |
| --- | --- |
| `torch` | tensors, automatic differentiation, model layers, and training |
| `numpy` | numerical interoperability and reproducible NumPy random state |

Development-only packages do not become model runtime requirements:

| Package | Purpose |
| --- | --- |
| `ruff` | formatting and linting |
| `mypy` | static type checking |
| `pytest` | tests |
| `pytest-cov` | test coverage |

## Structure

```text
brain/
├── src/slm/      reusable package code
├── tests/        correctness tests
├── data/         local training data, ignored by Git
└── artifacts/    checkpoints and outputs, ignored by Git
```

As the model grows, add focused packages such as `slm/nn`, `slm/models`,
`slm/training`, and `slm/evaluation` only when they contain real code.

## Verify changes

```sh
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

The current utilities provide reproducible seeding and automatic selection of
CUDA, Apple MPS, or CPU:

```python
from slm import get_default_device, seed_everything

seed_everything(42, deterministic=True)
device = get_default_device()
```
