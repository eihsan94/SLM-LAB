# ML Learning Lab

An independent notebook environment for learning the mathematics and mechanics
of machine learning. Nothing here is imported by the production SLM brain.

## Start

```sh
cd learning
uv sync
uv run jupyter lab
```

When using VS Code, open a notebook and select `learning/.venv/bin/python` as
its kernel. Start with `notebooks/00_slm_map.ipynb`, then follow
[`SYLLABUS.md`](SYLLABUS.md) in order.

All 40 prerequisite lessons are implemented and execute from a clean kernel.
They are intentionally small enough for local study; the final notebook trains
a tiny diagnostic decoder rather than a production model.

## How to study a lesson

1. Read the objectives and predict every important tensor shape.
2. Run one cell at a time and explain the output before continuing.
3. Reimplement the central algorithm without looking at the provided cell.
4. Complete the deliberate-practice exercises by changing the code.
5. Reproduce at least one listed failure mode and diagnose its symptom.
6. Answer the exit ticket in your own words before opening the next notebook.

The notebook generator lives at `scripts/build_notebooks.py`. The generated
`.ipynb` files are the lessons you study; rerun the generator only when editing
the curriculum source itself.

Validate lesson structure quickly with:

```sh
uv run python scripts/validate_notebooks.py
```

Execute every lesson in a separate clean kernel with:

```sh
uv run python scripts/validate_notebooks.py --execute
```

## Why each package is installed

| Package | Used for |
| --- | --- |
| `numpy` | arrays and learning numerical operations without autograd |
| `torch` | tensors, gradients, neural networks, and later model training |
| `matplotlib` | plotting functions, distributions, losses, and training curves |
| `jupyterlab` | running notebooks in a browser |
| `ipykernel` | connecting a notebook editor such as VS Code to this Python environment |
| `nbformat` and `nbconvert` | validating and clean-kernel executing the complete course |

The exact resolved versions are recorded in `uv.lock`. Add a package only when
a notebook needs it, and document its purpose in the table above.

## Curriculum boundary

The learning lab ends with a tiny decoder readiness capstone. Training MiniGPT,
modernizing its architecture, specializing Humor Machine, and building the
runtime happen afterward in `brain/` and `engine/`. This prevents advanced
systems work from delaying the first complete understanding of a transformer.

Only completed, executable notebooks are added to `notebooks/`; planned topics
live in the syllabus rather than as empty placeholder files.
