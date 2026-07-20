# SLM

A small language model built as three independent projects. Each project owns
its dependencies and can be developed without installing the other two.

```text
slm/
├── engine/    C++ execution and visualization engine
├── brain/     reusable Python model and training package
└── learning/  independent notebooks for learning ML mathematics
```

## Choose the project for the task

| Goal | Directory | Dependency file |
| --- | --- | --- |
| Learn tensors, gradients, and architectures | `learning/` | `learning/pyproject.toml` |
| Build and test the real model | `brain/` | `brain/pyproject.toml` |
| Build the high-performance runtime | `engine/` | `engine/vcpkg.json` |

The Python projects intentionally have separate `.venv` directories and
`uv.lock` files. A package installed for a notebook therefore does not become a
runtime dependency of the model.

## Curriculum

The [local executable syllabus](learning/SYLLABUS.md) turns the detailed
[Humor Machine V2 master syllabus](https://app.notion.com/p/39d7d956b4c881178b90e844a55fcecb)
into 40 focused prerequisite notebooks with one lesson for every Part I systems
module and explicit systems and mathematics capstones. The Notion page remains the long-term
knowledge specification; the local syllabus defines what to study and build in
which repository directory.

## Start learning

```sh
cd learning
uv sync
uv run jupyter lab
```

When using VS Code, JupyterLab does not need to run separately. Open
`learning/notebooks/00_slm_map.ipynb` and select
`learning/.venv/bin/python` as the kernel.

## Work on the brain

```sh
cd brain
uv sync
uv run pytest
```

## Work on the engine

```sh
cd engine
make run
```

The future Python/C++ integration will connect `brain/` to `engine/`; neither
the learning notebooks nor their visualization dependencies need to cross that
boundary.
