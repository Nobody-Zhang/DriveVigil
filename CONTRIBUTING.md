# Contributing

Thanks for your interest in this project! It is the **award-winning competition
code** for the Huawei Cloud track of the 18th Challenge Cup (a fatigue /
distracted-driving detection system with cloud-edge collaboration). It is shared
primarily as a **reference implementation and archive**, not as an actively
developed product — but issues, fixes, documentation improvements, and questions
are very welcome.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report bugs** — open an [issue](../../issues) using the bug-report template.
- **Improve docs** — typos, clearer explanations, translation fixes (the README
  is bilingual: [English](README.md) / [中文](README_zh.md)).
- **Fix or refactor** hand-maintained code (see the boundaries below).
- **Ask questions** — open an issue; there's no wrong question about how the
  system works.

## Development setup

```bash
bash scripts/setup_env.sh        # create .venv, install requirements, copy .env.example -> .env
bash scripts/download_assets.sh  # pull large binaries (weights, engines, videos) from GitHub Releases v1.0
```

Large binaries (~40 weights, TensorRT engines, OpenVINO IR, sample videos,
wheels) are **not in git** — they live in [GitHub Releases v1.0](../../releases)
and are documented in [docs/ASSETS.md](docs/ASSETS.md). Most modules will not run
until `download_assets.sh` has completed and `.env` is filled in.

## Linting and formatting

[ruff](https://docs.astral.sh/ruff/) is the only style gate (config in
`pyproject.toml`, enforced by CI on Python 3.8):

```bash
ruff check .            # lint   (ruff check --fix .  to auto-fix)
ruff format --check .   # format (ruff format .       to apply)
```

Or install the git hook so this runs automatically on every commit:

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

## Tests

A small pytest suite covers the **pure, hardware-free logic** — most importantly
the per-frame classifier `YOLO_Status` (the core IP) and its geometry helpers.
It needs no GPU, no cloud credentials, and no downloaded assets:

```bash
pip install -r requirements-dev.txt
pytest                  # or: pytest -v
```

If you change the detection-class ordering, the category mapping, or the driver-
selection heuristics in `cloud/preliminary/yolo/status.py`, **update the tests in
`tests/` to match** — those tests are the executable specification of that
contract.

## Things to know before you open a PR

### Python version split

- **Repo tooling / CI target Python 3.8.**
- **Cloud code deploys to ModelArts on Python 3.7**
  (`pytorch_1.8.0-cuda_10.2-py_3.7-ubuntu_18.04-x86_64`).

➡️ Do **not** introduce 3.8+-only syntax (e.g. the walrus operator in a
positional-only context, `dict` merge `|`, `functools.cached_property` quirks)
into anything under `cloud/` that ships to ModelArts.

### Vendored / third-party trees — do not edit or lint

These are upstream copies kept verbatim; they are excluded from ruff in
`pyproject.toml`. Please don't reformat or "clean them up":

- every `**/yolo/models/`, `**/yolo/utils/`, and `**/yolo/export.py` (Ultralytics YOLOv5)
- `edge/cloud_finetune/yolov5/` (full Ultralytics YOLOv5)
- `edge/voice/huaweicloud-python-sdk-sis-1.8.1/` (Huawei SIS SDK)
- every `apig_sdk/` (Huawei API Gateway SDK)
- the entire `edge/deepstream/` tree (vendored DeepStream-Yolo fork + Jetson
  competition scripts — see [edge/deepstream/COMPETITION.md](edge/deepstream/COMPETITION.md))
- the mtcnn `utils/`, and `edge/frontend/` (prebuilt Vue SPA)

ruff additionally ignores `E402`, `F403`, and `F405` repo-wide — these flag the
codebase's intentional patterns (appending to `sys.path` before importing
vendored packages, and `from ... import *` against vendored YOLO/MTCNN utils),
not real defects. Roughly 257 `.py` files are tracked but only ~60 are
hand-maintained and linted.

### Near-duplicate variants — mirror your changes

Some trees are intentional near-duplicates because each is a separately
deployable unit:

- `cloud/preliminary/` (canonical, best score **0.9741**) and
  `cloud/semifinal/` (score 0.8807) share the same divide-and-conquer
  algorithm — only thresholds differ.
- `edge/mtcnn/` and `edge/ota/mtcnn_landmarks/` are near-identical.

If you change shared algorithm logic, mirror the change across the relevant
copies (and say so in your PR).

### Competition scratch files

`edge/deepstream/` contains many one-off experiment scripts
(`test*.py`, `try*.py`, `gpt*.py`). These are kept for historical reference and
are **not** maintained. The real entry points are documented in
[edge/deepstream/COMPETITION.md](edge/deepstream/COMPETITION.md). Please don't
build new features on top of the scratch scripts.

### Secrets

Never commit credentials. All AK/SK, tokens, endpoints, and API keys are read
from environment variables via `.env` (template: `.env.example`). `.env` is
gitignored. See [SECURITY.md](SECURITY.md).

## Commit & PR conventions

- Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes —
  the history already uses `feat:`, `fix:`, `docs:`, `refactor:`, `style:`,
  `chore:`.
- Keep PRs focused; describe **what** changed and **why**.
- Make sure `ruff check .`, `ruff format --check .`, and `pytest` pass.
- Use the [pull-request template](.github/PULL_REQUEST_TEMPLATE.md) checklist.

Thank you! 🙏
