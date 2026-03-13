# Installation

## From source

Clone the repository:

```bash
git clone https://github.com/jpsferreira/hyper-surrogate.git
cd hyper-surrogate
```

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

To also install the ML extras (PyTorch):

```bash
uv sync --all-groups --extra ml
```

Run tests:

```bash
uv run pytest
```

## From PyPI

```bash
pip install hyper-surrogate
```

```bash
uv add hyper-surrogate
```

With ML support:

```bash
pip install hyper-surrogate[ml]
```
