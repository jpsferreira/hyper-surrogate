# Contributing to `hyper-surrogate`

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

You can contribute in many ways:

# Types of Contributions

## Report Bugs

Report bugs at https://github.com/jpsferreira/hyper-surrogate/issues

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs.
Anything tagged with "bug" and "help wanted" is open to whoever wants to implement a fix for it.

## Implement Features

Look through the GitHub issues for features.
Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

## Write Documentation

hyper-surrogate could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue at https://github.com/jpsferreira/hyper-surrogate/issues.

If you are proposing a new feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

# Get Started!

Ready to contribute? Here's how to set up `hyper-surrogate` for local development.
Please note this documentation assumes you already have [uv](https://docs.astral.sh/uv/) and `Git` installed and ready to go.

1. Fork the `hyper-surrogate` repo on GitHub.

2. Clone your fork locally:

```bash
cd <directory_in_which_repo_should_be_created>
git clone git@github.com:YOUR_NAME/hyper-surrogate.git
```

3. Now we need to install the environment. Navigate into the directory

```bash
cd hyper-surrogate
```

If you are using `pyenv`, select a version to use locally. (See installed versions with `pyenv versions`)

```bash
pyenv local <x.y.z>
```

Then, install the environment with all development dependencies:

```bash
uv sync --all-groups --extra ml
```

4. Install pre-commit to run linters/formatters at commit time:

```bash
uv run pre-commit install
```

5. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Branch prefixes: `feat/` for features, `fix/` for bug fixes, `chore/` for maintenance.

Now you can make your changes locally.

6. Don't forget to add test cases for your added functionality to the `tests` directory.

7. When you're done making changes, check that your changes pass the formatting tests.

```bash
make check
```

Now, validate that all unit tests are passing:

```bash
make test
```

To run only fast tests (skipping slow ones):

```bash
uv run pytest -m "not slow"
```

To run slow tests explicitly:

```bash
uv run pytest -m slow
```

8. Build and verify the documentation:

```bash
make docs-test   # check docs build without errors
make docs         # serve docs locally at http://127.0.0.1:8000
```

9. Commit your changes and push your branch to GitHub:

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

10. Submit a pull request through the GitHub website.

# Code Style

- **Formatter/linter:** [ruff](https://docs.astral.sh/ruff/) with 120-character line length
- **Type checking:** [mypy](https://mypy-lang.org/) in strict mode on `hyper_surrogate/`
- **Imports:** Use `from __future__ import annotations` in all modules
- **Testing:** [pytest](https://pytest.org/) with `@pytest.mark.slow` for expensive tests

All checks run via `make check` (ruff, mypy, deptry) and `make test` (pytest with coverage).

# Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
