# pyproject.toml Configuration Guide

A modern Python package requires a `pyproject.toml` file. This file is the single source of truth for your project metadata and build requirements.

## Essential Fields

For a successful PyPI publication, ensure the following `[project]` fields are present:

- `name`: A unique name for your package (check PyPI for availability). Use hyphens for the distribution name (e.g., `my-awesome-package`).
- `version`: Use Semantic Versioning (e.g., `0.1.0`).
- `description`: A short, one-line summary.
- `readme`: Reference your README file (e.g., `readme = "README.md"`).
- `requires-python`: Specify the supported Python versions (e.g., `requires-python = ">=3.8"`).
- `license`: Specify the license (e.g., `license = {text = "MIT"}`).
- `authors`: List the authors with email addresses.
- `dependencies`: A list of runtime dependencies.

## Build System Configuration

Every `pyproject.toml` must include a `[build-system]` table. This tells `pip` which tool to use to create the distribution files.

### Using Hatchling (Recommended)
```toml
[build-string]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Using Setuptools
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

## Example Template

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "example-package"
version = "0.1.0"
description = "An example package for PyPI publishing"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "requests >= 2.28.0",
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[tool.hatch.build.targets.wheel]
packages = ["src/example_package"]
```
