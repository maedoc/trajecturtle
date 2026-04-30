---
name: pypi-publisher
description: Automates the building, checking, and publishing of Python packages to PyPI and TestPyPI. Use when Codex needs to prepare a Python package for distribution, configure pyproject.toml, set up GitHub Actions for trusted publishing, or verify a package release.
---

# PyPI Publisher

This skill provides the instructions and automation tools to move a Python project from local development to a published state on PyPI.

## Core Workflow

When tasked with publishing a package, follow this sequence:

1.  **Validate Configuration**: Ensure `pyproject.toml` is correctly configured for the chosen build backend.
2.  **Build Distribution**: Generate both `sdist` and `wheel` formats using `python -m build`.
3.  **Check Metadata**: Verify the package integrity with `twine check`.
4.  **Test Deployment**: Upload the package to **TestPyPI** to verify the installation experience.
5.  **Production Release**: Execute the final upload to **PyPI** (ideally via automated Trusted Publishing).

## Automation Tools

Use the scripts in `scripts/` to automate repetitive tasks:

- `scripts/build_and_check.py`: Executes the build and runs `twine check` in a single command.
- `scripts/generate_gh_action.py`: Creates a `.github/workflows/publish.yml` file configured for **Trusted Publishing (OIDC)**.

## Essential References

For deep dives into specific parts of the process, refer to:

- [pyproject.toml Configuration](references/pyproject_toml_template.md): How to define metadata, dependencies, and build systems.
- [Trusted Publishing Guide](references/trusted_publishing_guide.md): Detailed steps for setting up OIDC on GitHub and PyPI.
- [Release Checklist](references/release_checklist.md): A final audit before you push the button.

## Usage Examples

**Task**: "Help me prepare my package for PyPI."
**Action**: Use the `build_and_check.py` script to ensure the current project is ready.

**Task**: "Set up a GitHub Action for my Python package release."
**Action**: Run `scripts/generate_gh_action.py` to scaffold the workflow file.
