# Python Release Checklist

Use this checklist to ensure a smooth and error-free release to PyPI.

## Pre-Release (Local)

- [ ] **Version Bump**: Increment the version number in `pyproject.toml` (and `__init__.py` if applicable).
- [ ] **Update Changelog**: Add a new entry in `CHANGELOG.md` describing the changes.
- [ ] **Run Tests**: Ensure all unit and integration tests pass locally (`pytest`).
- [ ] **Linting & Formatting**: Run `ruff`, `black`, or `isort` to ensure code quality.
- [ ] **Build Distribution**: Run `python -m build` to generate `sdist` and `wheel`.
- [ ] **Metadata Audit**: Run `twine check dist/*` to verify that your metadata is valid and renders correctly.
- [ ] **Dependency Check**: Ensure all required dependencies are correctly listed in `pyproject.toml`.

## TestPyPI (The Sandbox)

- [ ] **Upload to TestPyPI**: Use `twine upload --repository testpypi dist/*`.
- [ ] **Verify Installation**: Install your package from TestPyPI in a clean virtual environment:
    `pip install --index-url https://test.pypi.org/simple/ your-package-name`
- [ ] **Check Package Page**: Visit the package page on TestPyPI to ensure the README renders correctly.

## Production Release (PyPI)

- [ ] **Git Tagging**: Create a new Git tag for the release (e.g., `git tag v1.0.0` then `git push origin v1.0.0`).
- [ ] **Final Upload**: Execute the upload to PyPI (manual `twine` or automated GitHub Action).
- [ ] **Verify PyPI**: Check the official PyPI page for any errors in rendering or metadata.
- [ ] **Announce**: Update your website, mailing list, or social media if necessary.
