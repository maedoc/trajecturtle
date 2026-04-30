# Trusted Publishing Guide (OIDC)

**Trusted Publishing** is the modern, secure way to publish from GitHub Actions to PyPI. It removes the need to store long-lived API tokens in GitHub Secrets. Instead, PyPI verifies that the incoming request is coming from a specific GitHub repository and workflow.

## Step 1: Configure PyPI

Before your workflow can publish, you must register your GitHub repository as a trusted publisher on PyPI.

1.  Log in to [PyPI](https://pypi.org/).
2.  Navigate to your project page (or create a "pending publisher" if it's a new project).
3.  Go to **Settings** -> **Publishing**.
4.  Click **Add a new publisher**.
5.  Provide the following details:
    *   **Owner**: Your GitHub username or organization.
    *   **Repository**: The name of your repository.
    *   **Workflow name**: The name of your YAML file (e.g., `publish.yml`).
    *   **Environment**: (Optional) The name of the GitHub Environment you use (e.g., `pypi`).

## Step 2: Configure GitHub Actions

Your workflow must have the correct permissions to request an OIDC token from GitHub.

### Required Permissions

In your `.github/workflows/publish.yml`, you **must** include the `id-token: write` permission. Without this, the workflow cannot authenticate with PyPI.

```yaml
permissions:
  contents: write
  id-token: write
```

### Example Workflow Structure

Use the `pypa/gh-action-pypi-publish` action. It is designed for trusted publishing.

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*' # Trigger only on version tags

jobs:
  build-n-publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write # Essential for Trusted Publishing
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build dependencies
        run: python -m pip install build

      - name: Build package
        run: python -int build

      - name: Publish package
        uses: pypa/gh-action-pypi-publish@release/v1
```

## Why use Trusted Publishing?

*   **No Secrets to Leak**: There are no API tokens stored in GitHub Secrets that can be stolen.
*   **Automatic Expiration**: The OIDC token is short-lived and only valid for the duration of the workflow.
*   **Reduced Attack Surface**: Even if someone gains access to your repository, they cannot use your PyPI credentials outside of the authorized workflow.
