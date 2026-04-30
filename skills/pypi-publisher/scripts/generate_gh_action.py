import argparse
import os

def generate_workflow(project_name, use_env=True):
    workflow_content = f"""name: Publish {project_name}

on:
  push:
    tags:
      - 'v*' # Trigger on tag pushes (e.t. v1.0.0)

jobs:
  build-n-publish:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write # REQUIRED for Trusted Publishing
    environment: {"pypi" if use_env else "none"}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build dependencies
        run: python -m pip install build

      - name: Build package
        run: python -m build

      - name: Publish package
        uses: pypa/gh-action-pypi-publish@release/v1
"""
    return workflow_content

def main():
    parser = argparse.ArgumentParser(description="Generate a GitHub Action workflow for PyPI publishing.")
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument("--no-env", action="store_false", dest="use_env", help="Do not use a GitHub environment")
    
    args = parser.parse_args()
    
    workflow_content = generate_workflow(args.project_name, use_env=args.use_env)
    
    # In a real skill, this would be used to write a file to the target repository.
    # For this skill, the script will print the content or save it to a local file 
    # for the user to copy.
    
    output_file = ".github/workflows/publish.yml"
    
    # Ensure the directory exists
    os.makedirs(".github/workflows", exist_ok=True)
    
    with open(output_file, "w") as f:
        f.write(workflow_content)
    
    print(f"Successfully generated {output_file}")
    print("--- Workflow Content ---")
    print(workflow_content)

if __name__ == "__main__":
    main()
