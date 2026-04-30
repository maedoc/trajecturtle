import subprocess
import sys

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error:")
        print(result.stderr)
        return False
    print(result.stdout)
    return True

def main():
    print("--- Starting Python Package Build and Check ---")
    
    # 1. Install build and twine
    if not run_command([sys.executable, "-m", "pip", "install", "build", "twine"]):
        sys.exit(1)

    # 2. Build the package
    if not run_command([sys.executable, "-m", "build"]):
        sys.exit(1)

    # 3. Check the distribution
    # We assume the dist/ directory exists after a successful build
    if not run_command(["twine", "check", "dist/*"]):
        print("Twine check failed! Please fix your metadata.")
        sys_exit(1)

    print("--- Build and Check Successful! ---")
    print("Your distribution files are ready in the 'dist/' directory.")

if __name__ == "__main__":
    main()
