"""Assemble browser sources without npm or a network dependency.

The bundle intentionally has no imports so anywidget, file:// and offline
HTML exports use exactly the same code. Edit frontend/*.js, then rebuild.
"""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1] / "src/tvb_phaseplane"
SOURCES = (
    "vendor", "drawing-support", "engine", "coordinates", "experiment", "controller",
    "components", "plot-drawing", "control-panels", "plot-panels", "editor-panel", "presentations", "view",
)


def build(check=False):
    content = "\n".join((ROOT / "frontend" / f"{name}.js").read_text().rstrip() for name in SOURCES) + "\n"
    target = ROOT / "static/widget.js"
    if check:
        if target.read_text() != content:
            raise SystemExit("Frontend bundle is stale; run python scripts/build_frontend.py")
    else:
        target.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(parser.parse_args().check)
