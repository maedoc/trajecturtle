"""Assemble browser sources without npm or a network dependency.

The bundle intentionally has no imports so anywidget, file:// and offline
HTML exports use exactly the same code. Edit frontend/*.js, then rebuild.
"""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1] / "src/tvb_phaseplane"
SOURCES = (
    "vendor", "drawing-support", "models-core", "engine", "coordinates", "experiment", "auto-client", "controller",
    "components", "plot-drawing", "control-panels", "plot-panels", "bifurcation-panel", "editor-panel", "research-views", "navigator-view", "presentations", "view",
)
WORKER_SOURCES = (
    "worker-shim", "vendor", "models-core", "vendor-auto2000", "auto-worker",
)
TARGETS = (
    ("static/widget.js", SOURCES),
    ("static/auto2000-worker.js", WORKER_SOURCES),
)


def _assemble(sources):
    return "\n".join((ROOT / "frontend" / f"{name}.js").read_text().rstrip() for name in sources) + "\n"


def build(check=False):
    for rel_target, sources in TARGETS:
        content = _assemble(sources)
        target = ROOT / rel_target
        if check:
            if not target.exists() or target.read_text() != content:
                raise SystemExit(f"Frontend bundle {rel_target} is stale; run python scripts/build_frontend.py")
        else:
            target.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(parser.parse_args().check)
