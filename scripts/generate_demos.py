#!/usr/bin/env python3
"""Generate standalone HTML demos for the documentation site.

Usage:
    python scripts/generate_demos.py

Generates four demo widgets in docs/demos/:
    - wilson_cowan.html
    - fitzhugh_nagumo.html
    - mpr_bistable.html
    - mpr_limit_cycle.html
"""

import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase_plane_widget import PhasePlaneWidget


DEMOS = {
    "wilson_cowan": {
        "params": {
            "aee": 12.0, "aei": 10.0, "aie": 10.0, "aii": 2.0,
            "Pe": -2.0, "Pi": -8.0,
            "ke": 1.0, "ki": 1.0,
            "thetae": 4.0, "thetai": 4.0,
        },
        "x0": 0.3,
        "y0": 0.1,
        "t_max": 100.0,
    },
    "fitzhugh_nagumo": {
        "params": {"a": 0.7, "b": 0.8, "epsilon": 0.08, "I": 0.5},
        "x0": -1.0,
        "y0": -0.5,
        "t_max": 100.0,
    },
    "mpr_bistable": {
        "params": {"delta": 1.0, "eta_bar": -5.0, "J": 15.0, "I": 0.0},
        "x0": 0.1,
        "y0": -2.0,
        "t_max": 100.0,
    },
    "mpr_limit_cycle": {
        "params": {"delta": 1.0, "eta_bar": -2.0, "J": 18.0, "I": 0.0},
        "x0": 0.5,
        "y0": -1.0,
        "t_max": 100.0,
    },
}


def main():
    out_dir = Path(__file__).parent.parent / "docs" / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, config in DEMOS.items():
        model_name = name.split("_")[0] if name.startswith("mpr_") else name
        widget = PhasePlaneWidget(model_name=model_name)

        for k, v in config["params"].items():
            widget.params[k] = v
        widget.x0 = config["x0"]
        widget.y0 = config["y0"]
        widget.t_max = config["t_max"]

        out_path = out_dir / f"{name}.html"
        widget.to_standalone_html(str(out_path), title=f"Phase Plane: {name}")
        print(f"Generated {out_path.name} ({out_path.stat().st_size:,} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
