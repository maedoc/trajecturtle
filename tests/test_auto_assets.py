"""Tests for AUTO2000 WASM asset transport, standalone HTML embedding, and bifurcation traits."""

import base64
import json
import pathlib
import re

from tvb_phaseplane import PhasePlaneWidget
from tvb_phaseplane.widget import ExperimentView

_STATIC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "tvb_phaseplane" / "static"


def test_bifurcation_and_auto_traits_exist_and_synced():
    w = PhasePlaneWidget()
    expected_defaults = {
        "bifurcation_branches": [],
        "bifurcation_special_points": [],
        "bifurcation_config": {},
        "bifurcation_running": False,
        "bifurcation_metadata": {},
        "bifurcation_presets": [],
        "auto_wasm_url": "",
        "auto_worker_url": "",
    }
    traits = w.traits()
    for name, default_val in expected_defaults.items():
        assert name in traits, f"Missing trait {name}"
        assert traits[name].metadata.get("sync") is True, f"Trait {name} is not synced"
        assert getattr(w, name) == default_val


def test_request_auto_assets_custom_msg(monkeypatch):
    w = PhasePlaneWidget()
    sent = []

    def fake_send(content, buffers=None):
        sent.append((content, buffers))

    monkeypatch.setattr(w, "send", fake_send)
    w._on_custom_msg(w, {"type": "request_auto_assets"}, [])

    assert len(sent) == 1
    content, buffers = sent[0]
    assert content == {"type": "auto_assets"}
    assert len(buffers) == 2
    assert buffers[0] == (_STATIC_DIR / "auto2000_core.wasm").read_bytes()
    assert buffers[1] == (_STATIC_DIR / "auto2000-worker.js").read_bytes()


def test_to_standalone_html_include_auto_false(tmp_path):
    w = PhasePlaneWidget()
    out = tmp_path / "lightweight.html"
    w.to_standalone_html(out)
    html = out.read_text(encoding="utf-8")
    assert re.search(r"globalThis\.__PPW_AUTO_ASSETS__\s*=", html) is None
    assert '"auto_wasm_url"' in html


def test_to_standalone_html_include_auto_true(tmp_path):
    w = PhasePlaneWidget()
    out = tmp_path / "with_auto.html"
    w.to_standalone_html(out, include_auto=True)
    html = out.read_text(encoding="utf-8")
    m = re.search(r"globalThis\.__PPW_AUTO_ASSETS__\s*=\s*(\{.*?\});\n", html)
    assert m is not None, "Embedded __PPW_AUTO_ASSETS__ not found"

    payload = json.loads(m.group(1))
    assert base64.b64decode(payload["wasmBase64"]) == (_STATIC_DIR / "auto2000_core.wasm").read_bytes()
    assert payload["workerSource"] == (_STATIC_DIR / "auto2000-worker.js").read_text(encoding="utf-8")


def test_bifurcation_panel_and_presentation_registered():
    assert "bifurcation" in PhasePlaneWidget.PANEL_NAMES
    assert "bifurcation" in PhasePlaneWidget.presentation.values
    assert "bifurcation" in ExperimentView.presentation.values
    w = PhasePlaneWidget(panels=["bifurcation", "phase_plane"], presentation="bifurcation")
    assert w.panels == ["bifurcation", "phase_plane"]
    assert w.presentation == "bifurcation"


def test_export_bifurcation_tikz(tmp_path):
    w = PhasePlaneWidget(model_name="fitzhugh_nagumo")
    w.bifurcation_config = {"param1": "I", "observable": "v"}
    w.bifurcation_branches = [
        {
            "id": "b1",
            "type": "EP",
            "param1": "I",
            "points": [
                {"p": 0.0, "u": [-1.2, -0.5], "stable": True},
                {"p": 0.3, "u": [-0.8, -0.1], "stable": True},
                {"p": 0.4, "u": [-0.5, 0.1], "stable": False},
                {"p": 0.6, "u": [-0.2, 0.3], "stable": False},
            ],
        },
        {
            "id": "b2",
            "type": "LC",
            "param1": "I",
            "points": [
                {"p": 0.3, "u_max": [0.5, 0.2], "u_min": [-1.0, -0.4], "stable": True},
                {"p": 0.5, "u_max": [1.8, 0.7], "u_min": [-1.5, -0.8], "stable": True},
            ],
        },
    ]
    w.bifurcation_special_points = [
        {"id": "HB1", "ty": "HB", "label": "HB", "p": 0.3, "u": [-0.8, -0.1]},
    ]

    tex_str = w.export_bifurcation_tikz()
    assert r"\begin{axis}" in tex_str
    assert r"\addplot[blue!75!black, thick, no marks]" in tex_str
    assert r"\addplot[gray, thick, dashed, no marks]" in tex_str
    assert r"\addplot[green!60!black, very thick, no marks]" in tex_str
    assert "HB" in tex_str

    out = tmp_path / "bif.tex"
    returned = w.export_bifurcation_tikz(out)
    assert returned == str(out)
    assert out.read_text(encoding="utf-8") == tex_str
