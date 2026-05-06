"""Tests for TikZ / PGFPlots export."""

import pathlib

import pytest
from tvb_phaseplane import PhasePlaneWidget
from tvb_phaseplane.models import MODEL_REGISTRY


@pytest.fixture
def wc_widget():
    w = PhasePlaneWidget(model_name="wilson_cowan")
    w.params = {"aee": 10.0, "aei": 10.0, "aie": 10.0, "aii": 2.0, "Pe": -2.0, "Pi": -8.0, "ke": 1.0, "ki": 1.0, "thetae": 4.0, "thetai": 4.0}
    w.xlim = [-0.2, 1.2]
    w.ylim = [-0.2, 1.2]
    w.display = [0, 1]
    w.state_names = ["E", "I"]
    return w


class TestTikzExport:

    def test_empty_export_creates_file(self, wc_widget, tmp_path):
        out = tmp_path / "phase_plane.tex"
        path = wc_widget.export_tikz(out)
        assert pathlib.Path(path).exists()
        content = out.read_text()
        assert r"\documentclass[border=5pt]{standalone}" in content
        assert r"\usepackage{pgfplots}" in content
        assert r"\pgfplotsset{compat=1.17}" in content
        assert r"\begin{document}" in content
        assert r"\end{document}" in content
        assert "xlabel=$E$" in content
        assert "ylabel=$I$" in content

    def test_empty_export_has_no_plots(self, wc_widget, tmp_path):
        wc_widget.show_vector_field = False
        out = tmp_path / "phase_plane.tex"
        wc_widget.export_tikz(out)
        content = out.read_text()
        assert r"\addplot[" not in content
        assert r"\draw[" not in content

    def test_vector_field_draws(self, wc_widget, tmp_path):
        wc_widget.show_vector_field = True
        out = tmp_path / "vf.tex"
        wc_widget.export_tikz(out)
        content = out.read_text()
        assert r"\draw[-stealth, gray]" in content
        assert "(axis cs:" in content

    def test_trajectory_coords(self, wc_widget, tmp_path):
        wc_widget.show_trajectory = True
        wc_widget.trajectory = [[0.0, 0.1, 0.2], [0.1, 0.15, 0.25], [0.2, 0.2, 0.3]]
        out = tmp_path / "traj.tex"
        wc_widget.export_tikz(out)
        content = out.read_text()
        assert r"\addplot[green!60!black" in content
        assert "coordinates {" in content
        assert "(0.100000,0.200000)" in content
        assert "(0.150000,0.250000)" in content

    def test_nullcline_coords(self, wc_widget, tmp_path):
        wc_widget.show_nullclines = True
        wc_widget.nullcline_x = [[0.0, 0.5], [0.5, 0.6]]
        wc_widget.nullcline_y = [[0.0, 0.3], [0.5, 0.4]]
        out = tmp_path / "nc.tex"
        wc_widget.export_tikz(out)
        content = out.read_text()
        assert r"\addplot[blue" in content
        assert r"\addplot[red" in content
        assert "(0.000000,0.500000)" in content
        assert "(0.500000,0.400000)" in content

    def test_fixed_points_rendering(self, wc_widget, tmp_path):
        wc_widget.show_fixed_points = True
        wc_widget.fixed_points = [
            [0.1, 0.2, "stable_node"],
            [0.3, 0.4, "stable_focus"],
            [0.5, 0.6, "unstable_node"],
            [0.7, 0.8, "saddle"],
        ]
        out = tmp_path / "fp.tex"
        wc_widget.export_tikz(out)
        content = out.read_text()
        assert "circle" in content
        assert "diamond" in content
        assert "green!70!black" in content
        assert "red!70!black" in content
        assert "purple" in content
        assert content.count("inner sep=0.8pt") == 1
        assert "(axis cs:0.1" in content

    def test_param_annotation(self, wc_widget, tmp_path):
        out = tmp_path / "annot.tex"
        wc_widget.export_tikz(out)
        content = out.read_text()
        assert "wilson_cowan" in content or "wilson\\_cowan" in content
        assert "aee=" in content

    def test_default_filename(self, wc_widget):
        path = wc_widget.export_tikz()
        assert pathlib.Path(path).name == "phase_plane.tex"
        pathlib.Path(path).unlink()

    def test_all_models_export(self):
        for name in MODEL_REGISTRY:
            w = PhasePlaneWidget(model_name=name)
            w.show_vector_field = True
            w.show_nullclines = True
            w.show_trajectory = True
            w.show_fixed_points = True
            nullcline = []
            for i in range(21):
                nullcline.append(
                    [
                        (float(w.xlim[0]) + (float(3.0/20.0) * i)),
                        (float(w.ylim[0]) + (float(3.0/20.0) * i)),
                    ]
                )
            w.nullcline_x = nullcline
            w.nullcline_y = nullcline
            w.trajectory = [
                [float(i), (float(j) + (float(3.0/20.0) * i)), 0.0, 0.0]
                for j in range(2)
                for i in range(21)
            ]
            w.fixed_points = [
                [0.1, 0.2, "stable_node"],
            ]
            out = f"/tmp/test_tikz_{name}.tex"
            p = w.export_tikz(out)
            assert pathlib.Path(p).exists()
            content = pathlib.Path(p).read_text()
            assert r"\documentclass" in content
            assert r"\end{document}" in content
            pathlib.Path(p).unlink()

    @pytest.mark.skipif(
        not (pathlib.Path("/usr/bin/pdflatex").exists() or pathlib.Path("/usr/local/bin/pdflatex").exists()),
        reason="pdflatex not installed",
    )
    def test_pdflatex_compiles(self, wc_widget, tmp_path):
        import subprocess

        wc_widget.show_vector_field = True
        wc_widget.trajectory = [[0.0, 0.1, 0.2], [0.1, 0.15, 0.25]]
        wc_widget.show_trajectory = True
        out = tmp_path / "compile_test.tex"
        wc_widget.export_tikz(out)
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(out)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        # Accept a PDF even if old PGFPlots versions warn on compat=1.17
        assert (tmp_path / "compile_test.pdf").exists(), "pdflatex did not produce a PDF"
