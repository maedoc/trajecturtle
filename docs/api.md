# API Reference

## `PhasePlaneWidget`

::: phase_plane_widget.widget.PhasePlaneWidget
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - to_standalone_html
        - run_sweep

## Model Base Class

::: phase_plane_widget.models.BaseModel
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - f
        - compute_trajectory
        - compute_phase_plane_data
        - detect_regime
        - run_sweep

## Wilson-Cowan

::: phase_plane_widget.models.WilsonCowan
    handler: python
    options:
      show_root_heading: true
      show_source: true

## FitzHugh-Nagumo

::: phase_plane_widget.models.FitzHughNagumo
    handler: python
    options:
      show_root_heading: true
      show_source: true

## MPR (Quadratic Integrate-and-Fire)

::: phase_plane_widget.models.MPRModel
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - find_fixed_points
        - compute_saddle_node_boundary
