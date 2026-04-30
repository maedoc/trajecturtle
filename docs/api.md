# API Reference

## `PhasePlaneWidget`

::: tvb_phaseplane.widget.PhasePlaneWidget
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - to_standalone_html
        - run_sweep

## Model Base Class

::: tvb_phaseplane.models.BaseModel
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

::: tvb_phaseplane.models.WilsonCowan
    handler: python
    options:
      show_root_heading: true
      show_source: true

## FitzHugh-Nagumo

::: tvb_phaseplane.models.FitzHughNagumo
    handler: python
    options:
      show_root_heading: true
      show_source: true

## MPR (Quadratic Integrate-and-Fire)

::: tvb_phaseplane.models.MPRModel
    handler: python
    options:
      show_root_heading: true
      show_source: true
      members:
        - find_fixed_points
        - compute_saddle_node_boundary
