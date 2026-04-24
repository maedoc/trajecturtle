"""Interactive phase plane widget for neural mass models."""

from .models import BaseModel, FitzHughNagumo, MPRModel, WilsonCowan, MODEL_REGISTRY
from .widget import PhasePlaneWidget

__all__ = ["PhasePlaneWidget", "BaseModel", "WilsonCowan", "FitzHughNagumo", "MODEL_REGISTRY"]
__version__ = "0.1.0"
