"""Data loading for Phlogiston.

Phase 1 focuses on acquiring the GNoME dataset (``phlogiston.data.gnome``).
Later phases add Materials Project structure records and crystal-graph
featurization.
"""

from phlogiston.data import gnome
from phlogiston.data import materials_project
from phlogiston.data import properties
from phlogiston.data import graph
from phlogiston.data import dataset

__all__ = ["gnome", "materials_project", "properties", "graph", "dataset"]
