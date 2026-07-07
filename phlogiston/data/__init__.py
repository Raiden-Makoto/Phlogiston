"""Data loading for Phlogiston.

Phase 1 focuses on acquiring the GNoME dataset (``phlogiston.data.gnome``).
Later phases add Materials Project structure records and crystal-graph
featurization.
"""

from phlogiston.data import dataset, gnome, graph, materials_project, properties

__all__ = ["gnome", "materials_project", "properties", "graph", "dataset"]
