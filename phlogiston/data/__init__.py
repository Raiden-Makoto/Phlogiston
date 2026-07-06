"""Data loading for Phlogiston.

Phase 1 focuses on acquiring the GNoME dataset (``phlogiston.data.gnome``).
Later phases add Materials Project structure records and crystal-graph
featurization.
"""

from phlogiston.data import gnome

__all__ = ["gnome"]
