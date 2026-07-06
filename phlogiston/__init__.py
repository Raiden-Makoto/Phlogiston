"""Phlogiston: a crystal-structure ML framework for materials discovery.

Pipeline
--------
1. ``phlogiston.data.gnome``            -- load GNoME stable-material candidates.
2. ``phlogiston.data.materials_project`` -- fetch structural records from MP.
3. ``phlogiston.data.graph``            -- convert crystal structures to graphs.
4. ``phlogiston.data.dataset``          -- torch Dataset + batching.
5. ``phlogiston.models.cgcnn``          -- Crystal Graph Convolutional Neural Net.
6. ``phlogiston.train`` / ``discover``  -- train a property model, then screen.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
