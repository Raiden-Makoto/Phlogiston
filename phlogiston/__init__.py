"""Phlogiston: a crystal-structure ML framework for materials discovery.

Pipeline
--------
1. ``phlogiston.data.gnome``             -- load GNoME stable-material candidates.
2. ``phlogiston.data.materials_project`` -- fetch structural records from MP.
3. ``phlogiston.data.graph``             -- convert crystal structures to graphs.
4. ``phlogiston.data.dataset``           -- torch Dataset + masked multi-task batching.
5. ``phlogiston.models``                 -- shared equivariant encoder + predictor,
                                            synthesizability, and CDVAE generator heads.
6. ``phlogiston.train``                  -- predictor / synth / CDVAE trainers.
7. ``phlogiston.discovery``              -- generate, screen, gate, rank, persist.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
