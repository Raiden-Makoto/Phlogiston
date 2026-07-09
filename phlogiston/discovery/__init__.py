"""Discovery loop (pipeline §6-7): generate -> novelty -> stability gate ->
property screen -> multi-objective rank. Connects the CDVAE generator and the
trained Predictor by *usage* (not weights)."""

from phlogiston.discovery.feasibility import (
    FeasibilityReport,
    composition_feasibility,
    feasibility_filter,
)
from phlogiston.discovery.loop import (
    discover,
    format_report,
    load_generator,
    sample_candidates,
    save_candidates,
)
from phlogiston.discovery.rank import multi_objective_score, pareto_front, rank_candidates
from phlogiston.discovery.screen import PropertyScreen, load_predictor

__all__ = [
    "PropertyScreen",
    "load_predictor",
    "load_generator",
    "sample_candidates",
    "discover",
    "format_report",
    "save_candidates",
    "multi_objective_score",
    "pareto_front",
    "rank_candidates",
    "FeasibilityReport",
    "composition_feasibility",
    "feasibility_filter",
]
