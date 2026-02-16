"""
Algorithm selection based on problem structure.
"""
from typing import Tuple, List
from data_structures import StructureReport


class AlgorithmSelector:
    """Select optimization algorithm based on problem structure."""
    
    def select(self, structure: StructureReport) -> Tuple[str, str, List[str]]:
        """
        Select solver based on structure report.
        
        Args:
            structure: StructureReport from inference
        
        Returns:
            (solver_name, reason, alternatives_considered)
        """
        alternatives = []
        
        # Handle constrained problems first
        if structure.has_constraints:
            return self._select_constrained(structure)
        else:
            return self._select_unconstrained(structure)
    
    def _select_constrained(self, structure: StructureReport) -> Tuple[str, str, List[str]]:
        """Select solver for constrained problems."""
        alternatives = ["SLSQP", "Interior Point", "Augmented Lagrangian"]
        
        # Check LICQ
        if structure.licq_status == "VIOLATED":
            return (
                "Augmented Lagrangian",
                "LICQ violated, penalty approach needed",
                alternatives
            )
        
        # Convex with linear constraints
        if (structure.has_linear_constraints and 
            structure.convexity_status == "CONVEX"):
            return (
                "Interior Point",
                "Convex with linear constraints",
                alternatives
            )
        
        # General constrained problem with LICQ
        return (
            "SLSQP",
            "General constrained problem with LICQ",
            alternatives
        )
    
    def _select_unconstrained(self, structure: StructureReport) -> Tuple[str, str, List[str]]:
        """Select solver for unconstrained problems."""
        alternatives = ["L-BFGS", "Gradient Descent", "Trust-Region Newton", "Nelder-Mead"]
        
        # Convex problems
        if structure.convexity_status == "CONVEX":
            if structure.condition_number_median < 100:
                return (
                    "L-BFGS",
                    "Well-conditioned convex problem",
                    alternatives
                )
            else:
                return (
                    "Gradient Descent",
                    "Ill-conditioned convex, robust choice",
                    alternatives
                )
        
        # Nonconvex problems
        elif structure.convexity_status == "NONCONVEX":
            if structure.is_smooth:
                return (
                    "Trust-Region Newton",
                    "Nonconvex smooth, second-order method",
                    alternatives
                )
            else:
                return (
                    "Nelder-Mead",
                    "Nonconvex non-smooth, derivative-free",
                    alternatives
                )
        
        # Uncertain
        else:
            return (
                "L-BFGS",
                "Uncertain structure, robust quasi-Newton (will use multi-start)",
                alternatives
            )


def select_algorithm(structure: StructureReport) -> Tuple[str, str, List[str]]:
    """
    Main entry point for algorithm selection.
    
    Args:
        structure: StructureReport from inference
    
    Returns:
        (solver_name, reason, alternatives_considered)
    """
    selector = AlgorithmSelector()
    return selector.select(structure)
