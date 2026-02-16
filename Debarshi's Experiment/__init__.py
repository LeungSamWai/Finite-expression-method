"""
OptimAI: Structure-Aware Optimization with Formal Verification
"""

from .optimai import optimize, OptimAI
from .data_structures import (
    OptimizationProblem,
    StructureReport,
    SolverResult,
    VerificationResult,
    ReasoningTrace
)
from .parsing import parse_problem
from .structure_inference import infer_structure
from .algorithm_selection import select_algorithm
from .solver_execution import solve_problem
from .verification import verify_solution

__version__ = "0.1.0"
__all__ = [
    "optimize",
    "OptimAI",
    "OptimizationProblem",
    "StructureReport",
    "SolverResult",
    "VerificationResult",
    "ReasoningTrace",
    "parse_problem",
    "infer_structure",
    "select_algorithm",
    "solve_problem",
    "verify_solution"
]
