"""
Data structures for OptimAI optimization system.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np
from sympy import Symbol
from sympy.core.expr import Expr as SymbolicExpression


@dataclass
class OptimizationProblem:
    """Canonical representation of an optimization problem."""
    objective: SymbolicExpression
    variables: List[Symbol]
    inequality_constraints: List[SymbolicExpression] = field(default_factory=list)
    equality_constraints: List[SymbolicExpression] = field(default_factory=list)
    bounds: Dict[Symbol, Tuple[float, float]] = field(default_factory=dict)
    initial_point: Optional[np.ndarray] = None
    
    @property
    def n_vars(self) -> int:
        """Number of decision variables."""
        return len(self.variables)
    
    @property
    def n_inequality_constraints(self) -> int:
        """Number of inequality constraints."""
        return len(self.inequality_constraints)
    
    @property
    def n_equality_constraints(self) -> int:
        """Number of equality constraints."""
        return len(self.equality_constraints)
    
    @property
    def has_constraints(self) -> bool:
        """Check if problem has constraints."""
        return (self.n_inequality_constraints > 0 or 
                self.n_equality_constraints > 0 or 
                len(self.bounds) > 0)


@dataclass
class StructureReport:
    """Report of problem structure analysis."""
    # Convexity
    convexity_status: str  # "CONVEX", "NONCONVEX", "UNCERTAIN"
    convexity_confidence: float  # 0.0 to 1.0
    convexity_method: str  # "symbolic", "sampling"
    convexity_ci: Optional[Tuple[float, float]] = None  # Confidence interval if sampling
    
    # Smoothness
    lipschitz_lower_bound: float = 0.0
    condition_number_median: float = 1.0
    is_smooth: bool = True
    
    # Constraints
    has_constraints: bool = False
    has_linear_constraints: bool = False
    licq_status: str = "UNKNOWN"  # "SATISFIED", "VIOLATED", "UNKNOWN"
    
    # Scale
    problem_dimension: int = 0
    problem_scale: str = "small"  # "small" (<10), "medium" (10-50), "large" (>50)
    
    def __post_init__(self):
        """Determine problem scale from dimension."""
        if self.problem_dimension < 10:
            self.problem_scale = "small"
        elif self.problem_dimension <= 50:
            self.problem_scale = "medium"
        else:
            self.problem_scale = "large"


@dataclass
class SolverResult:
    """Result from optimization solver."""
    x: np.ndarray  # Solution point
    f: float  # Objective value at solution
    success: bool  # Whether solver succeeded
    message: str  # Status message
    nit: int  # Number of iterations
    nfev: int  # Number of function evaluations
    gradient_norm: float = 0.0  # Gradient norm at solution
    
    # For constrained problems
    lambda_ineq: Optional[np.ndarray] = None  # Inequality multipliers
    lambda_eq: Optional[np.ndarray] = None  # Equality multipliers
    constraint_violation: float = 0.0  # Max constraint violation


@dataclass
class VerificationResult:
    """Result of solution verification."""
    status: str  # "PASS", "FAIL"
    gradient_norm: float = 0.0
    
    # KKT residuals for constrained problems
    kkt_stationarity: float = 0.0
    kkt_primal_feasibility: float = 0.0
    kkt_dual_feasibility: float = 0.0
    kkt_complementarity: float = 0.0
    
    # Second-order conditions
    hessian_eigenvalues: Optional[np.ndarray] = None
    sosc_status: str = "UNKNOWN"  # "PASS", "FAIL", "UNKNOWN"
    
    # Details
    details: Dict = field(default_factory=dict)
    
    def __str__(self):
        if self.status == "PASS":
            return f"Verification PASSED (gradient norm: {self.gradient_norm:.2e})"
        else:
            return f"Verification FAILED: {self.details.get('failure_reason', 'Unknown')}"


@dataclass
class ReasoningTrace:
    """Complete trace of optimization reasoning and execution."""
    problem: Dict
    structural_analysis: Dict
    solver_selection: Dict
    execution: Dict
    verification: Dict
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem": self.problem,
            "structural_analysis": self.structural_analysis,
            "solver_selection": self.solver_selection,
            "execution": self.execution,
            "verification": self.verification,
            "warnings": self.warnings
        }
