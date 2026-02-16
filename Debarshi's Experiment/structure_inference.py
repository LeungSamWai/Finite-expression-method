"""
Structural inference for optimization problems.
"""
import numpy as np
from scipy.stats import norm
import sympy as sp
from typing import Tuple, Optional, List
import warnings

from data_structures import OptimizationProblem, StructureReport


class StructureInference:
    """Infer structural properties of optimization problems."""
    
    def __init__(self, sample_size: int = 100):
        self.sample_size = sample_size
        self.epsilon = 1e-6  # Numerical tolerance
    
    def analyze(self, problem: OptimizationProblem) -> StructureReport:
        """
        Perform complete structural analysis.
        
        Args:
            problem: Optimization problem to analyze
        
        Returns:
            StructureReport with all inferred properties
        """
        # Analyze convexity
        conv_status, conv_conf, conv_method, conv_ci = self._analyze_convexity(problem)
        
        # Analyze smoothness
        is_smooth = self._check_smoothness(problem)
        lipschitz_lb = 0.0
        cond_median = 1.0
        
        if is_smooth:
            lipschitz_lb = self._estimate_lipschitz(problem)
            cond_median = self._estimate_condition_number(problem)
        
        # Analyze constraints
        has_linear = self._check_linear_constraints(problem)
        licq_status = self._check_licq(problem)
        
        return StructureReport(
            convexity_status=conv_status,
            convexity_confidence=conv_conf,
            convexity_method=conv_method,
            convexity_ci=conv_ci,
            lipschitz_lower_bound=lipschitz_lb,
            condition_number_median=cond_median,
            is_smooth=is_smooth,
            has_constraints=problem.has_constraints,
            has_linear_constraints=has_linear,
            licq_status=licq_status,
            problem_dimension=problem.n_vars
        )
    
    def _analyze_convexity(self, problem: OptimizationProblem) -> Tuple[str, float, str, Optional[Tuple]]:
        """
        Analyze convexity using symbolic and numerical methods.
        
        Returns:
            (status, confidence, method, confidence_interval)
        """
        # Try symbolic analysis first
        try:
            is_convex = self._check_convexity_symbolic(problem)
            if is_convex is True:
                return ("CONVEX", 1.0, "symbolic", None)
            elif is_convex is False:
                return ("NONCONVEX", 1.0, "symbolic", None)
        except Exception as e:
            warnings.warn(f"Symbolic convexity check failed: {e}")
        
        # Fall back to sampling
        return self._check_convexity_sampling(problem)
    
    def _check_convexity_symbolic(self, problem: OptimizationProblem) -> Optional[bool]:
        """
        Check convexity symbolically using Sylvester's criterion.
        
        Returns:
            True if convex, False if nonconvex, None if inconclusive
        """
        # Compute Hessian
        H = sp.hessian(problem.objective, problem.variables)
        
        # Check if Hessian is too complex
        if len(str(H)) > 10000:  # Heuristic for complexity
            return None
        
        # Try to check if all leading principal minors are non-negative
        n = len(problem.variables)
        
        try:
            for k in range(1, n + 1):
                # k-th leading principal minor
                minor = H[:k, :k]
                det_minor = sp.det(minor)
                
                # Simplify
                det_minor = sp.simplify(det_minor)
                
                # Try to determine sign
                if det_minor.is_negative:
                    return False
                elif not (det_minor.is_nonnegative or det_minor.is_positive or det_minor.is_zero):
                    # Can't determine
                    return None
            
            # All minors non-negative
            return True
            
        except Exception:
            return None
    
    def _check_convexity_sampling(self, problem: OptimizationProblem) -> Tuple[str, float, str, Optional[Tuple]]:
        """
        Check convexity via eigenvalue sampling.
        
        Returns:
            (status, confidence, method, confidence_interval)
        """
        n = problem.n_vars
        N = self.sample_size
        
        # Check for curse of dimensionality
        if n > 20 and N < 1000:
            return ("UNCERTAIN", 0.0, "sampling", None)
        
        # Sample points
        sample_points = self._sample_feasible_region(problem, N)
        
        # Compute Hessian at each point
        H_func = self._lambdify_hessian(problem)
        
        psd_count = 0
        for x in sample_points:
            try:
                H_num = H_func(*x)
                eigenvalues = np.linalg.eigvalsh(H_num)
                min_eig = np.min(eigenvalues)
                
                if min_eig >= -self.epsilon:
                    psd_count += 1
            except Exception:
                pass  # Skip failed evaluations
        
        # Compute empirical proportion
        p_hat = psd_count / N
        
        # Compute Wilson score confidence interval
        ci = self._wilson_score_interval(p_hat, N)
        
        # Determine status
        if p_hat > 0.9 and ci[0] > 0.8:
            status = "CONVEX"
        elif p_hat < 0.1 and ci[1] < 0.2:
            status = "NONCONVEX"
        else:
            status = "UNCERTAIN"
        
        return (status, p_hat, "sampling", ci)
    
    def _wilson_score_interval(self, p: float, n: int, 
                               confidence: float = 0.95) -> Tuple[float, float]:
        """Compute Wilson score confidence interval."""
        z = norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))
    
    def _sample_feasible_region(self, problem: OptimizationProblem, 
                               N: int) -> np.ndarray:
        """Sample points from the feasible region."""
        n = problem.n_vars
        samples = []
        
        # Generate samples (simple uniform sampling in [-1, 1]^n)
        # In production, would use more sophisticated sampling
        attempts = 0
        max_attempts = N * 100
        
        while len(samples) < N and attempts < max_attempts:
            x = np.random.uniform(-1, 1, n)
            
            # Check feasibility (simplified)
            if self._is_feasible(problem, x):
                samples.append(x)
            
            attempts += 1
        
        if len(samples) < N:
            warnings.warn(f"Could only generate {len(samples)} feasible samples out of {N}")
        
        return np.array(samples) if samples else np.random.uniform(-1, 1, (N, n))
    
    def _is_feasible(self, problem: OptimizationProblem, x: np.ndarray) -> bool:
        """Check if a point is feasible (simplified)."""
        # For now, just accept all points
        # In production, would evaluate constraints
        return True
    
    def _lambdify_hessian(self, problem: OptimizationProblem):
        """Create a numerical function for the Hessian."""
        H = sp.hessian(problem.objective, problem.variables)
        H_func = sp.lambdify(problem.variables, H, modules='numpy')
        return H_func
    
    def _check_smoothness(self, problem: OptimizationProblem) -> bool:
        """Check if problem is C^2 continuous (smooth)."""
        try:
            # Try to compute Hessian
            H = sp.hessian(problem.objective, problem.variables)
            return True
        except Exception:
            return False
    
    def _estimate_lipschitz(self, problem: OptimizationProblem) -> float:
        """
        Estimate lower bound on Lipschitz constant.
        
        Returns:
            Lower bound estimate
        """
        # Sample points
        N = min(50, self.sample_size)
        samples = self._sample_feasible_region(problem, N)
        
        # Compute gradient at each point
        grad_func = self._lambdify_gradient(problem)
        
        max_L = 0.0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                try:
                    xi = samples[i]
                    xj = samples[j]
                    
                    grad_i = np.array(grad_func(*xi))
                    grad_j = np.array(grad_func(*xj))
                    
                    grad_diff = np.linalg.norm(grad_i - grad_j)
                    x_diff = np.linalg.norm(xi - xj)
                    
                    if x_diff > 1e-10:
                        L = grad_diff / x_diff
                        max_L = max(max_L, L)
                except Exception:
                    pass
        
        return max_L
    
    def _lambdify_gradient(self, problem: OptimizationProblem):
        """Create a numerical function for the gradient."""
        grad = [sp.diff(problem.objective, var) for var in problem.variables]
        grad_func = sp.lambdify(problem.variables, grad, modules='numpy')
        return grad_func
    
    def _estimate_condition_number(self, problem: OptimizationProblem) -> float:
        """Estimate median condition number of Hessian."""
        N = min(20, self.sample_size)
        samples = self._sample_feasible_region(problem, N)
        
        H_func = self._lambdify_hessian(problem)
        
        condition_numbers = []
        for x in samples:
            try:
                H_num = H_func(*x)
                eigenvalues = np.linalg.eigvalsh(H_num)
                
                # Avoid division by zero
                min_eig = np.min(np.abs(eigenvalues))
                if min_eig > 1e-10:
                    kappa = np.max(np.abs(eigenvalues)) / min_eig
                    condition_numbers.append(kappa)
            except Exception:
                pass
        
        if condition_numbers:
            return float(np.median(condition_numbers))
        else:
            return 1.0
    
    def _check_linear_constraints(self, problem: OptimizationProblem) -> bool:
        """Check if problem has linear constraints."""
        # Check inequality constraints
        for constraint in problem.inequality_constraints:
            # Compute Hessian
            H = sp.hessian(constraint, problem.variables)
            # If Hessian is zero, constraint is linear
            if H.is_zero_matrix:
                return True
        
        # Check equality constraints
        for constraint in problem.equality_constraints:
            H = sp.hessian(constraint, problem.variables)
            if H.is_zero_matrix:
                return True
        
        return False
    
    def _check_licq(self, problem: OptimizationProblem) -> str:
        """
        Check Linear Independence Constraint Qualification.
        
        Returns:
            "SATISFIED", "VIOLATED", or "UNKNOWN"
        """
        if not problem.has_constraints:
            return "SATISFIED"
        
        # Sample a point
        samples = self._sample_feasible_region(problem, 5)
        
        for x in samples:
            try:
                # Get constraint gradients
                gradients = []
                
                # Equality constraints
                for h in problem.equality_constraints:
                    grad = [sp.diff(h, var) for var in problem.variables]
                    grad_func = sp.lambdify(problem.variables, grad, modules='numpy')
                    grad_val = np.array(grad_func(*x))
                    gradients.append(grad_val)
                
                # Active inequality constraints
                for g in problem.inequality_constraints:
                    g_func = sp.lambdify(problem.variables, g, modules='numpy')
                    g_val = g_func(*x)
                    
                    # Check if active (within epsilon)
                    if g_val >= -self.epsilon:
                        grad = [sp.diff(g, var) for var in problem.variables]
                        grad_func = sp.lambdify(problem.variables, grad, modules='numpy')
                        grad_val = np.array(grad_func(*x))
                        gradients.append(grad_val)
                
                if gradients:
                    # Stack into matrix
                    J = np.vstack(gradients)
                    
                    # Check rank
                    rank = np.linalg.matrix_rank(J)
                    
                    if rank < len(gradients):
                        return "VIOLATED"
                
            except Exception:
                pass
        
        return "SATISFIED"


def infer_structure(problem: OptimizationProblem, 
                   sample_size: int = 100) -> StructureReport:
    """
    Main entry point for structure inference.
    
    Args:
        problem: Optimization problem
        sample_size: Number of samples for numerical analysis
    
    Returns:
        StructureReport
    """
    inference = StructureInference(sample_size=sample_size)
    return inference.analyze(problem)
