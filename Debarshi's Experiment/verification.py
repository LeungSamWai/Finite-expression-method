"""
Solution verification using optimality conditions.
"""
import numpy as np
from typing import Optional
import sympy as sp

from data_structures import OptimizationProblem, SolverResult, VerificationResult


class SolutionVerifier:
    """Verify optimization solutions."""
    
    def __init__(self, problem: OptimizationProblem, tolerance: float = 1e-6):
        self.problem = problem
        self.tolerance = tolerance
        self.epsilon = tolerance
        
        # Create numerical functions
        self._create_numerical_functions()
    
    def _create_numerical_functions(self):
        """Create numerical functions from symbolic expressions."""
        # Gradient
        grad_syms = [sp.diff(self.problem.objective, var) 
                     for var in self.problem.variables]
        self.grad_f = sp.lambdify(self.problem.variables, 
                                 grad_syms, 
                                 modules='numpy')
        
        # Hessian
        H = sp.hessian(self.problem.objective, self.problem.variables)
        self.hess_f = sp.lambdify(self.problem.variables, 
                                 H, 
                                 modules='numpy')
        
        # Constraint functions
        self.g_funcs = []
        self.grad_g_funcs = []
        
        for g in self.problem.inequality_constraints:
            g_func = sp.lambdify(self.problem.variables, g, modules='numpy')
            self.g_funcs.append(g_func)
            
            grad_g = [sp.diff(g, var) for var in self.problem.variables]
            grad_g_func = sp.lambdify(self.problem.variables, grad_g, modules='numpy')
            self.grad_g_funcs.append(grad_g_func)
        
        self.h_funcs = []
        self.grad_h_funcs = []
        
        for h in self.problem.equality_constraints:
            h_func = sp.lambdify(self.problem.variables, h, modules='numpy')
            self.h_funcs.append(h_func)
            
            grad_h = [sp.diff(h, var) for var in self.problem.variables]
            grad_h_func = sp.lambdify(self.problem.variables, grad_h, modules='numpy')
            self.grad_h_funcs.append(grad_h_func)
    
    def verify(self, result: SolverResult) -> VerificationResult:
        """
        Verify a solution.
        
        Args:
            result: SolverResult to verify
        
        Returns:
            VerificationResult
        """
        if not result.success:
            return VerificationResult(
                status="FAIL",
                details={"failure_reason": "Solver did not converge"}
            )
        
        x_star = result.x
        
        # Check first-order optimality
        if not self.problem.has_constraints:
            return self._verify_unconstrained(x_star)
        else:
            return self._verify_constrained(x_star, result)
    
    def _verify_unconstrained(self, x_star: np.ndarray) -> VerificationResult:
        """Verify unconstrained problem."""
        # First-order: gradient should be zero
        grad = np.array(self.grad_f(*x_star), dtype=float)
        grad_norm = np.linalg.norm(grad)
        
        # Second-order: Hessian should be positive definite
        H = np.array(self.hess_f(*x_star), dtype=float)
        eigenvalues = np.linalg.eigvalsh(H)
        min_eig = np.min(eigenvalues)
        
        # Determine SOSC status
        if min_eig > self.tolerance:
            sosc_status = "PASS"
        elif min_eig > -self.tolerance:
            sosc_status = "MARGINAL"
        else:
            sosc_status = "FAIL"
        
        # Overall status
        if grad_norm < self.tolerance and sosc_status == "PASS":
            status = "PASS"
        else:
            status = "FAIL"
        
        return VerificationResult(
            status=status,
            gradient_norm=grad_norm,
            hessian_eigenvalues=eigenvalues,
            sosc_status=sosc_status,
            details={
                "min_eigenvalue": min_eig,
                "failure_reason": self._get_failure_reason(grad_norm, sosc_status)
            }
        )
    
    def _verify_constrained(self, x_star: np.ndarray, 
                          result: SolverResult) -> VerificationResult:
        """Verify constrained problem using KKT conditions."""
        # Evaluate gradient
        grad_f = np.array(self.grad_f(*x_star), dtype=float)
        
        # Evaluate constraints
        g_vals = []
        grad_g_vals = []
        for i, g_func in enumerate(self.g_funcs):
            g_val = float(g_func(*x_star))
            g_vals.append(g_val)
            grad_g = np.array(self.grad_g_funcs[i](*x_star), dtype=float)
            grad_g_vals.append(grad_g)
        
        h_vals = []
        grad_h_vals = []
        for i, h_func in enumerate(self.h_funcs):
            h_val = float(h_func(*x_star))
            h_vals.append(h_val)
            grad_h = np.array(self.grad_h_funcs[i](*x_star), dtype=float)
            grad_h_vals.append(grad_h)
        
        # Get multipliers
        mu = result.lambda_ineq if result.lambda_ineq is not None else np.zeros(len(g_vals))
        lambda_eq = result.lambda_eq if result.lambda_eq is not None else np.zeros(len(h_vals))
        
        # KKT stationarity: grad_f + sum(mu_i * grad_g_i) + sum(lambda_j * grad_h_j) = 0
        stationarity_residual = grad_f.copy()
        
        for i in range(len(mu)):
            if i < len(grad_g_vals):
                stationarity_residual += mu[i] * grad_g_vals[i]
        
        for j in range(len(lambda_eq)):
            if j < len(grad_h_vals):
                stationarity_residual += lambda_eq[j] * grad_h_vals[j]
        
        r_stationarity = np.linalg.norm(stationarity_residual)
        
        # Primal feasibility
        r_primal = 0.0
        if g_vals:
            r_primal = max(r_primal, max(g_vals))
        if h_vals:
            r_primal = max(r_primal, max(abs(h) for h in h_vals))
        
        # Dual feasibility
        r_dual = 0.0
        if len(mu) > 0:
            r_dual = -np.min(mu)  # Should be >= 0
        
        # Complementarity
        r_comp = 0.0
        for i in range(len(mu)):
            if i < len(g_vals):
                r_comp = max(r_comp, abs(mu[i] * g_vals[i]))
        
        # Check second-order sufficient conditions
        sosc_status = self._check_sosc_constrained(x_star, mu, lambda_eq, g_vals)
        
        # Overall verification
        kkt_satisfied = (
            r_stationarity < self.tolerance and
            r_primal < self.tolerance and
            r_dual < 0 and  # All multipliers non-negative
            r_comp < self.tolerance
        )
        
        if kkt_satisfied and sosc_status == "PASS":
            status = "PASS"
        else:
            status = "FAIL"
        
        # Compute Hessian eigenvalues for reporting
        H = np.array(self.hess_f(*x_star), dtype=float)
        eigenvalues = np.linalg.eigvalsh(H)
        
        return VerificationResult(
            status=status,
            gradient_norm=r_stationarity,  # Use stationarity residual
            kkt_stationarity=r_stationarity,
            kkt_primal_feasibility=r_primal,
            kkt_dual_feasibility=r_dual,
            kkt_complementarity=r_comp,
            hessian_eigenvalues=eigenvalues,
            sosc_status=sosc_status,
            details={
                "multipliers_ineq": mu.tolist() if len(mu) > 0 else [],
                "multipliers_eq": lambda_eq.tolist() if len(lambda_eq) > 0 else [],
                "constraint_values_ineq": g_vals,
                "constraint_values_eq": h_vals,
                "failure_reason": self._get_failure_reason_constrained(
                    r_stationarity, r_primal, r_dual, r_comp, sosc_status
                )
            }
        )
    
    def _check_sosc_constrained(self, x_star: np.ndarray, 
                               mu: np.ndarray, lambda_eq: np.ndarray,
                               g_vals: list) -> str:
        """Check second-order sufficient conditions for constrained problem."""
        try:
            # Identify active constraints
            active_ineq = []
            for i, g_val in enumerate(g_vals):
                if g_val >= -self.epsilon:
                    active_ineq.append(i)
            
            # Build constraint Jacobian
            constraint_grads = []
            
            # Equality constraints
            for grad_h_func in self.grad_h_funcs:
                grad = np.array(grad_h_func(*x_star), dtype=float)
                constraint_grads.append(grad)
            
            # Active inequality constraints
            for i in active_ineq:
                if i < len(self.grad_g_funcs):
                    grad = np.array(self.grad_g_funcs[i](*x_star), dtype=float)
                    constraint_grads.append(grad)
            
            if not constraint_grads:
                # No active constraints, check unconstrained SOSC
                H = np.array(self.hess_f(*x_star), dtype=float)
                eigenvalues = np.linalg.eigvalsh(H)
                return "PASS" if np.min(eigenvalues) > self.tolerance else "FAIL"
            
            # Construct Jacobian matrix
            J = np.vstack(constraint_grads)
            
            # Compute nullspace basis
            _, _, Vt = np.linalg.svd(J, full_matrices=True)
            rank = np.linalg.matrix_rank(J)
            
            if rank < J.shape[0]:
                # Degenerate case
                return "UNKNOWN"
            
            Z = Vt[rank:].T  # Nullspace basis
            
            if Z.shape[1] == 0:
                # Nullspace is trivial
                return "PASS"
            
            # Check if Z^T H Z is positive definite
            H = np.array(self.hess_f(*x_star), dtype=float)
            projected_H = Z.T @ H @ Z
            
            eigenvalues = np.linalg.eigvalsh(projected_H)
            min_eig = np.min(eigenvalues)
            
            if min_eig > self.tolerance:
                return "PASS"
            elif min_eig > -self.tolerance:
                return "MARGINAL"
            else:
                return "FAIL"
                
        except Exception as e:
            return "UNKNOWN"
    
    def _get_failure_reason(self, grad_norm: float, sosc_status: str) -> Optional[str]:
        """Get failure reason for unconstrained problem."""
        reasons = []
        
        if grad_norm >= self.tolerance:
            reasons.append(f"Gradient norm too large ({grad_norm:.2e} >= {self.tolerance:.2e})")
        
        if sosc_status != "PASS":
            reasons.append(f"Second-order conditions not satisfied (status: {sosc_status})")
        
        return "; ".join(reasons) if reasons else None
    
    def _get_failure_reason_constrained(self, r_stat: float, r_primal: float,
                                       r_dual: float, r_comp: float,
                                       sosc_status: str) -> Optional[str]:
        """Get failure reason for constrained problem."""
        reasons = []
        
        if r_stat >= self.tolerance:
            reasons.append(f"KKT stationarity violated ({r_stat:.2e})")
        
        if r_primal >= self.tolerance:
            reasons.append(f"Primal feasibility violated ({r_primal:.2e})")
        
        if r_dual >= 0:
            reasons.append(f"Dual feasibility violated (min multiplier: {-r_dual:.2e})")
        
        if r_comp >= self.tolerance:
            reasons.append(f"Complementarity violated ({r_comp:.2e})")
        
        if sosc_status != "PASS":
            reasons.append(f"SOSC not satisfied (status: {sosc_status})")
        
        return "; ".join(reasons) if reasons else None


def verify_solution(problem: OptimizationProblem, 
                   result: SolverResult,
                   tolerance: float = 1e-6) -> VerificationResult:
    """
    Main entry point for solution verification.
    
    Args:
        problem: OptimizationProblem
        result: SolverResult to verify
        tolerance: Numerical tolerance
    
    Returns:
        VerificationResult
    """
    verifier = SolutionVerifier(problem, tolerance)
    return verifier.verify(result)
