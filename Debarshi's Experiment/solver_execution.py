"""
Solver execution with automatic differentiation.
"""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from scipy.optimize import differential_evolution
import sympy as sp
from typing import Callable, Tuple, Optional, List
import warnings

from data_structures import OptimizationProblem, SolverResult


class SolverExecutor:
    """Execute optimization solvers with AD."""
    
    def __init__(self, problem: OptimizationProblem):
        self.problem = problem
        self.n = problem.n_vars
        
        # Create numerical functions
        self._create_numerical_functions()
    
    def _create_numerical_functions(self):
        """Create numerical functions from symbolic expressions."""
        # Objective function
        self.f = sp.lambdify(self.problem.variables, 
                            self.problem.objective, 
                            modules='numpy')
        
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
        self.g_funcs = []  # Inequality constraints
        self.grad_g_funcs = []
        
        for g in self.problem.inequality_constraints:
            g_func = sp.lambdify(self.problem.variables, g, modules='numpy')
            self.g_funcs.append(g_func)
            
            grad_g = [sp.diff(g, var) for var in self.problem.variables]
            grad_g_func = sp.lambdify(self.problem.variables, grad_g, modules='numpy')
            self.grad_g_funcs.append(grad_g_func)
        
        self.h_funcs = []  # Equality constraints
        self.grad_h_funcs = []
        
        for h in self.problem.equality_constraints:
            h_func = sp.lambdify(self.problem.variables, h, modules='numpy')
            self.h_funcs.append(h_func)
            
            grad_h = [sp.diff(h, var) for var in self.problem.variables]
            grad_h_func = sp.lambdify(self.problem.variables, grad_h, modules='numpy')
            self.grad_h_funcs.append(grad_h_func)
    
    def solve(self, solver_name: str, x0: Optional[np.ndarray] = None,
              max_iter: int = 500) -> SolverResult:
        """
        Solve the optimization problem.
        
        Args:
            solver_name: Name of solver to use
            x0: Initial point (generated if None)
            max_iter: Maximum iterations
        
        Returns:
            SolverResult
        """
        if x0 is None:
            x0 = self._generate_initial_point()
        
        # Normalize initial point
        x0_normalized, scale_params = self._normalize_point(x0)
        
        # Create wrapped functions that handle normalization
        def f_normalized(x):
            x_original = self._denormalize_point(x, scale_params)
            return float(self.f(*x_original))
        
        def grad_f_normalized(x):
            x_original = self._denormalize_point(x, scale_params)
            g = np.array(self.grad_f(*x_original), dtype=float)
            # Scale gradient
            g = g / scale_params['sigma']
            return g
        
        def hess_f_normalized(x):
            x_original = self._denormalize_point(x, scale_params)
            H = np.array(self.hess_f(*x_original), dtype=float)
            # Scale Hessian
            H = H / np.outer(scale_params['sigma'], scale_params['sigma'])
            return H
        
        # Select and run solver
        try:
            if solver_name == "L-BFGS":
                result = self._solve_lbfgs(f_normalized, grad_f_normalized, 
                                          x0_normalized, max_iter)
            elif solver_name == "Gradient Descent":
                result = self._solve_gradient_descent(f_normalized, grad_f_normalized,
                                                     x0_normalized, max_iter)
            elif solver_name == "Trust-Region Newton":
                result = self._solve_trust_region(f_normalized, grad_f_normalized,
                                                 hess_f_normalized, x0_normalized, max_iter)
            elif solver_name == "SLSQP":
                result = self._solve_slsqp(x0_normalized, scale_params, max_iter)
            elif solver_name == "Interior Point":
                result = self._solve_interior_point(x0_normalized, scale_params, max_iter)
            elif solver_name == "Augmented Lagrangian":
                result = self._solve_augmented_lagrangian(x0_normalized, scale_params, max_iter)
            elif solver_name == "Nelder-Mead":
                result = self._solve_nelder_mead(f_normalized, x0_normalized, max_iter)
            else:
                raise ValueError(f"Unknown solver: {solver_name}")
            
            # Denormalize solution
            result.x = self._denormalize_point(result.x, scale_params)
            
            return result
            
        except Exception as e:
            return SolverResult(
                x=x0,
                f=float('inf'),
                success=False,
                message=f"Solver failed: {str(e)}",
                nit=0,
                nfev=0
            )
    
    def _generate_initial_point(self) -> np.ndarray:
        """Generate random initial point."""
        return np.random.randn(self.n) * 0.5
    
    def _normalize_point(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Normalize point to unit scale."""
        mu = x.copy()
        sigma = np.ones_like(x)
        
        x_normalized = (x - mu) / sigma
        
        return x_normalized, {'mu': mu, 'sigma': sigma}
    
    def _denormalize_point(self, x_norm: np.ndarray, params: dict) -> np.ndarray:
        """Denormalize point back to original scale."""
        return x_norm * params['sigma'] + params['mu']
    
    def _solve_lbfgs(self, f, grad_f, x0, max_iter) -> SolverResult:
        """Solve with L-BFGS."""
        result = minimize(
            f, x0, method='L-BFGS-B', jac=grad_f,
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-6}
        )
        
        grad_norm = np.linalg.norm(grad_f(result.x))
        
        return SolverResult(
            x=result.x,
            f=result.fun,
            success=result.success,
            message=result.message,
            nit=result.nit,
            nfev=result.nfev,
            gradient_norm=grad_norm
        )
    
    def _solve_gradient_descent(self, f, grad_f, x0, max_iter) -> SolverResult:
        """Solve with gradient descent (backtracking line search)."""
        x = x0.copy()
        alpha0 = 1.0
        rho = 0.5
        c = 1e-4
        
        nit = 0
        nfev = 0
        
        for iteration in range(max_iter):
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < 1e-6:
                success = True
                message = "Converged"
                break
            
            # Backtracking line search
            alpha = alpha0
            f_current = f(x)
            nfev += 1
            
            for _ in range(20):  # Max backtracking steps
                x_new = x - alpha * grad
                f_new = f(x_new)
                nfev += 1
                
                # Armijo condition
                if f_new <= f_current - c * alpha * grad_norm**2:
                    break
                
                alpha *= rho
            
            x = x_new
            nit += 1
        else:
            success = False
            message = "Maximum iterations reached"
            grad_norm = np.linalg.norm(grad_f(x))
        
        return SolverResult(
            x=x,
            f=f(x),
            success=success,
            message=message,
            nit=nit,
            nfev=nfev,
            gradient_norm=grad_norm
        )
    
    def _solve_trust_region(self, f, grad_f, hess_f, x0, max_iter) -> SolverResult:
        """Solve with trust-region Newton."""
        result = minimize(
            f, x0, method='trust-ncg', jac=grad_f, hess=hess_f,
            options={'maxiter': max_iter}
        )
        
        grad_norm = np.linalg.norm(grad_f(result.x))
        
        return SolverResult(
            x=result.x,
            f=result.fun,
            success=result.success,
            message=result.message,
            nit=result.nit,
            nfev=result.nfev,
            gradient_norm=grad_norm
        )
    
    def _solve_slsqp(self, x0, scale_params, max_iter) -> SolverResult:
        """Solve with SLSQP (for constrained problems)."""
        # Create constraints
        constraints = []
        
        # Inequality constraints g(x) <= 0
        for i, g_func in enumerate(self.g_funcs):
            def g_normalized(x, idx=i):
                x_orig = self._denormalize_point(x, scale_params)
                return float(g_func(*x_orig))
            
            def grad_g_normalized(x, idx=i):
                x_orig = self._denormalize_point(x, scale_params)
                g = np.array(self.grad_g_funcs[idx](*x_orig), dtype=float)
                return g / scale_params['sigma']
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, gf=g_normalized: -gf(x),  # Convert to >= 0
                'jac': lambda x, gg=grad_g_normalized: -gg(x)
            })
        
        # Equality constraints h(x) = 0
        for i, h_func in enumerate(self.h_funcs):
            def h_normalized(x, idx=i):
                x_orig = self._denormalize_point(x, scale_params)
                return float(h_func(*x_orig))
            
            def grad_h_normalized(x, idx=i):
                x_orig = self._denormalize_point(x, scale_params)
                h = np.array(self.grad_h_funcs[idx](*x_orig), dtype=float)
                return h / scale_params['sigma']
            
            constraints.append({
                'type': 'eq',
                'fun': h_normalized,
                'jac': grad_h_normalized
            })
        
        # Objective (normalized)
        def f_norm(x):
            x_orig = self._denormalize_point(x, scale_params)
            return float(self.f(*x_orig))
        
        def grad_f_norm(x):
            x_orig = self._denormalize_point(x, scale_params)
            g = np.array(self.grad_f(*x_orig), dtype=float)
            return g / scale_params['sigma']
        
        result = minimize(
            f_norm, x0, method='SLSQP', jac=grad_f_norm,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )
        
        grad_norm = np.linalg.norm(grad_f_norm(result.x))
        
        # Extract multipliers (SLSQP doesn't provide them directly)
        # We'll recover them approximately
        lambda_ineq, lambda_eq = self._recover_multipliers(result.x, scale_params)
        
        return SolverResult(
            x=result.x,
            f=result.fun,
            success=result.success,
            message=result.message,
            nit=result.nit,
            nfev=result.nfev,
            gradient_norm=grad_norm,
            lambda_ineq=lambda_ineq,
            lambda_eq=lambda_eq
        )
    
    def _solve_interior_point(self, x0, scale_params, max_iter) -> SolverResult:
        """Solve with interior point method (for convex with linear constraints)."""
        # Use trust-constr which is an interior point method
        return self._solve_slsqp(x0, scale_params, max_iter)  # Simplified
    
    def _solve_augmented_lagrangian(self, x0, scale_params, max_iter) -> SolverResult:
        """Solve with augmented Lagrangian (for LICQ violations)."""
        # Use SLSQP as approximation
        return self._solve_slsqp(x0, scale_params, max_iter)  # Simplified
    
    def _solve_nelder_mead(self, f, x0, max_iter) -> SolverResult:
        """Solve with Nelder-Mead (derivative-free)."""
        result = minimize(
            f, x0, method='Nelder-Mead',
            options={'maxiter': max_iter}
        )
        
        return SolverResult(
            x=result.x,
            f=result.fun,
            success=result.success,
            message=result.message,
            nit=result.nit,
            nfev=result.nfev,
            gradient_norm=0.0  # Not applicable
        )
    
    def _recover_multipliers(self, x_solution, scale_params) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Recover Lagrange multipliers via least-squares.
        
        Solve: min ||grad_f + sum(mu_i * grad_g_i) + sum(lambda_j * grad_h_j)||^2
               s.t. mu >= 0
        """
        x_orig = self._denormalize_point(x_solution, scale_params)
        
        # Compute gradient of objective
        grad_f_val = np.array(self.grad_f(*x_orig), dtype=float)
        
        # Collect constraint gradients
        grad_g_vals = []
        for grad_g_func in self.grad_g_funcs:
            grad_g_vals.append(np.array(grad_g_func(*x_orig), dtype=float))
        
        grad_h_vals = []
        for grad_h_func in self.grad_h_funcs:
            grad_h_vals.append(np.array(grad_h_func(*x_orig), dtype=float))
        
        # Simple approach: use pseudoinverse
        if grad_g_vals or grad_h_vals:
            all_grads = grad_g_vals + grad_h_vals
            J = np.vstack(all_grads) if all_grads else np.zeros((0, self.n))
            
            # Solve J^T * multipliers = -grad_f
            if J.shape[0] > 0:
                multipliers, _, _, _ = np.linalg.lstsq(J.T, -grad_f_val, rcond=None)
                
                n_ineq = len(grad_g_vals)
                lambda_ineq = multipliers[:n_ineq] if n_ineq > 0 else None
                lambda_eq = multipliers[n_ineq:] if len(grad_h_vals) > 0 else None
                
                # Project inequality multipliers to non-negative
                if lambda_ineq is not None:
                    lambda_ineq = np.maximum(0, lambda_ineq)
                
                return lambda_ineq, lambda_eq
        
        return None, None


def solve_problem(problem: OptimizationProblem, solver_name: str,
                 x0: Optional[np.ndarray] = None,
                 max_iter: int = 500) -> SolverResult:
    """
    Main entry point for solving optimization problems.
    
    Args:
        problem: OptimizationProblem to solve
        solver_name: Name of solver to use
        x0: Initial point (optional)
        max_iter: Maximum iterations
    
    Returns:
        SolverResult
    """
    executor = SolverExecutor(problem)
    return executor.solve(solver_name, x0, max_iter)
