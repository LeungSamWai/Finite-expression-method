"""
Main OptimAI orchestrator.
"""
import numpy as np
import json
from typing import Optional, Dict, List
import warnings

from data_structures import (
    OptimizationProblem, StructureReport, SolverResult, 
    VerificationResult, ReasoningTrace
)
from parsing import parse_problem, ProblemParser
from structure_inference import infer_structure
from algorithm_selection import select_algorithm
from solver_execution import solve_problem, SolverExecutor
from verification import verify_solution


class OptimAI:
    """
    Structure-aware optimization system with formal verification.
    """
    
    def __init__(self, sample_size: int = 100, max_iter: int = 500, 
                 tolerance: float = 1e-6, multi_start: int = 1):
        """
        Initialize OptimAI.
        
        Args:
            sample_size: Number of samples for structure inference
            max_iter: Maximum solver iterations
            tolerance: Numerical tolerance for verification
            multi_start: Number of random starts for nonconvex problems
        """
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.multi_start = multi_start
        
        self.warnings = []
    
    def optimize(self, description: str, format: str = "natural_language",
                initial_point: Optional[np.ndarray] = None) -> Dict:
        """
        Solve an optimization problem end-to-end.
        
        Args:
            description: Problem description
            format: Input format ("natural_language" or "latex")
            initial_point: Optional initial point
        
        Returns:
            Dictionary with complete reasoning trace and solution
        """
        self.warnings = []
        
        # 1. Parse problem
        print("Step 1: Parsing problem...")
        try:
            problem = parse_problem(description, format)
            print(f"  Parsed {problem.n_vars} variables, "
                  f"{problem.n_inequality_constraints} inequality constraints, "
                  f"{problem.n_equality_constraints} equality constraints")
        except Exception as e:
            return {
                "status": "FAILED",
                "error": f"Parsing failed: {str(e)}",
                "warnings": self.warnings
            }
        
        # 2. Infer structure
        print("\nStep 2: Analyzing problem structure...")
        try:
            structure = infer_structure(problem, self.sample_size)
            print(f"  Convexity: {structure.convexity_status} "
                  f"(confidence: {structure.convexity_confidence:.2f}, "
                  f"method: {structure.convexity_method})")
            if structure.convexity_ci:
                print(f"  Confidence interval: [{structure.convexity_ci[0]:.3f}, "
                      f"{structure.convexity_ci[1]:.3f}]")
            print(f"  Condition number: {structure.condition_number_median:.2e}")
            print(f"  LICQ status: {structure.licq_status}")
            
            # Add warnings for uncertain structure
            if structure.convexity_status == "UNCERTAIN":
                warning = "Convexity could not be reliably determined"
                self.warnings.append(warning)
                print(f"  WARNING: {warning}")
        except Exception as e:
            return {
                "status": "FAILED",
                "error": f"Structure inference failed: {str(e)}",
                "warnings": self.warnings
            }
        
        # 3. Select algorithm
        print("\nStep 3: Selecting algorithm...")
        try:
            solver_name, reason, alternatives = select_algorithm(structure)
            print(f"  Selected: {solver_name}")
            print(f"  Reason: {reason}")
            print(f"  Alternatives considered: {', '.join(alternatives)}")
        except Exception as e:
            return {
                "status": "FAILED",
                "error": f"Algorithm selection failed: {str(e)}",
                "warnings": self.warnings
            }
        
        # 4. Execute solver
        print("\nStep 4: Executing solver...")
        
        # Determine if multi-start is needed
        use_multi_start = (
            structure.convexity_status in ["NONCONVEX", "UNCERTAIN"] and
            self.multi_start > 1
        )
        
        if use_multi_start:
            print(f"  Using multi-start strategy ({self.multi_start} starts)")
            results = self._solve_multistart(problem, solver_name, initial_point)
        else:
            if initial_point is None:
                initial_point = ProblemParser.generate_initial_point(problem)
            results = [solve_problem(problem, solver_name, initial_point, self.max_iter)]
        
        # Get best result
        best_result = min(results, key=lambda r: r.f if r.success else float('inf'))
        
        print(f"  Converged: {best_result.success}")
        print(f"  Iterations: {best_result.nit}")
        print(f"  Function evaluations: {best_result.nfev}")
        print(f"  Final objective: {best_result.f:.6e}")
        
        # 5. Verify solution
        print("\nStep 5: Verifying solution...")
        try:
            verification = verify_solution(problem, best_result, self.tolerance)
            print(f"  Status: {verification.status}")
            print(f"  Gradient norm: {verification.gradient_norm:.2e}")
            if problem.has_constraints:
                print(f"  KKT stationarity: {verification.kkt_stationarity:.2e}")
                print(f"  KKT primal feasibility: {verification.kkt_primal_feasibility:.2e}")
                print(f"  KKT complementarity: {verification.kkt_complementarity:.2e}")
            print(f"  SOSC status: {verification.sosc_status}")
            
            if verification.status == "FAIL":
                failure_reason = verification.details.get('failure_reason', 'Unknown')
                warning = f"Verification failed: {failure_reason}"
                self.warnings.append(warning)
                print(f"  WARNING: {warning}")
                
                # Try recovery
                print("\n  Attempting recovery...")
                recovered_result, recovered_verification = self._attempt_recovery(
                    problem, structure, solver_name, best_result
                )
                
                if recovered_verification and recovered_verification.status == "PASS":
                    print("  Recovery successful!")
                    best_result = recovered_result
                    verification = recovered_verification
                else:
                    print("  Recovery failed.")
        except Exception as e:
            verification = VerificationResult(
                status="ERROR",
                details={"error": str(e)}
            )
            self.warnings.append(f"Verification error: {str(e)}")
        
        # 6. Generate reasoning trace
        trace = self._generate_trace(
            problem, structure, solver_name, reason, alternatives,
            best_result, verification, use_multi_start, results
        )
        
        # 7. Generate natural language explanation
        explanation = self._generate_explanation(
            problem, structure, solver_name, reason,
            best_result, verification
        )
        
        print("\n" + "="*60)
        print("EXPLANATION")
        print("="*60)
        print(explanation)
        print("="*60)
        
        return {
            "status": "SUCCESS" if verification.status == "PASS" else "COMPLETED_WITH_WARNINGS",
            "solution": best_result.x.tolist(),
            "objective_value": float(best_result.f),
            "verification": verification.status,
            "explanation": explanation,
            "trace": trace.to_dict(),
            "warnings": self.warnings
        }
    
    def _solve_multistart(self, problem: OptimizationProblem, 
                         solver_name: str,
                         initial_point: Optional[np.ndarray]) -> List[SolverResult]:
        """Solve with multiple random starts."""
        results = []
        
        for i in range(self.multi_start):
            if i == 0 and initial_point is not None:
                x0 = initial_point
            else:
                x0 = ProblemParser.generate_initial_point(problem, seed=i)
            
            result = solve_problem(problem, solver_name, x0, self.max_iter)
            results.append(result)
            
            if result.success:
                print(f"    Start {i+1}/{self.multi_start}: f = {result.f:.6e}")
        
        return results
    
    def _attempt_recovery(self, problem: OptimizationProblem,
                         structure: StructureReport,
                         original_solver: str,
                         original_result: SolverResult):
        """Attempt to recover from verification failure."""
        # Try different solver
        if original_solver == "L-BFGS":
            new_solver = "Trust-Region Newton"
        elif original_solver == "Trust-Region Newton":
            new_solver = "Gradient Descent"
        else:
            new_solver = "L-BFGS"
        
        print(f"    Trying alternative solver: {new_solver}")
        
        # Generate new initial point
        x0 = ProblemParser.generate_initial_point(problem)
        
        # Solve
        result = solve_problem(problem, new_solver, x0, self.max_iter)
        
        # Verify
        verification = verify_solution(problem, result, self.tolerance)
        
        return result, verification
    
    def _generate_trace(self, problem, structure, solver_name, reason, 
                       alternatives, result, verification, multi_start, all_results):
        """Generate reasoning trace."""
        return ReasoningTrace(
            problem={
                "objective": str(problem.objective),
                "n_variables": problem.n_vars,
                "n_inequality_constraints": problem.n_inequality_constraints,
                "n_equality_constraints": problem.n_equality_constraints,
                "dimension": problem.n_vars
            },
            structural_analysis={
                "convexity_status": structure.convexity_status,
                "convexity_confidence": structure.convexity_confidence,
                "convexity_method": structure.convexity_method,
                "convexity_ci": structure.convexity_ci,
                "lipschitz_lower_bound": structure.lipschitz_lower_bound,
                "condition_number_median": structure.condition_number_median,
                "is_smooth": structure.is_smooth,
                "licq_status": structure.licq_status
            },
            solver_selection={
                "chosen": solver_name,
                "reason": reason,
                "alternatives_considered": alternatives,
                "multi_start_used": multi_start,
                "num_starts": len(all_results) if multi_start else 1
            },
            execution={
                "iterations": result.nit,
                "function_evals": result.nfev,
                "final_objective": result.f,
                "convergence": "SUCCESS" if result.success else "FAILED",
                "message": result.message,
                "objective_values_all_starts": [r.f for r in all_results] if multi_start else None
            },
            verification={
                "gradient_norm": verification.gradient_norm,
                "kkt_stationarity": verification.kkt_stationarity,
                "kkt_primal_feasibility": verification.kkt_primal_feasibility,
                "kkt_dual_feasibility": verification.kkt_dual_feasibility,
                "kkt_complementarity": verification.kkt_complementarity,
                "hessian_eigenvalues": verification.hessian_eigenvalues.tolist() if verification.hessian_eigenvalues is not None else None,
                "sosc_status": verification.sosc_status,
                "overall_status": verification.status,
                "details": verification.details
            },
            warnings=self.warnings
        )
    
    def _generate_explanation(self, problem, structure, solver_name, reason,
                            result, verification):
        """Generate natural language explanation."""
        explanation = []
        
        # Structure analysis
        explanation.append("STRUCTURAL ANALYSIS:")
        explanation.append(f"I analyzed the problem and determined it is {structure.convexity_status.lower()}")
        
        if structure.convexity_method == "symbolic":
            explanation.append(f"with 100% confidence based on symbolic Hessian analysis (Sylvester's criterion).")
        else:
            explanation.append(f"with {structure.convexity_confidence:.0%} empirical confidence based on sampling.")
            if structure.convexity_ci:
                explanation.append(f"The 95% confidence interval is [{structure.convexity_ci[0]:.0%}, {structure.convexity_ci[1]:.0%}].")
        
        if structure.is_smooth:
            explanation.append(f"The problem is smooth (twice continuously differentiable) " 
                             f"with median condition number {structure.condition_number_median:.2e}.")
        
        if problem.has_constraints:
            explanation.append(f"The problem has constraints. LICQ status: {structure.licq_status}.")
        
        # Solver selection
        explanation.append("\nSOLVER SELECTION:")
        explanation.append(f"Based on these properties, I selected {solver_name}.")
        explanation.append(f"Rationale: {reason}")
        
        # Execution
        explanation.append("\nEXECUTION:")
        if result.success:
            explanation.append(f"The solver converged in {result.nit} iterations "
                             f"to objective value {result.f:.6e}.")
        else:
            explanation.append(f"The solver did not fully converge ({result.message}) "
                             f"but reached objective value {result.f:.6e}.")
        
        # Verification
        explanation.append("\nVERIFICATION:")
        if verification.status == "PASS":
            explanation.append("The solution is VERIFIED as a local optimum.")
            
            if not problem.has_constraints:
                explanation.append("Verification checks passed:")
                explanation.append(f"  - First-order optimality: gradient norm = {verification.gradient_norm:.2e}")
                explanation.append(f"  - Second-order sufficiency: minimum Hessian eigenvalue = "
                                 f"{np.min(verification.hessian_eigenvalues):.2e} > 0")
            else:
                explanation.append("KKT conditions verified:")
                explanation.append(f"  - Stationarity residual: {verification.kkt_stationarity:.2e}")
                explanation.append(f"  - Primal feasibility: {verification.kkt_primal_feasibility:.2e}")
                explanation.append(f"  - Complementarity: {verification.kkt_complementarity:.2e}")
                explanation.append(f"  - SOSC status: {verification.sosc_status}")
            
            if structure.convexity_status == "CONVEX":
                explanation.append("\nSince the problem is convex, this local optimum is also the GLOBAL optimum.")
            else:
                explanation.append("\nNote: For nonconvex problems, this verification only certifies LOCAL optimality.")
        else:
            explanation.append("WARNING: Solution verification FAILED.")
            if verification.details.get('failure_reason'):
                explanation.append(f"Reason: {verification.details['failure_reason']}")
            explanation.append("The solution may not be optimal. Consider:")
            explanation.append("  - Reformulating the problem")
            explanation.append("  - Adjusting numerical tolerances")
            explanation.append("  - Using a different solver")
        
        return "\n".join(explanation)


def optimize(description: str, format: str = "natural_language",
            sample_size: int = 100, max_iter: int = 500,
            tolerance: float = 1e-6, multi_start: int = 1,
            initial_point: Optional[np.ndarray] = None) -> Dict:
    """
    Optimize a problem using OptimAI.
    
    Args:
        description: Problem description
        format: Input format ("natural_language" or "latex")
        sample_size: Number of samples for structure inference
        max_iter: Maximum solver iterations
        tolerance: Numerical tolerance
        multi_start: Number of random starts for nonconvex problems
        initial_point: Optional initial point
    
    Returns:
        Dictionary with solution and reasoning trace
    """
    optimai = OptimAI(sample_size, max_iter, tolerance, multi_start)
    return optimai.optimize(description, format, initial_point)
