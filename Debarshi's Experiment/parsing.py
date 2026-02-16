"""
Parsing and canonicalization of optimization problems.
"""
import re
from typing import Tuple, List, Dict, Optional
import sympy as sp
from sympy import Symbol, sympify
from sympy.core.expr import Expr as SymbolicExpression
import numpy as np

from data_structures import OptimizationProblem


class ProblemParser:
    """Parse optimization problems from various formats."""
    
    @staticmethod
    def parse_from_natural_language(description: str) -> OptimizationProblem:
        """
        Parse a natural language problem description.
        
        This is a simplified parser. In production, this would use an LLM
        to extract the objective and constraints.
        """
        # Extract objective
        obj_match = re.search(r'minimize?\s+(.+?)(?:subject to|s\.t\.|$)', 
                             description, re.IGNORECASE)
        if not obj_match:
            obj_match = re.search(r'maximize?\s+(.+?)(?:subject to|s\.t\.|$)', 
                                 description, re.IGNORECASE)
            if obj_match:
                # Negate for maximization
                obj_str = f"-({obj_match.group(1).strip()})"
            else:
                raise ValueError("Could not extract objective function")
        else:
            obj_str = obj_match.group(1).strip()
        
        # Extract constraints
        constraint_strs = []
        constraint_section = re.search(r'(?:subject to|s\.t\.)\s+(.+)', 
                                      description, re.IGNORECASE | re.DOTALL)
        if constraint_section:
            # Split by common delimiters
            constraints_text = constraint_section.group(1)
            constraint_strs = re.split(r'[,;]|\sand\s', constraints_text)
            constraint_strs = [c.strip() for c in constraint_strs if c.strip()]
        
        return ProblemParser.parse_from_strings(obj_str, constraint_strs)
    
    @staticmethod
    def parse_from_strings(objective_str: str, 
                          constraint_strs: List[str]) -> OptimizationProblem:
        """Parse from objective string and list of constraint strings."""
        # Parse objective
        try:
            objective = sympify(objective_str)
        except Exception as e:
            raise ValueError(f"Could not parse objective '{objective_str}': {e}")
        
        # Extract variables
        variables = sorted(objective.free_symbols, key=lambda s: s.name)
        
        # Parse constraints
        inequality_constraints = []
        equality_constraints = []
        bounds = {}
        
        for c_str in constraint_strs:
            c_str = c_str.strip()
            if not c_str:
                continue
            
            try:
                # Check for inequality
                if '<=' in c_str or '<' in c_str:
                    parts = re.split(r'<=|<', c_str)
                    if len(parts) == 2:
                        lhs = sympify(parts[0].strip())
                        rhs = sympify(parts[1].strip())
                        # Convert to g(x) <= 0 form
                        constraint = lhs - rhs
                        inequality_constraints.append(constraint)
                        # Update variables
                        variables = sorted(set(variables) | constraint.free_symbols, 
                                         key=lambda s: s.name)
                
                elif '>=' in c_str or '>' in c_str:
                    parts = re.split(r'>=|>', c_str)
                    if len(parts) == 2:
                        lhs = sympify(parts[0].strip())
                        rhs = sympify(parts[1].strip())
                        # Convert to g(x) <= 0 form (flip sign)
                        constraint = rhs - lhs
                        inequality_constraints.append(constraint)
                        variables = sorted(set(variables) | constraint.free_symbols, 
                                         key=lambda s: s.name)
                
                elif '=' in c_str and '==' not in c_str:
                    parts = c_str.split('=')
                    if len(parts) == 2:
                        lhs = sympify(parts[0].strip())
                        rhs = sympify(parts[1].strip())
                        # Convert to h(x) = 0 form
                        constraint = lhs - rhs
                        equality_constraints.append(constraint)
                        variables = sorted(set(variables) | constraint.free_symbols, 
                                         key=lambda s: s.name)
                
            except Exception as e:
                print(f"Warning: Could not parse constraint '{c_str}': {e}")
        
        # Validate
        if not variables:
            raise ValueError("No variables found in problem")
        
        # Create problem
        problem = OptimizationProblem(
            objective=objective,
            variables=variables,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            bounds=bounds
        )
        
        # Validate differentiability
        ProblemParser._validate_differentiability(problem)
        
        return problem
    
    @staticmethod
    def _validate_differentiability(problem: OptimizationProblem) -> None:
        """Validate that the problem is differentiable."""
        try:
            # Try to compute gradient
            for var in problem.variables:
                sp.diff(problem.objective, var)
            
            # Try to compute constraint gradients
            for c in problem.inequality_constraints:
                for var in problem.variables:
                    sp.diff(c, var)
            
            for c in problem.equality_constraints:
                for var in problem.variables:
                    sp.diff(c, var)
                    
        except Exception as e:
            raise ValueError(f"Problem is not differentiable: {e}")
    
    @staticmethod
    def parse_from_latex(latex_str: str) -> OptimizationProblem:
        """Parse from LaTeX format (simplified)."""
        # Remove LaTeX commands
        latex_str = latex_str.replace(r'\min', 'minimize')
        latex_str = latex_str.replace(r'\text{s.t.}', 'subject to')
        latex_str = latex_str.replace(r'\le', '<=')
        latex_str = latex_str.replace(r'\ge', '>=')
        latex_str = latex_str.replace(r'\leq', '<=')
        latex_str = latex_str.replace(r'\geq', '>=')
        
        return ProblemParser.parse_from_natural_language(latex_str)
    
    @staticmethod
    def generate_initial_point(problem: OptimizationProblem,
                              seed: Optional[int] = None) -> np.ndarray:
        """Generate a random initial point within bounds."""
        if seed is not None:
            np.random.seed(seed)
        
        n = problem.n_vars
        x0 = np.random.randn(n) * 0.5  # Small random values
        
        # Apply bounds if present
        for i, var in enumerate(problem.variables):
            if var in problem.bounds:
                lb, ub = problem.bounds[var]
                x0[i] = np.clip(x0[i], lb, ub)
        
        return x0


def parse_problem(description: str, 
                 format: str = "natural_language") -> OptimizationProblem:
    """
    Main entry point for parsing optimization problems.
    
    Args:
        description: Problem description
        format: One of "natural_language", "latex"
    
    Returns:
        OptimizationProblem instance
    """
    parser = ProblemParser()
    
    if format == "natural_language":
        return parser.parse_from_natural_language(description)
    elif format == "latex":
        return parser.parse_from_latex(description)
    else:
        raise ValueError(f"Unknown format: {format}")
