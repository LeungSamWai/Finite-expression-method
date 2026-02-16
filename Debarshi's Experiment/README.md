# OptimAI: Structure-Aware Optimization with Formal Verification

A mathematically rigorous optimization system that combines structure inference, algorithm selection, and formal verification.

## Overview

OptimAI is designed to solve continuous optimization problems by:

1. **Parsing** problems from natural language or LaTeX
2. **Inferring structure** (convexity, smoothness, conditioning) with uncertainty quantification
3. **Selecting algorithms** based on mathematical properties
4. **Executing solvers** with automatic differentiation
5. **Verifying solutions** using KKT conditions and second-order tests

## Key Features

- **Explicit Uncertainty**: Reports confidence intervals for structure inference
- **Multi-start for Nonconvex**: Automatically uses multiple initializations
- **Formal Verification**: KKT conditions, SOSC, multiplier recovery
- **Constrained Optimization**: SLSQP, Interior Point, Augmented Lagrangian
- **Complete Reasoning Traces**: JSON output with full audit trail
- **Natural Language Explanations**: Human-readable summaries

## Installation

```bash
pip install numpy scipy sympy
```

## Quick Start

```python
from optimai import optimize

# Solve a simple problem
result = optimize("minimize x**2 + y**2")

print("Solution:", result['solution'])
print("Objective:", result['objective_value'])
print("Verified:", result['verification'])
```

## Usage

### Basic Optimization

```python
from optimai import optimize

result = optimize(
    description="minimize (1-x)**2 + 100*(y-x**2)**2",  # Rosenbrock
    sample_size=100,      # Samples for structure inference
    max_iter=1000,        # Maximum iterations
    multi_start=5         # Number of random starts (nonconvex)
)
```

### Constrained Problems

```python
result = optimize(
    description="minimize x**2 + y**2 subject to x + y >= 1"
)
```

### Viewing Results

```python
import json

# Solution
print("x* =", result['solution'])
print("f(x*) =", result['objective_value'])

# Verification status
print("Verified:", result['verification'])

# Natural language explanation
print(result['explanation'])

# Complete trace (JSON)
print(json.dumps(result['trace'], indent=2))
```

## Architecture

### 1. Parsing (`parsing.py`)
- Converts natural language to canonical form
- Extracts objective and constraints
- Validates differentiability

### 2. Structure Inference (`structure_inference.py`)
- **Convexity detection**: Sylvester's criterion (symbolic) or eigenvalue sampling
- **Uncertainty quantification**: Wilson score confidence intervals
- **Smoothness analysis**: Hessian computation
- **Condition number estimation**: Median over samples
- **LICQ checking**: Constraint Jacobian rank

### 3. Algorithm Selection (`algorithm_selection.py`)
- **Unconstrained convex**: L-BFGS or Gradient Descent
- **Unconstrained nonconvex**: Trust-Region Newton
- **Constrained with LICQ**: SLSQP or Interior Point
- **LICQ violated**: Augmented Lagrangian

### 4. Solver Execution (`solver_execution.py`)
- Automatic differentiation (SymPy + JAX-ready)
- Input normalization
- Multiple solver implementations
- Multiplier recovery for verification

### 5. Verification (`verification.py`)
- **Unconstrained**: ∇f(x*) ≈ 0, H(x*) ≻ 0
- **Constrained**: Full KKT conditions
- **SOSC**: Projected Hessian positive definiteness
- Detailed failure diagnostics

## Structure Report

```python
{
  "convexity_status": "CONVEX",          # CONVEX, NONCONVEX, UNCERTAIN
  "convexity_confidence": 1.0,           # 0.0 to 1.0
  "convexity_method": "symbolic",        # symbolic or sampling
  "convexity_ci": [0.95, 1.00],         # Confidence interval (if sampling)
  "condition_number_median": 1.5,
  "is_smooth": true,
  "licq_status": "SATISFIED"
}
```

## Verification Output

```python
{
  "status": "PASS",                      # PASS or FAIL
  "gradient_norm": 1.2e-7,
  "kkt_stationarity": 1.5e-8,           # Constrained only
  "kkt_primal_feasibility": 3.2e-9,
  "kkt_complementarity": 2.1e-9,
  "sosc_status": "PASS",
  "hessian_eigenvalues": [2.0, 2.0]
}
```

## Examples

See `examples.py` for complete demonstrations:

```bash
python examples.py
```

### Example 1: Simple Quadratic
```python
optimize("minimize x**2 + y**2")
# Solution: [0, 0], Verified: PASS
```

### Example 2: Rosenbrock (Multi-start)
```python
optimize("minimize (1-x)**2 + 100*(y-x**2)**2", multi_start=5)
# Solution: ~[1, 1], Shows all start results
```

### Example 3: Constrained
```python
optimize("minimize x**2 + y**2 subject to x + y >= 1")
# Uses SLSQP, verifies KKT conditions
```

## Testing

Run the test suite:

```bash
python test_suite.py
```

Tests cover:
- Convex quadratics
- Nonconvex problems (Rosenbrock)
- Constrained optimization
- Least squares
- Multi-dimensional problems

## Limitations

### Explicit Scope Restrictions
- **Continuous variables only** (no discrete optimization)
- **Differentiable objectives** (requires gradients/Hessians)
- **Local optimality** for nonconvex (use multi-start for better solutions)
- **Curse of dimensionality** (n > 20 flagged as UNCERTAIN if using sampling)

### Known Issues
- Symbolic analysis limited to ~50 variables
- LICQ may fail at degenerate points
- Multiplier recovery approximate for some solvers

## Advanced Usage

### Custom Initial Points
```python
import numpy as np

x0 = np.array([0.5, 1.5])
result = optimize("minimize x**2 + y**2", initial_point=x0)
```

### Lower Tolerance
```python
result = optimize(
    description="minimize x**2 + y**2",
    tolerance=1e-8  # Stricter verification
)
```

### Accessing Full Trace
```python
trace = result['trace']

# Solver selection reasoning
print(trace['solver_selection']['reason'])

# All objective values from multi-start
if trace['solver_selection']['multi_start_used']:
    print(trace['execution']['objective_values_all_starts'])
```

## Implementation Notes

### Uncertainty Quantification
When symbolic convexity analysis is inconclusive, the system:
1. Samples N points from feasible region
2. Computes Hessian eigenvalues at each point
3. Calculates empirical proportion of PSD matrices
4. Reports Wilson score 95% confidence interval
5. Flags high-dimensional problems as UNCERTAIN

### Multiplier Recovery
For solvers that don't track multipliers (L-BFGS, GD):
```
min_{μ,λ} ||∇f(x*) + Σμᵢ∇gᵢ(x*) + Σλⱼ∇hⱼ(x*)||²
s.t. μ ≥ 0
```
Solved via constrained least-squares.

### Multi-Start Strategy
For nonconvex/uncertain problems:
1. Generate K random initial points
2. Run solver from each
3. Return best verified solution
4. Report distribution of objective values

## Mathematical Guarantees

### Convex Problems
- If verification passes and problem is CONVEX (symbolic): **global optimum certified**
- Tolerance: ‖∇f(x*)‖ < 10⁻⁶, λₘᵢₙ(H(x*)) > 10⁻⁶

### Nonconvex Problems
- Verification certifies **local optimality only**
- Multi-start improves solution quality (no guarantee)
- Reports: "This verification only certifies LOCAL optimality"

### Constrained Problems
- Full KKT verification when LICQ satisfied
- Routes to penalty methods when LICQ violated
- SOSC checked on tangent space

## Contributing

This implementation follows the design in `document_revised.pdf`. Key principles:
- **Explicit uncertainty** over false confidence
- **Honest limitations** in explanations
- **Complete audit trails** in reasoning traces
- **Mathematical rigor** in verification

## References

Based on:
- Nocedal & Wright, "Numerical Optimization"
- Boyd & Vandenberghe, "Convex Optimization"
- Conn, Gould & Toint, "Trust-Region Methods"

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{optimai2024,
  title={OptimAI: Structure-Aware Optimization with Formal Verification},
  author={OptimAI Team},
  year={2024},
  url={https://github.com/optimai/optimai}
}
```
