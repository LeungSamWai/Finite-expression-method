# OptimAI Quick Start Guide

## Installation (2 minutes)

```bash
cd optimai
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- numpy, scipy, sympy

## Your First Optimization (30 seconds)

```python
from optimai import optimize

# Solve a simple problem
result = optimize("minimize x**2 + y**2")

print(result['solution'])      # [0.0, 0.0]
print(result['objective_value'])  # ~0.0
print(result['verification'])  # PASS
```

## Run Examples (5 minutes)

```bash
# Comprehensive demo with 5 problems
python demo.py

# Detailed examples with commentary
python examples.py

# Run test suite (6 tests)
python test_suite.py
```

## Common Use Cases

### 1. Minimize a Function
```python
result = optimize("minimize (x-1)**2 + (y-2)**2")
# Solution: [1.0, 2.0]
```

### 2. Nonconvex Problem (use multi-start)
```python
result = optimize(
    "minimize (1-x)**2 + 100*(y-x**2)**2",  # Rosenbrock
    multi_start=5  # Try 5 random initializations
)
# Solution: ~[1.0, 1.0]
```

### 3. Constrained Problem
```python
result = optimize(
    "minimize x**2 + y**2 subject to x + y >= 1"
)
# Solution: [0.5, 0.5]
# Constraint: x + y = 1.0 ✓
```

### 4. View Complete Reasoning
```python
import json

result = optimize("minimize x**2 + y**2")

# Natural language explanation
print(result['explanation'])

# Complete JSON trace
print(json.dumps(result['trace'], indent=2))
```

## Understanding Results

Every result contains:

```python
{
  'status': 'SUCCESS',           # or 'COMPLETED_WITH_WARNINGS'
  'solution': [0.0, 0.0],       # Optimal point x*
  'objective_value': 0.0,        # f(x*)
  'verification': 'PASS',        # or 'FAIL'
  'explanation': '...',          # Natural language
  'trace': {...},                # Complete audit trail
  'warnings': []                 # List of warnings
}
```

### Verification Status
- **PASS**: Solution satisfies optimality conditions
  - Convex: Global optimum certified
  - Nonconvex: Local optimum certified
- **FAIL**: Solution did not pass verification
  - See warnings and trace for details

## Key Parameters

```python
optimize(
    description="...",    # Problem in natural language
    
    # Structure inference
    sample_size=100,      # Samples for convexity detection
    
    # Solver
    max_iter=500,         # Maximum iterations
    tolerance=1e-6,       # Verification tolerance
    
    # Multi-start (for nonconvex)
    multi_start=1,        # Number of random starts
    
    # Optional
    initial_point=None,   # Starting point (numpy array)
    format="natural_language"  # or "latex"
)
```

## Problem Formats

### Natural Language (default)
```python
optimize("minimize x**2 + y**2")
optimize("minimize x**2 + y**2 subject to x + y >= 1")
optimize("minimize (1-x)**2 + 100*(y-x**2)**2")
```

### LaTeX
```python
optimize(
    r"\min x^2 + y^2 \text{ s.t. } x + y \geq 1",
    format="latex"
)
```

## Interpreting Warnings

Common warnings and their meanings:

1. **"Convexity could not be reliably determined"**
   - Symbolic analysis failed
   - Using sampling-based inference
   - Check confidence interval in trace

2. **"Verification failed: ..."**
   - Solution may not be optimal
   - System attempted recovery
   - See trace for details

3. **"Solution did not pass verification"**
   - First-order or second-order conditions not satisfied
   - Consider: different solver, more iterations, reformulation

## Advanced Usage

### Custom Initial Point
```python
import numpy as np

x0 = np.array([0.5, 1.5])
result = optimize("minimize x**2 + y**2", initial_point=x0)
```

### Strict Tolerances
```python
result = optimize(
    "minimize x**2 + y**2",
    tolerance=1e-8,  # Stricter than default 1e-6
    max_iter=1000
)
```

### Extract Specific Information
```python
result = optimize("minimize x**2 + y**2")

# Solver used
solver = result['trace']['solver_selection']['chosen']
print(f"Solver: {solver}")

# Convexity analysis
convex = result['trace']['structural_analysis']['convexity_status']
confidence = result['trace']['structural_analysis']['convexity_confidence']
print(f"Convexity: {convex} (confidence: {confidence:.0%})")

# Iterations
iters = result['trace']['execution']['iterations']
print(f"Iterations: {iters}")
```

## Troubleshooting

### Problem: "Could not parse objective"
**Solution**: Use Python syntax (x**2 not x^2)

### Problem: "Verification failed"
**Solutions**:
1. Increase max_iter: `max_iter=1000`
2. Use multi-start: `multi_start=5`
3. Try different solver (system tries automatically)
4. Check if problem is well-formulated

### Problem: "Convexity UNCERTAIN"
**Explanation**: 
- Problem dimension > 20, or
- Symbolic analysis inconclusive
- Still solvable! Uses robust solvers
- Consider multi-start for better solutions

### Problem: Slow performance
**Solutions**:
1. Reduce sample_size: `sample_size=50`
2. Use simpler structure inference
3. For production: cache structure analysis

## What's Happening Under the Hood

When you call `optimize()`:

1. **Parse** → Convert text to mathematical form
2. **Infer Structure** → Analyze convexity, smoothness, constraints
3. **Select Algorithm** → Choose best solver based on structure
4. **Execute** → Run solver (possibly with multi-start)
5. **Verify** → Check KKT conditions, SOSC
6. **Recovery** → If failed, try alternative approaches
7. **Report** → Generate explanation and trace

## Next Steps

- Read full documentation: `README.md`
- Try examples: `python examples.py`
- Run tests: `python test_suite.py`
- Read implementation summary: `IMPLEMENTATION_SUMMARY.md`
- Explore the code: Start with `optimai.py`

## Quick Reference Card

```python
# Basic
from optimai import optimize
result = optimize("minimize f(x)")

# Multi-start (nonconvex)
result = optimize("minimize f(x)", multi_start=5)

# Constrained
result = optimize("minimize f(x) subject to g(x) <= 0")

# Access results
x_star = result['solution']
f_star = result['objective_value']
verified = result['verification'] == 'PASS'

# View explanation
print(result['explanation'])

# View trace
import json
print(json.dumps(result['trace'], indent=2))
```

## Support

For questions:
1. Check README.md for detailed documentation
2. Review examples.py for usage patterns
3. Inspect test_suite.py for edge cases
4. Examine trace output for debugging

Happy optimizing! 🎯
