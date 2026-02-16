"""
Comprehensive demonstration of OptimAI capabilities.
"""
import sys
sys.path.insert(0, '/home/claude/optimai')

from optimai import optimize
import json


def demo():
    """Run comprehensive demo."""
    
    print("="*80)
    print(" "*20 + "OPTIMAI DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases OptimAI's key capabilities:")
    print("  1. Structure-aware algorithm selection")
    print("  2. Formal verification with KKT conditions")
    print("  3. Uncertainty quantification")
    print("  4. Multi-start for nonconvex problems")
    print("  5. Complete reasoning traces")
    print("\n" + "="*80)
    
    # Demo 1: Simple Convex Problem
    print("\n\nDEMO 1: SIMPLE CONVEX QUADRATIC")
    print("-" * 80)
    print("Problem: minimize x² + y²")
    print("Expected: x* = [0, 0], f* = 0 (global optimum)\n")
    
    result1 = optimize("minimize x**2 + y**2", sample_size=30, max_iter=200)
    
    print(f"\n✓ Solution: {result1['solution']}")
    print(f"✓ Objective: {result1['objective_value']:.2e}")
    print(f"✓ Verification: {result1['verification']}")
    
    # Show structural analysis
    struct = result1['trace']['structural_analysis']
    print(f"\nStructural Analysis:")
    print(f"  - Convexity: {struct['convexity_status']} (confidence: {struct['convexity_confidence']:.0%})")
    print(f"  - Method: {struct['convexity_method']}")
    print(f"  - Solver selected: {result1['trace']['solver_selection']['chosen']}")
    
    # Demo 2: Nonconvex with Multi-Start
    print("\n\n" + "="*80)
    print("\nDEMO 2: NONCONVEX PROBLEM WITH MULTI-START")
    print("-" * 80)
    print("Problem: minimize (1-x)² + 100(y-x²)²  (Rosenbrock function)")
    print("Expected: x* = [1, 1], f* = 0")
    print("Note: This is a challenging nonconvex problem with a narrow curved valley\n")
    
    result2 = optimize(
        "minimize (1-x)**2 + 100*(y-x**2)**2",
        sample_size=30,
        max_iter=1000,
        multi_start=5
    )
    
    print(f"\n✓ Solution: {result2['solution']}")
    print(f"✓ Objective: {result2['objective_value']:.2e}")
    print(f"✓ Verification: {result2['verification']}")
    
    # Show multi-start results
    obj_vals = result2['trace']['execution'].get('objective_values_all_starts', [])
    if obj_vals:
        print(f"\nMulti-start results (5 random initializations):")
        for i, val in enumerate(obj_vals):
            marker = "←" if val == min(obj_vals) else " "
            print(f"  Start {i+1}: f = {val:.6e} {marker}")
        print(f"\n  Improvement from multi-start: {max(obj_vals)/min(obj_vals):.1f}x better")
    
    # Demo 3: Constrained Optimization
    print("\n\n" + "="*80)
    print("\nDEMO 3: CONSTRAINED OPTIMIZATION")
    print("-" * 80)
    print("Problem: minimize x² + y²  subject to  x + y ≥ 1")
    print("Expected: x* = [0.5, 0.5], f* = 0.5\n")
    
    result3 = optimize(
        "minimize x**2 + y**2 subject to x + y >= 1",
        sample_size=30,
        max_iter=500
    )
    
    print(f"\n✓ Solution: {result3['solution']}")
    print(f"✓ Objective: {result3['objective_value']:.6f}")
    print(f"✓ Verification: {result3['verification']}")
    
    # Check constraint
    x, y = result3['solution']
    constraint_val = x + y
    print(f"\nConstraint verification:")
    print(f"  x + y = {constraint_val:.6f} {'≥' if constraint_val >= 0.999 else '<'} 1.0")
    print(f"  Status: {'✓ Satisfied' if constraint_val >= 0.999 else '✗ Violated'}")
    
    # Show KKT verification
    verif = result3['trace']['verification']
    print(f"\nKKT Conditions:")
    print(f"  - Stationarity: {verif['kkt_stationarity']:.2e}")
    print(f"  - Primal feasibility: {verif['kkt_primal_feasibility']:.2e}")
    print(f"  - Complementarity: {verif['kkt_complementarity']:.2e}")
    
    # Demo 4: Uncertainty Quantification
    print("\n\n" + "="*80)
    print("\nDEMO 4: UNCERTAINTY QUANTIFICATION")
    print("-" * 80)
    print("Problem: minimize x⁴ - x² + y⁴ - y²  (nonconvex with multiple local minima)")
    print("Note: Symbolic convexity test will be inconclusive → sampling with CI\n")
    
    result4 = optimize(
        "minimize x**4 - x**2 + y**4 - y**2",
        sample_size=50,
        max_iter=500,
        multi_start=3
    )
    
    print(f"\n✓ Solution: {result4['solution']}")
    print(f"✓ Objective: {result4['objective_value']:.6f}")
    print(f"✓ Verification: {result4['verification']}")
    
    struct4 = result4['trace']['structural_analysis']
    print(f"\nConvexity Analysis:")
    print(f"  - Status: {struct4['convexity_status']}")
    print(f"  - Empirical proportion PSD: {struct4['convexity_confidence']:.0%}")
    
    if struct4.get('convexity_ci'):
        ci_low, ci_high = struct4['convexity_ci']
        print(f"  - 95% Confidence Interval: [{ci_low:.0%}, {ci_high:.0%}]")
        print(f"  - Method: {struct4['convexity_method']}")
    
    # Demo 5: Complete Reasoning Trace
    print("\n\n" + "="*80)
    print("\nDEMO 5: COMPLETE REASONING TRACE (JSON)")
    print("-" * 80)
    print("Every OptimAI run produces a complete, auditable trace.\n")
    
    result5 = optimize("minimize (x-1)**2 + (y-2)**2", sample_size=20)
    
    print("Sample trace structure:")
    print(json.dumps({
        "problem": result5['trace']['problem'],
        "structural_analysis": {
            k: v for k, v in result5['trace']['structural_analysis'].items()
            if k not in ['convexity_ci']  # Simplify output
        },
        "solver_selection": result5['trace']['solver_selection'],
        "execution": {
            k: v for k, v in result5['trace']['execution'].items()
            if k not in ['objective_values_all_starts']
        },
        "verification": {
            k: v for k, v in result5['trace']['verification'].items()
            if k not in ['hessian_eigenvalues', 'details']
        }
    }, indent=2))
    
    # Summary
    print("\n\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)
    
    print("\n✓ All demos completed successfully!")
    print("\nKey Takeaways:")
    print("  1. Convex problems are identified and globally optimized")
    print("  2. Nonconvex problems use multi-start and report local optimality")
    print("  3. Constraints are handled with KKT verification")
    print("  4. Uncertainty is quantified with confidence intervals")
    print("  5. Complete reasoning traces enable full auditability")
    
    print("\n" + "="*80)
    print("For more examples, see examples.py")
    print("For tests, run: python test_suite.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    demo()
