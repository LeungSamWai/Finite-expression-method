"""
Example usage of OptimAI.
"""
import sys
sys.path.append('/home/claude/optimai')

from optimai import optimize
import json


def example1_simple_quadratic():
    """Example 1: Simple quadratic minimization."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Quadratic")
    print("="*70)
    
    result = optimize(
        description="minimize x**2 + y**2",
        sample_size=50,
        max_iter=500,
        multi_start=1
    )
    
    print("\nSolution found:", result['solution'])
    print("Objective value:", result['objective_value'])
    print("Verification:", result['verification'])


def example2_rosenbrock():
    """Example 2: Rosenbrock function with multi-start."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Rosenbrock Function (Multi-start)")
    print("="*70)
    
    result = optimize(
        description="minimize (1-x)**2 + 100*(y-x**2)**2",
        sample_size=50,
        max_iter=1000,
        multi_start=5  # Use 5 random starts
    )
    
    print("\nSolution found:", result['solution'])
    print("Objective value:", result['objective_value'])
    print("Verification:", result['verification'])
    
    # Show multi-start results
    if 'trace' in result:
        obj_values = result['trace']['execution'].get('objective_values_all_starts')
        if obj_values:
            print("\nObjective values from all starts:")
            for i, val in enumerate(obj_values):
                print(f"  Start {i+1}: {val:.6e}")


def example3_constrained():
    """Example 3: Constrained optimization."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Constrained Optimization")
    print("="*70)
    
    result = optimize(
        description="minimize x**2 + y**2 subject to x + y >= 1",
        sample_size=50,
        max_iter=500
    )
    
    print("\nSolution found:", result['solution'])
    print("Objective value:", result['objective_value'])
    print("Verification:", result['verification'])
    
    # Check constraint satisfaction
    x, y = result['solution']
    constraint_value = x + y
    print(f"\nConstraint x + y >= 1: {constraint_value:.6f} "
          f"({'satisfied' if constraint_value >= 0.999 else 'violated'})")


def example4_viewing_trace():
    """Example 4: Viewing the complete reasoning trace."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Viewing Complete Reasoning Trace")
    print("="*70)
    
    result = optimize(
        description="minimize (x-1)**2 + (y-2)**2",
        sample_size=30
    )
    
    print("\n--- PROBLEM ---")
    print(json.dumps(result['trace']['problem'], indent=2))
    
    print("\n--- STRUCTURAL ANALYSIS ---")
    print(json.dumps(result['trace']['structural_analysis'], indent=2))
    
    print("\n--- SOLVER SELECTION ---")
    print(json.dumps(result['trace']['solver_selection'], indent=2))
    
    print("\n--- EXECUTION ---")
    print(json.dumps(result['trace']['execution'], indent=2))
    
    print("\n--- VERIFICATION ---")
    verification_summary = {
        k: v for k, v in result['trace']['verification'].items()
        if k != 'hessian_eigenvalues'  # Skip eigenvalues for brevity
    }
    print(json.dumps(verification_summary, indent=2))


def example5_ill_conditioned():
    """Example 5: Ill-conditioned problem."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Ill-conditioned Problem")
    print("="*70)
    
    # Create an ill-conditioned quadratic
    result = optimize(
        description="minimize x**2 + 10000*y**2",
        sample_size=50,
        max_iter=1000
    )
    
    print("\nSolution found:", result['solution'])
    print("Objective value:", result['objective_value'])
    print("Verification:", result['verification'])
    
    # Show condition number
    if 'trace' in result:
        cond_num = result['trace']['structural_analysis']['condition_number_median']
        print(f"\nCondition number: {cond_num:.2e}")
        print(f"Problem is {'ill-conditioned' if cond_num > 1000 else 'well-conditioned'}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("OPTIMAI EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate the capabilities of OptimAI:")
    print("- Structure-aware solver selection")
    print("- Multi-start for nonconvex problems")  
    print("- Constrained optimization")
    print("- Complete reasoning traces")
    print("- Handling ill-conditioned problems")
    
    try:
        example1_simple_quadratic()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example2_rosenbrock()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example3_constrained()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example4_viewing_trace()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    try:
        example5_ill_conditioned()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    run_all_examples()
