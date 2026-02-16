"""
Test suite for OptimAI.
"""
import numpy as np
import sys
sys.path.append('/home/claude/optimai')

from optimai import optimize


def test_simple_quadratic():
    """Test 1: Simple convex quadratic."""
    print("\n" + "="*70)
    print("TEST 1: Simple Quadratic (Convex)")
    print("="*70)
    
    description = "minimize x**2 + y**2"
    result = optimize(description, sample_size=50, multi_start=1)
    
    print(f"\nResult status: {result['status']}")
    print(f"Solution: {result['solution']}")
    print(f"Objective value: {result['objective_value']}")
    
    # Expected: [0, 0]
    assert result['verification'] == 'PASS', "Verification should pass"
    assert abs(result['objective_value']) < 1e-4, "Objective should be near 0"
    
    return result


def test_rosenbrock():
    """Test 2: Rosenbrock function (nonconvex)."""
    print("\n" + "="*70)
    print("TEST 2: Rosenbrock Function (Nonconvex)")
    print("="*70)
    
    description = "minimize (1-x)**2 + 100*(y-x**2)**2"
    result = optimize(description, sample_size=50, multi_start=5, max_iter=1000)
    
    print(f"\nResult status: {result['status']}")
    print(f"Solution: {result['solution']}")
    print(f"Objective value: {result['objective_value']}")
    
    # Expected: [1, 1] with f = 0
    x_star = np.array(result['solution'])
    expected = np.array([1.0, 1.0])
    error = np.linalg.norm(x_star - expected)
    
    print(f"Distance to true optimum [1, 1]: {error:.6f}")
    
    return result


def test_constrained_quadratic():
    """Test 3: Constrained quadratic."""
    print("\n" + "="*70)
    print("TEST 3: Constrained Quadratic")
    print("="*70)
    
    description = "minimize x**2 + y**2 subject to x + y >= 1"
    result = optimize(description, sample_size=50, max_iter=500)
    
    print(f"\nResult status: {result['status']}")
    print(f"Solution: {result['solution']}")
    print(f"Objective value: {result['objective_value']}")
    
    # Expected: [0.5, 0.5] with f = 0.5
    x_star = np.array(result['solution'])
    expected = np.array([0.5, 0.5])
    error = np.linalg.norm(x_star - expected)
    
    print(f"Distance to expected optimum [0.5, 0.5]: {error:.6f}")
    
    return result


def test_least_squares():
    """Test 4: Least squares."""
    print("\n" + "="*70)
    print("TEST 4: Least Squares (Convex)")
    print("="*70)
    
    description = "minimize (x-1)**2 + (y-2)**2 + (x+y-3)**2"
    result = optimize(description, sample_size=50)
    
    print(f"\nResult status: {result['status']}")
    print(f"Solution: {result['solution']}")
    print(f"Objective value: {result['objective_value']}")
    
    assert result['verification'] == 'PASS', "Verification should pass"
    
    return result


def test_three_variables():
    """Test 5: Three variables."""
    print("\n" + "="*70)
    print("TEST 5: Three Variables")
    print("="*70)
    
    description = "minimize x**2 + 2*y**2 + 3*z**2"
    result = optimize(description, sample_size=50)
    
    print(f"\nResult status: {result['status']}")
    print(f"Solution: {result['solution']}")
    print(f"Objective value: {result['objective_value']}")
    
    # Expected: [0, 0, 0]
    assert result['verification'] == 'PASS', "Verification should pass"
    assert abs(result['objective_value']) < 1e-4, "Objective should be near 0"
    
    return result


def test_nonconvex_with_constraint():
    """Test 6: Nonconvex with constraint."""
    print("\n" + "="*70)
    print("TEST 6: Nonconvex with Constraint")
    print("="*70)
    
    description = "minimize x**4 + y**4 subject to x + y >= 1"
    result = optimize(description, sample_size=50, multi_start=3)
    
    print(f"\nResult status: {result['status']}")
    print(f"Solution: {result['solution']}")
    print(f"Objective value: {result['objective_value']}")
    
    return result


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING OPTIMAI TEST SUITE")
    print("="*70)
    
    results = {}
    
    try:
        results['quadratic'] = test_simple_quadratic()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        results['quadratic'] = None
    
    try:
        results['rosenbrock'] = test_rosenbrock()
    except Exception as e:
        print(f"Test 2 failed: {e}")
        results['rosenbrock'] = None
    
    try:
        results['constrained'] = test_constrained_quadratic()
    except Exception as e:
        print(f"Test 3 failed: {e}")
        results['constrained'] = None
    
    try:
        results['least_squares'] = test_least_squares()
    except Exception as e:
        print(f"Test 4 failed: {e}")
        results['least_squares'] = None
    
    try:
        results['three_vars'] = test_three_variables()
    except Exception as e:
        print(f"Test 5 failed: {e}")
        results['three_vars'] = None
    
    try:
        results['nonconvex_constrained'] = test_nonconvex_with_constraint()
    except Exception as e:
        print(f"Test 6 failed: {e}")
        results['nonconvex_constrained'] = None
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r and r['verification'] == 'PASS')
    total = len(results)
    
    print(f"Passed verification: {passed}/{total}")
    
    for name, result in results.items():
        if result:
            status = result['verification']
            print(f"  {name}: {status}")
        else:
            print(f"  {name}: ERROR")
    
    return results


if __name__ == "__main__":
    run_all_tests()
