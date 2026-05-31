import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import project2_pde_methods as pde


class Project2PdeMethodTests(unittest.TestCase):
    def test_laplace_boundary_and_exact_solution(self):
        problem = pde.LAPLACE_PROBLEM
        self.assertAlmostEqual(problem.boundary(1.0, 0.75), 75.0)
        self.assertAlmostEqual(problem.boundary(0.25, 1.0), 25.0)
        self.assertAlmostEqual(float(problem.exact(np.array(0.5), np.array(0.5))), 25.0)

    def test_poisson_source_matches_manufactured_solution(self):
        x = np.array(0.5)
        y = np.array(0.5)
        source = float(pde.POISSON_PROBLEM.source(x, y))
        self.assertAlmostEqual(source, -2.0 * math.pi**2)
        self.assertAlmostEqual(float(pde.POISSON_PROBLEM.exact(x, y)), 1.0)

    def test_elliptic_matrix_dimensions(self):
        matrix, rhs = pde.build_elliptic_system(pde.LAPLACE_PROBLEM, 0.25)
        self.assertEqual(matrix.shape, (9, 9))
        self.assertEqual(rhs.shape, (9,))

    def test_laplace_errors_decrease_under_refinement(self):
        errors = [
            pde.max_error(pde.solve_elliptic_direct(pde.LAPLACE_PROBLEM, h))
            for h in pde.ELLIPTIC_MESHES
        ]
        self.assertLess(errors[-1], 1e-10)
        self.assertLessEqual(errors[-1], errors[0] + 1e-12)

    def test_heat_method_shapes(self):
        case = pde.HeatCase("shape", "shape test", 1, 0.1, 0.004)
        for method in pde.HEAT_METHODS:
            result = method(case)
            self.assertEqual(result.numerical.shape, result.exact.shape)
            self.assertEqual(result.numerical.shape, (26, 11))

    def test_forward_stable_case_improves_with_refinement(self):
        coarse = pde.forward_difference(pde.HeatCase("coarse", "coarse", 1, 0.1, 0.004))
        fine = pde.forward_difference(pde.HeatCase("fine", "fine", 1, 0.05, 0.001))
        self.assertGreater(pde.final_error(coarse), pde.final_error(fine))

    def test_implicit_methods_remain_bounded_for_larger_lambda(self):
        case = pde.HeatCase("large_lambda", "large lambda", 1, 0.1, 0.02)
        for method in (pde.backward_difference, pde.crank_nicolson):
            result = method(case)
            self.assertTrue(np.all(np.isfinite(result.numerical)))
            self.assertLess(np.max(np.abs(result.numerical)), 2.0)


if __name__ == "__main__":
    unittest.main()
