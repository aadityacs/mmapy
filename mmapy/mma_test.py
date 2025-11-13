"""Tests for mma."""

from absl.testing import absltest
from jax import value_and_grad
import jax.numpy as jnp
import numpy as np

import mma


class MmaTest(absltest.TestCase):
  def test_mma_state_to_array(self):
    mma_state = mma.MMAState(
      x=np.array([[1.0, 2.0, 3.0]]).T,
      x_old_1=np.array([[4.0, 5.0, 6.0]]).T,
      x_old_2=np.array([[7.0, 8.0, 9.0]]).T,
      low=np.array([[0.0, 0.0, 0.0]]).T,
      upp=np.array([[10.0, 10.0, 10.0]]).T,
      is_converged=True,
      epoch=20,
      kkt_norm=10.0,
      change_design_var=20.0,
    )
    mma_state_array = mma_state.to_array()
    expected_array = np.array(
      [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
        1,
        20,
        10.0,
        20.0,
      ]
    )
    np.testing.assert_array_equal(mma_state_array, expected_array)

  def test_mma_state_from_array(self):
    mma_state_array = np.array(
      [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
        1,
        20,
        10.0,
        20.0,
      ]
    )
    mma_state = mma.MMAState.from_array(mma_state_array, num_design_var=3)
    expected_mma_state = mma.MMAState(
      x=np.array([[1.0, 2.0, 3.0]]).T,
      x_old_1=np.array([[4.0, 5.0, 6.0]]).T,
      x_old_2=np.array([[7.0, 8.0, 9.0]]).T,
      low=np.array([[0.0, 0.0, 0.0]]).T,
      upp=np.array([[10.0, 10.0, 10.0]]).T,
      is_converged=True,
      epoch=20,
      kkt_norm=10.0,
      change_design_var=20.0,
    )
    np.testing.assert_array_equal(mma_state.x, expected_mma_state.x)
    np.testing.assert_array_equal(mma_state.x_old_1, expected_mma_state.x_old_1)
    np.testing.assert_array_equal(mma_state.x_old_2, expected_mma_state.x_old_2)
    np.testing.assert_array_equal(mma_state.low, expected_mma_state.low)
    np.testing.assert_array_equal(mma_state.upp, expected_mma_state.upp)
    np.testing.assert_equal(mma_state.is_converged, expected_mma_state.is_converged)
    np.testing.assert_equal(mma_state.epoch, expected_mma_state.epoch)
    np.testing.assert_equal(mma_state.kkt_norm, expected_mma_state.kkt_norm)
    np.testing.assert_equal(
      mma_state.change_design_var, expected_mma_state.change_design_var
    )

  def test_mma_state_from_array_raises_error_if_passed_wrong_num_desvar(self):
    mma_state_array = np.array(
      [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
        1,
        20,
        10.0,
        20.0,
      ]
    )
    with self.assertRaisesRegex(ValueError, "`state_array` shape is incompatible with"):
      _ = mma.MMAState.from_array(mma_state_array, num_design_var=2)

  def test_single_variable_objective_with_no_constraint(self):
    """Test if mma handles single variable optimization correctly.

    Objective: min (x-50)^2 + 25
    Constraint: -
    bounds: 1 <= x <= 100
    """

    # we need at least one constraint and hence pass a dummy one
    def dummy_constraint(x):
      del x
      return (
        np.array([0.0]).reshape((num_cons, 1)),
        np.array([0.0]).reshape((num_cons, num_design_var)),
      )

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 50.0) ** 2 + 25.0

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.random.uniform(0.0, 100.0, (1)).reshape((-1, 1))
    num_design_var = 1
    num_cons = 1
    lower_bound = np.zeros((num_design_var, 1))
    upper_bound = 100.0 * np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-3,
      step_tol=1e-3,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)

      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state = mma.update_mma(
        mma_state, mma_params, objective, grad_obj, constr, grad_cons
      )

    self.assertAlmostEqual(mma_state.x[0, 0], 50.0, places=2)

  def test_two_variable_optimization_with_no_constraints(self):
    """Test if mma handles two variable optimization correctly.

    Objective: min (x-30)^2 + (y-1)^2 + 1000
    constraints -
    bounds: 0 <= x, y <= 1000
    """

    # we need at least one constraint and hence pass a dummy one
    def dummy_constraint(x):
      del x
      return (
        np.array([0.0]).reshape((num_cons, 1)),
        np.array([0.0, 0.0]).reshape((num_cons, num_design_var)),
      )

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 30.0) ** 2 + (x[1, 0] - 1.0) ** 2 + 1000.0

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.random.uniform(0.0, 100.0, (2)).reshape((-1, 1))
    num_design_var = 2
    num_cons = 1
    lower_bound = np.zeros((num_design_var, 1))
    upper_bound = 1000.0 * np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-3,
      step_tol=1e-3,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)

      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state = mma.update_mma(
        mma_state, mma_params, objective, grad_obj, constr, grad_cons
      )

    self.assertAlmostEqual(np.abs(mma_state.x[0, 0] - 30.0) / 30.0, 0.0, places=1)
    self.assertAlmostEqual(np.abs(mma_state.x[1, 0] - 1.0), 0.0, places=1)

  def test_gomez_and_levy_constraint_problem(self):
    """Test if mma handles 2D gomes and levy optimization correctly.

    Obj: min 4*x**2 - 2.1*x**4 + 0.333*x**6 + x*y - 4*y**2 + 4*y**4
    constraints: -sin(4*pi*x) + 2*sin(2*pi*y)**2 <= 1.5
    bounds: -1. <= x, y <= 1.
    expected result: x = 0.089 , y = -0.71
    """

    # we need at least one constraint and hence pass a dummy one
    def dummy_constraint(x):
      def confn(x):
        return (
          -jnp.sin(4 * np.pi * x[0, 0]) + 2 * jnp.sin(2 * np.pi * x[1, 0]) ** 2 - 1.5
        )

      c, dc = value_and_grad(confn)(x)
      return (
        np.array(c).reshape((num_cons, 1)),
        np.array(dc).reshape((num_cons, num_design_var)),
      )

    def objective_fn(x):
      def objfn(xy):
        x = xy[0, 0]
        y = xy[1, 0]
        return 4 * x**2 - 2.1 * x**4 + 0.333 * x**6 + x * y - 4 * y**2 + 4 * y**4

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.zeros((2)).reshape((-1, 1))
    num_design_var = 2
    num_cons = 1
    lower_bound = -np.ones((num_design_var, 1))
    upper_bound = np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-6,
      step_tol=1e-6,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = dummy_constraint(mma_state.x)
      mma_state = mma.update_mma(
        mma_state, mma_params, objective, grad_obj, constr, grad_cons
      )

    self.assertAlmostEqual(mma_state.x[0, 0], 0.089, places=1)
    self.assertAlmostEqual(mma_state.x[1, 0], -0.71, places=1)

  def test_sphere_function(self):
    """Test if MMA handles multivariable optimization correctly.

    Objective: min f(x1, x2, ..., xn) = x1^2 + x2^2 + ... + xn^2
    bounds: -10 < xi < 10 , i = 1,2,...,100

    expected result:
      xi* = 0 , i = 1,2,...,100
      f(x*) = 0
    """
    num_design_var = 100
    num_cons = 1  # dummy constraint

    def dummy_constraint(x):
      del x
      return (np.zeros((num_cons, 1)), np.zeros((num_cons, num_design_var)))

    def objective_fn(x):
      def objfn(x):
        return jnp.sum(x**2)

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.random.uniform(-10, 10, (num_design_var, 1))
    lower_bound = -10 * np.ones((num_design_var, 1))
    upper_bound = 10 * np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
      max_iter=500,
      kkt_tol=1e-2,
      step_tol=1e-2,
      move_limit=5e-3,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)

      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state = mma.update_mma(
        mma_state, mma_params, objective, grad_obj, constr, grad_cons
      )

    np.testing.assert_almost_equal(objective, 0.0, decimal=3)
    for i in range(num_design_var):
      np.testing.assert_almost_equal(mma_state.x[i, 0], 0.0, decimal=3)

  def test_beale_function(self):
    """Test if MMA handles multivariable optimization correctly.

    Objective: min f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 +
                            (2.625 - x + xy^3)^2
    bounds: -4.5 < x, y < 4.5

    expected result:
      x* = 3 , y* = 0.5
      f(x*, y*) = 0
    """
    num_design_var = 2
    num_cons = 1  # dummy constraint

    def dummy_constraint(x):
      del x
      return (np.zeros((num_cons, 1)), np.zeros((num_cons, num_design_var)))

    def objective_fn(x):
      def objfn(x):
        t1 = (1.5 - x[0, 0] + x[0, 0] * x[1, 0]) ** 2
        t2 = (2.25 - x[0, 0] + x[0, 0] * x[1, 0] ** 2) ** 2
        t3 = (2.625 - x[0, 0] + x[0, 0] * x[1, 0] ** 3) ** 2
        return t1 + t2 + t3

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.zeros((num_design_var, 1))
    lower_bound = -4.5 * np.ones((num_design_var, 1))
    upper_bound = 4.5 * np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
      max_iter=500,
      kkt_tol=1e-3,
      step_tol=1e-2,
      move_limit=5e-3,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)

      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state = mma.update_mma(
        mma_state, mma_params, objective, grad_obj, constr, grad_cons
      )

    np.testing.assert_allclose(objective, 0.0, atol=1e-2)
    np.testing.assert_allclose(mma_state.x[0, 0], 3.0, rtol=1e-1)
    np.testing.assert_allclose(mma_state.x[1, 0], 0.5, rtol=1e-1)

  def test_booth_function(self):
    """Test if MMA handles multivariable optimization correctly.

    Objective: min f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    bounds: -10 < x, y < 10

    expected result:
      x* = 1 , y* = 3
      f(x*, y*) = 0
    """
    num_design_var = 2
    num_cons = 1  # dummy constraint

    def dummy_constraint(x):
      del x
      return (np.zeros((num_cons, 1)), np.zeros((num_cons, num_design_var)))

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] + 2 * x[1, 0] - 7.0) ** 2 + (2 * x[0, 0] + x[1, 0] - 5.0) ** 2

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.zeros((num_design_var, 1))
    lower_bound = -10 * np.ones((num_design_var, 1))
    upper_bound = 10 * np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
      max_iter=500,
      kkt_tol=1e-3,
      step_tol=1e-2,
      move_limit=5e-3,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)

      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state = mma.update_mma(
        mma_state, mma_params, objective, grad_obj, constr, grad_cons
      )

    np.testing.assert_allclose(objective, 0.0, atol=1e-2)
    np.testing.assert_allclose(mma_state.x[0, 0], 1.0, rtol=1e-1)
    np.testing.assert_allclose(mma_state.x[1, 0], 3, rtol=1e-1)

  def test_optional_return_gives_multipliers(self):
    """Test that return_lagrange_multipliers=True returns tuple.

    Verifies if returns are the expected types and optionally return the multipliers.
    """

    def dummy_constraint(x):
      del x
      return (np.array([0.0]).reshape((1, 1)), np.array([0.0]).reshape((1, 1)))

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 50.0) ** 2 + 25.0

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[10.0]])
    num_design_var = 1
    num_cons = 1
    lower_bound = np.array([[1.0]])
    upper_bound = np.array([[100.0]])

    mma_params = mma.MMAParams(
      max_iter=50,
      kkt_tol=1e-3,
      step_tol=1e-3,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    objective, grad_obj = objective_fn(mma_state.x)
    constr, grad_cons = dummy_constraint(mma_state.x)

    result = mma.update_mma(
      mma_state,
      mma_params,
      objective,
      grad_obj,
      constr,
      grad_cons,
      return_lagrange_multipliers=True,
    )

    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)

    mma_state, lm = result

    self.assertIsInstance(mma_state, mma.MMAState)
    self.assertIsInstance(lm, mma.LagrangeMultipliers)

    # Check shapes
    self.assertEqual(lm.general_constraints.shape, (num_cons, 1))
    self.assertEqual(lm.lower_bounds.shape, (num_design_var, 1))
    self.assertEqual(lm.upper_bounds.shape, (num_design_var, 1))

  def test_multipliers_nonnegative(self):
    """Test KKT condition: all multipliers must be non-negative.

    This is a fundamental requirement for optimality.
    """

    def dummy_constraint(x):
      del x
      return (np.array([0.0]).reshape((1, 1)), np.array([0.0]).reshape((1, 1)))

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 50.0) ** 2

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[10.0]])
    num_design_var = 1
    num_cons = 1
    lower_bound = np.array([[1.0]])
    upper_bound = np.array([[100.0]])

    mma_params = mma.MMAParams(
      max_iter=100,
      kkt_tol=1e-6,
      step_tol=1e-6,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    # Check at every iteration
    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state, lm = mma.update_mma(
        mma_state,
        mma_params,
        objective,
        grad_obj,
        constr,
        grad_cons,
        return_lagrange_multipliers=True,
      )

      # All multipliers must be non-negative
      self.assertTrue(np.all(lm.general_constraints >= -1e-10))
      self.assertTrue(np.all(lm.lower_bounds >= -1e-10))
      self.assertTrue(np.all(lm.upper_bounds >= -1e-10))
      self.assertTrue(np.all(lm.slack_nonnegativity >= -1e-10))
      self.assertTrue(lm.regularization_nonneg >= -1e-10)

  def test_active_lower_bound_positive_multiplier(self):
    """Test that an active lower bound has a positive multiplier.

    Problem: min (x - 0.5)^2, subject to 1 ≤ x ≤ 100

    Expected:
    - Optimal x* = 1 (at lower bound)
    - Lower bound multiplier > 0
    - Upper bound multiplier ≈ 0
    """

    def dummy_constraint(x):
      del x
      return (np.array([0.0]).reshape((1, 1)), np.array([0.0]).reshape((1, 1)))

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 0.5) ** 2

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[10.0]])
    num_design_var = 1
    num_cons = 1
    lower_bound = np.array([[1.0]])
    upper_bound = np.array([[100.0]])

    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-6,
      step_tol=1e-6,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    # Run to convergence
    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = dummy_constraint(mma_state.x)
      mma_state, lm = mma.update_mma(
        mma_state,
        mma_params,
        objective,
        grad_obj,
        constr,
        grad_cons,
        return_lagrange_multipliers=True,
      )

    # # Check solution
    x_opt = mma_state.x[0, 0]
    self.assertAlmostEqual(x_opt, 1.0, places=2, msg="Should be at lower bound")

    # # Check multipliers
    self.assertGreater(
      lm.lower_bounds[0, 0],
      0.01,
      msg=f"Active lower bound should have positive multiplier, "
      f"got {lm.lower_bounds[0, 0]}",
    )
    self.assertAlmostEqual(
      lm.upper_bounds[0, 0],
      0.0,
      places=1,
      msg="Inactive upper bound should have ~0 multiplier",
    )

  def test_inactive_bounds_zero_multipliers(self):
    """Test complementary slackness: inactive bounds have zero multipliers.

    Problem: min (x - 50)^2, subject to 1 ≤ x ≤ 100

    Expected:
    - Optimal x* = 50 (interior solution)
    - Both bound multipliers ≈ 0
    """

    def dummy_constraint(x):
      del x
      return (np.array([0.0]).reshape((1, 1)), np.array([0.0]).reshape((1, 1)))

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 50.0) ** 2

      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[10.0]])
    num_design_var = 1
    num_cons = 1
    lower_bound = np.array([[1.0]])
    upper_bound = np.array([[100.0]])

    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-6,
      step_tol=1e-6,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = dummy_constraint(mma_state.x)
      mma_state, lm = mma.update_mma(
        mma_state,
        mma_params,
        objective,
        grad_obj,
        constr,
        grad_cons,
        return_lagrange_multipliers=True,
      )

    x_opt = mma_state.x[0, 0]
    self.assertAlmostEqual(x_opt, 50.0, places=2)

    # Interior solution: both multipliers should be near zero
    self.assertAlmostEqual(
      lm.lower_bounds[0, 0],
      0.0,
      places=2,
      msg="Inactive lower bound should have ~0 multiplier",
    )
    self.assertAlmostEqual(
      lm.upper_bounds[0, 0],
      0.0,
      places=2,
      msg="Inactive upper bound should have ~0 multiplier",
    )

  def test_stationarity_with_active_general_constraint(self):
    """Test stationarity when a general constraint is active.

    Theory: At optimum with active constraint:
      ∇f(x*) + λ*∇g(x*) + bound terms = 0

    Problem:
      min x^2 + y^2
      s.t. x + y ≥ 1.5  (equivalently: -x - y + 1.5 ≤ 0)
           -2 ≤ x, y ≤ 2

    Expected:
      x* = y* = 0.75 (on constraint boundary)
      Constraint active: -0.75 - 0.75 + 1.5 = 0
      λ > 0 (positive multiplier)
      Stationarity: ∇f + λ*∇g ≈ 0
    """
    num_design_var = 2
    num_cons = 1

    def constraint_fn(x):
      def confn(x):
        # Constraint: x + y ≥ 1.5 → -x - y + 1.5 ≤ 0
        return -x[0, 0] - x[1, 0] + 1.5

      c, dc = value_and_grad(confn)(x)
      return (
        np.array(c).reshape((num_cons, 1)),
        np.array(dc).reshape((num_cons, num_design_var)),
      )

    def objective_fn(xy):
      def objfn(xy):
        return xy[0, 0] ** 2 + xy[1, 0] ** 2

      obj, grad_obj = value_and_grad(objfn)(xy)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[0.5], [0.5]])
    lower_bound = -2.0 * np.ones((num_design_var, 1))
    upper_bound = 2.0 * np.ones((num_design_var, 1))

    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-5,
      step_tol=1e-5,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = constraint_fn(mma_state.x)
      mma_state, lm = mma.update_mma(
        mma_state,
        mma_params,
        objective,
        grad_obj,
        constr,
        grad_cons,
        return_lagrange_multipliers=True,
      )

    x_opt = mma_state.x

    # Recompute gradients at optimum
    _, grad_f = objective_fn(x_opt)
    c_val, grad_c = constraint_fn(x_opt)

    # Check constraint is active
    self.assertLess(
      abs(c_val[0, 0]), 0.1, msg="Constraint should be approximately active"
    )

    # Check multiplier is positive
    self.assertGreater(
      lm.general_constraints[0, 0],
      0.1,
      msg="Active constraint should have positive multiplier",
    )

    # Compute stationarity: ∇f + λ*∇g + bound terms
    grad_lagrangian = grad_f.copy()
    grad_lagrangian += lm.general_constraints[0, 0] * grad_c.T
    grad_lagrangian -= lm.lower_bounds
    grad_lagrangian += lm.upper_bounds

    stationarity_norm = np.linalg.norm(grad_lagrangian)

    # Verify stationarity
    self.assertLess(
      stationarity_norm,
      1e-3,
      msg=f"Stationarity violated: ||∇L|| = {stationarity_norm}",
    )

  def test_stationarity_at_lower_bound(self):
    """Test stationarity when solution is at lower bound.

    Theory: When x_j is at lower bound:
      (∇f)_j - ξ_j ≈ 0, with ξ_j > 0

    Problem:
      min (x - 0.5)^2 + (y - 0.5)^2
      s.t. 1 ≤ x, y ≤ 10

    Expected:
      x* = y* = 1 (at lower bounds)
      ∇f(x*) = [1.0, 1.0]
      ξ ≈ ∇f for stationarity
    """
    num_design_var = 2
    num_cons = 1

    def dummy_constraint(x):
      del x
      return (
        np.array([0.0]).reshape((num_cons, 1)),
        np.array([0.0, 0.0]).reshape((num_cons, num_design_var)),
      )

    def objective_fn(xy):
      def objfn(xy):
        return (xy[0, 0] - 0.5) ** 2 + (xy[1, 0] - 0.5) ** 2

      obj, grad_obj = value_and_grad(objfn)(xy)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[5.0], [5.0]])
    lower_bound = np.array([[1.0], [1.0]])
    upper_bound = np.array([[10.0], [10.0]])

    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-6,
      step_tol=1e-6,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = dummy_constraint(mma_state.x)
      mma_state, lm = mma.update_mma(
        mma_state,
        mma_params,
        objective,
        grad_obj,
        constr,
        grad_cons,
        return_lagrange_multipliers=True,
      )

    x_opt = mma_state.x

    # Verify at lower bounds
    self.assertAlmostEqual(x_opt[0, 0], 1.0, places=2)
    self.assertAlmostEqual(x_opt[1, 0], 1.0, places=2)

    # Recompute gradient at optimum
    _, grad_f = objective_fn(x_opt)

    # For bound constraints: ∇f - ξ ≈ 0 (when at lower bound)
    residual = grad_f - lm.lower_bounds
    residual_norm = np.linalg.norm(residual)

    # Both components should have positive multipliers
    self.assertGreater(lm.lower_bounds[0, 0], 0.01)
    self.assertGreater(lm.lower_bounds[1, 0], 0.01)

    # Stationarity: ∇f - ξ should be small
    self.assertLess(
      residual_norm,
      0.5,
      msg=f"Stationarity at bounds violated: ||∇f - ξ|| = {residual_norm}",
    )

  def test_stationarity_at_upper_bound(self):
    """Test stationarity when solution is at upper bound.

    Theory: When x_j is at upper bound:
      (∇f)_j + η_j ≈ 0, with η_j > 0

    Problem:
      min (x - 10.5)^2 + (y - 10.5)^2
      s.t. 1 ≤ x, y ≤ 10

    Expected:
      x* = y* = 10 (at upper bounds)
      ∇f(x*) = [-1.0, -1.0]
      η ≈ -∇f for stationarity
    """
    num_design_var = 2
    num_cons = 1

    def dummy_constraint(x):
      del x
      return (
        np.array([0.0]).reshape((num_cons, 1)),
        np.array([0.0, 0.0]).reshape((num_cons, num_design_var)),
      )

    def objective_fn(xy):
      def objfn(xy):
        return (xy[0, 0] - 10.5) ** 2 + (xy[1, 0] - 10.5) ** 2

      obj, grad_obj = value_and_grad(objfn)(xy)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[5.0], [5.0]])
    lower_bound = np.array([[1.0], [1.0]])
    upper_bound = np.array([[10.0], [10.0]])

    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-6,
      step_tol=1e-6,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = dummy_constraint(mma_state.x)
      mma_state, lm = mma.update_mma(
        mma_state,
        mma_params,
        objective,
        grad_obj,
        constr,
        grad_cons,
        return_lagrange_multipliers=True,
      )

    x_opt = mma_state.x

    # Verify at upper bounds
    self.assertAlmostEqual(x_opt[0, 0], 10.0, places=2)
    self.assertAlmostEqual(x_opt[1, 0], 10.0, places=2)

    # Recompute gradient at optimum
    _, grad_f = objective_fn(x_opt)

    # For upper bound: ∇f + η ≈ 0
    residual = grad_f + lm.upper_bounds
    residual_norm = np.linalg.norm(residual)

    # Both components should have positive multipliers
    self.assertGreater(lm.upper_bounds[0, 0], 0.01)
    self.assertGreater(lm.upper_bounds[1, 0], 0.01)

    # Stationarity: ∇f + η should be small
    self.assertLess(
      residual_norm,
      0.5,
      msg=f"Stationarity at upper bounds violated: ||∇f + η|| = {residual_norm}",
    )

  def test_stationarity_mixed_bounds(self):
    """Test stationarity with variables at different bounds.

    Problem:
      min (x - 0.5)^2 + (y - 10.5)^2 + (z - 5)^2
      s.t. 1 ≤ x, y, z ≤ 10

    Expected:
      x* = 1 (at lower), y* = 10 (at upper), z* = 5 (interior)
      ξ_x > 0, η_y > 0, ξ_z ≈ 0, η_z ≈ 0
    """
    num_design_var = 3
    num_cons = 1

    def dummy_constraint(x):
      del x
      return (
        np.array([0.0]).reshape((num_cons, 1)),
        np.array([0.0, 0.0, 0.0]).reshape((num_cons, num_design_var)),
      )

    def objective_fn(xyz):
      def objfn(xyz):
        return (xyz[0, 0] - 0.5) ** 2 + (xyz[1, 0] - 10.5) ** 2 + (xyz[2, 0] - 5.0) ** 2

      obj, grad_obj = value_and_grad(objfn)(xyz)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.array([[5.0], [5.0], [5.0]])
    lower_bound = np.array([[1.0], [1.0], [1.0]])
    upper_bound = np.array([[10.0], [10.0], [10.0]])

    mma_params = mma.MMAParams(
      max_iter=200,
      kkt_tol=1e-6,
      step_tol=1e-6,
      move_limit=1e-2,
      num_design_var=num_design_var,
      num_cons=num_cons,
      lower_bound=lower_bound,
      upper_bound=upper_bound,
    )
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      constr, grad_cons = dummy_constraint(mma_state.x)
      mma_state, lm = mma.update_mma(
        mma_state,
        mma_params,
        objective,
        grad_obj,
        constr,
        grad_cons,
        return_lagrange_multipliers=True,
      )

    x_opt = mma_state.x

    # Verify expected bounds
    self.assertAlmostEqual(x_opt[0, 0], 1.0, places=2, msg="x should be at lower")
    self.assertAlmostEqual(x_opt[1, 0], 10.0, places=2, msg="y should be at upper")
    self.assertAlmostEqual(x_opt[2, 0], 5.0, places=2, msg="z should be interior")

    # Check multipliers
    self.assertGreater(lm.lower_bounds[0, 0], 0.01, msg="x at lower → ξ_x > 0")
    self.assertGreater(lm.upper_bounds[1, 0], 0.01, msg="y at upper → η_y > 0")
    self.assertAlmostEqual(
      lm.lower_bounds[2, 0], 0.0, places=1, msg="z interior → ξ_z ≈ 0"
    )
    self.assertAlmostEqual(
      lm.upper_bounds[2, 0], 0.0, places=1, msg="z interior → η_z ≈ 0"
    )

  def test_multiplier_scales_with_objective(self):
    """Test that multipliers scale linearly with objective magnitude.

    Theory: For scaled problem min α*f(x) s.t. g(x) ≤ 0
      Solution x* is the same, but λ_scaled = α * λ_original

    Problem:
      min α * (x - 0.5)^2
      s.t. 1 ≤ x ≤ 10

    Test with α = 1, 10, 100

    Expected:
      x* = 1 for all α (same solution)
      λ_bound(α) = α * λ_bound(1)
    """
    num_design_var = 1
    num_cons = 1

    def dummy_constraint(x):
      del x
      return (
        np.array([0.0]).reshape((num_cons, 1)),
        np.array([0.0]).reshape((num_cons, num_design_var)),
      )

    def create_objective_fn(scale_factor):
      def objective_fn(x):
        def objfn(x):
          return scale_factor * (x[0, 0] - 0.5) ** 2

        obj, grad_obj = value_and_grad(objfn)(x)
        return obj.reshape((-1)), grad_obj.reshape((-1, 1))

      return objective_fn

    lower_bound = np.array([[1.0]])
    upper_bound = np.array([[10.0]])

    scales = [1.0, 10.0, 100.0]
    results = []

    for scale in scales:
      design_var = np.array([[5.0]])

      mma_params = mma.MMAParams(
        max_iter=200,
        kkt_tol=1e-6,
        step_tol=1e-6,
        move_limit=1e-2,
        num_design_var=num_design_var,
        num_cons=num_cons,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
      )
      mma_state = mma.init_mma(design_var, mma_params)

      objective_fn = create_objective_fn(scale)

      while not mma_state.is_converged:
        objective, grad_obj = objective_fn(mma_state.x)
        constr, grad_cons = dummy_constraint(mma_state.x)
        mma_state, lm = mma.update_mma(
          mma_state,
          mma_params,
          objective,
          grad_obj,
          constr,
          grad_cons,
          return_lagrange_multipliers=True,
        )

      results.append(
        {
          "scale": scale,
          "x_opt": mma_state.x[0, 0],
          "lambda_lower": lm.lower_bounds[0, 0],
        }
      )

    # All should converge to same x*
    for r in results:
      self.assertAlmostEqual(
        r["x_opt"], 1.0, places=1, msg="Solution should be same for all scales"
      )

    # Check scaling relationship
    base_lambda = results[0]["lambda_lower"]
    for i, r in enumerate(results[1:], start=1):
      expected_lambda = base_lambda * (r["scale"] / results[0]["scale"])
      actual_lambda = r["lambda_lower"]
      relative_error = abs(actual_lambda - expected_lambda) / max(expected_lambda, 1e-6)

      self.assertLess(
        relative_error,
        0.2,  # 20% tolerance
        msg=f"Multiplier should scale with objective. "
        f"Expected {expected_lambda:.4f}, got {actual_lambda:.4f}",
      )

  def test_multiplier_scales_inversely_with_constraint(self):
    """Test that constraint multipliers scale inversely with constraint scaling.

    Theory: Scaling constraint α*g doesn't change feasible region,
      but λ scales as 1/α to maintain stationarity balance.

    Problem:
      min x^2 + y^2
      s.t. α*(-x - y + 1.5) ≤ 0
           -2 ≤ x, y ≤ 2

    Test with α = 1, 2, 5

    Expected:
      x*, y* same for all α
      λ(α) * α ≈ constant
    """
    num_design_var = 2
    num_cons = 1

    def create_constraint_fn(scale_factor):
      def constraint_fn(x):
        def confn(x):
          return scale_factor * (-x[0, 0] - x[1, 0] + 1.5)

        c, dc = value_and_grad(confn)(x)
        return (
          np.array(c).reshape((num_cons, 1)),
          np.array(dc).reshape((num_cons, num_design_var)),
        )

      return constraint_fn

    def objective_fn(xy):
      def objfn(xy):
        return xy[0, 0] ** 2 + xy[1, 0] ** 2

      obj, grad_obj = value_and_grad(objfn)(xy)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    lower_bound = -2.0 * np.ones((num_design_var, 1))
    upper_bound = 2.0 * np.ones((num_design_var, 1))

    scales = [1.0, 2.0, 5.0]
    results = []

    for scale in scales:
      design_var = np.array([[0.5], [0.5]])

      mma_params = mma.MMAParams(
        max_iter=200,
        kkt_tol=1e-5,
        step_tol=1e-5,
        move_limit=1e-2,
        num_design_var=num_design_var,
        num_cons=num_cons,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
      )
      mma_state = mma.init_mma(design_var, mma_params)

      constraint_fn = create_constraint_fn(scale)

      while not mma_state.is_converged:
        objective, grad_obj = objective_fn(mma_state.x)
        constr, grad_cons = constraint_fn(mma_state.x)
        mma_state, lm = mma.update_mma(
          mma_state,
          mma_params,
          objective,
          grad_obj,
          constr,
          grad_cons,
          return_lagrange_multipliers=True,
        )

      results.append(
        {
          "scale": scale,
          "x_opt": mma_state.x[0, 0],
          "y_opt": mma_state.x[1, 0],
          "lambda_constraint": lm.general_constraints[0, 0],
        }
      )

    # Solutions should be approximately the same
    for i in range(len(results) - 1):
      self.assertAlmostEqual(
        results[i]["x_opt"],
        results[i + 1]["x_opt"],
        places=1,
        msg="x* should be same for all scalings",
      )
      self.assertAlmostEqual(
        results[i]["y_opt"],
        results[i + 1]["y_opt"],
        places=1,
        msg="y* should be same for all scalings",
      )

    # Check inverse scaling: λ * scale should be approximately constant
    products = [r["lambda_constraint"] * r["scale"] for r in results]
    mean_product = np.mean(products)
    for prod in products:
      relative_deviation = abs(prod - mean_product) / mean_product
      self.assertLess(
        relative_deviation,
        0.3,  # 30% tolerance
        msg="λ * scale should be approximately constant",
      )


if __name__ == "__main__":
  absltest.main()
