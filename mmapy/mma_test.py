"""Tests for mma."""

from absl.testing import absltest
from jax import value_and_grad
import jax.numpy as jnp
import numpy as np

import mma


class MmaTest(absltest.TestCase):

  def test_mma_state_to_array(self):
    mma_state = mma.MMAState(x=np.array([[1., 2., 3.]]).T,
                             x_old_1=np.array([[4., 5., 6.]]).T,
                             x_old_2=np.array([[7., 8., 9.]]).T,
                             low=np.array([[0., 0., 0.]]).T,
                             upp=np.array([[10., 10., 10.]]).T,
                             is_converged=True,
                             epoch=20,
                             kkt_norm=10.,
                             change_design_var=20.)
    mma_state_array = mma_state.to_array()
    expected_array = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 0., 0.,
                               10., 10., 10., 1, 20, 10., 20.])
    np.testing.assert_array_equal(mma_state_array, expected_array)

  def test_mma_state_from_array(self):
    mma_state_array = np.array([
        1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 0., 0., 10., 10., 10., 1, 20,
        10., 20.
    ])
    mma_state = mma.MMAState.from_array(mma_state_array, num_design_var=3)
    expected_mma_state = mma.MMAState(
        x=np.array([[1., 2., 3.]]).T,
        x_old_1=np.array([[4., 5., 6.]]).T,
        x_old_2=np.array([[7., 8., 9.]]).T,
        low=np.array([[0., 0., 0.]]).T,
        upp=np.array([[10., 10., 10.]]).T,
        is_converged=True,
        epoch=20,
        kkt_norm=10.,
        change_design_var=20.)
    np.testing.assert_array_equal(mma_state.x, expected_mma_state.x)
    np.testing.assert_array_equal(mma_state.x_old_1, expected_mma_state.x_old_1)
    np.testing.assert_array_equal(mma_state.x_old_2, expected_mma_state.x_old_2)
    np.testing.assert_array_equal(mma_state.low, expected_mma_state.low)
    np.testing.assert_array_equal(mma_state.upp, expected_mma_state.upp)
    np.testing.assert_equal(mma_state.is_converged,
                            expected_mma_state.is_converged)
    np.testing.assert_equal(mma_state.epoch, expected_mma_state.epoch)
    np.testing.assert_equal(mma_state.kkt_norm, expected_mma_state.kkt_norm)
    np.testing.assert_equal(mma_state.change_design_var,
                            expected_mma_state.change_design_var)

  def test_mma_state_from_array_raises_error_if_passed_wrong_num_desvar(self):
    mma_state_array = np.array([
        1., 2., 3., 4., 5., 6., 7., 8., 9., 0., 0., 0., 10., 10., 10., 1, 20,
        10., 20.
    ])
    with self.assertRaisesRegex(
        ValueError, '`state_array` shape is incompatible with'):
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
      return (np.array([0.]).reshape((num_cons, 1)),
              np.array([0.]).reshape((num_cons, num_design_var)))

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 50.)**2 + 25.
      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.random.uniform(0., 100., (1)).reshape((-1, 1))
    num_design_var = 1
    num_cons = 1
    lower_bound = np.zeros((num_design_var, 1))
    upper_bound = 100.*np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
        max_iter=200, kkt_tol=1e-3, step_tol=1e-3, move_limit=1e-2,
        num_design_var=num_design_var, num_cons=num_cons,
        lower_bound=lower_bound, upper_bound=upper_bound)
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)

      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state = mma.update_mma(mma_state, mma_params, objective, grad_obj,
                                 constr, grad_cons)

    self.assertAlmostEqual(mma_state.x[0, 0], 50., places=2)

  def test_two_variable_optimization_with_no_constraints(self):
    """Test if mma handles two variable optimization correctly.

            Objective: min (x-30)^2 + (y-1)^2 + 1000
            constraints -
            bounds: 0 <= x, y <= 1000
    """
    # we need at least one constraint and hence pass a dummy one
    def dummy_constraint(x):
      del x
      return (np.array([0.]).reshape((num_cons, 1)),
              np.array([0., 0.]).reshape((num_cons, num_design_var)))

    def objective_fn(x):
      def objfn(x):
        return (x[0, 0] - 30.)**2 + (x[1, 0] - 1.)**2 + 1000.
      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.random.uniform(0., 100., (2)).reshape((-1, 1))
    num_design_var = 2
    num_cons = 1
    lower_bound = np.zeros((num_design_var, 1))
    upper_bound = 1000.*np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
        max_iter=200, kkt_tol=1e-3, step_tol=1e-3, move_limit=1e-2,
        num_design_var=num_design_var, num_cons=num_cons,
        lower_bound=lower_bound, upper_bound=upper_bound)
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)

      constr, grad_cons = dummy_constraint(mma_state.x)

      mma_state = mma.update_mma(mma_state, mma_params, objective, grad_obj,
                                 constr, grad_cons)

    self.assertAlmostEqual(np.abs(mma_state.x[0, 0] - 30.)/30., 0., places=1)
    self.assertAlmostEqual(np.abs(mma_state.x[1, 0]- 1.), 0., places=1)

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
        return -jnp.sin(4*np.pi*x[0, 0]) + 2*jnp.sin(2*np.pi*x[1, 0])**2 - 1.5
      c, dc = value_and_grad(confn)(x)
      return (np.array(c).reshape((num_cons, 1)),
              np.array(dc).reshape((num_cons, num_design_var)))

    def objective_fn(x):
      def objfn(xy):
        x = xy[0, 0]
        y = xy[1, 0]
        return 4*x**2 - 2.1*x**4 + 0.333*x**6 + x*y - 4*y**2 + 4*y**4
      obj, grad_obj = value_and_grad(objfn)(x)
      return obj.reshape((-1)), grad_obj.reshape((-1, 1))

    design_var = np.zeros((2)).reshape((-1, 1))
    num_design_var = 2
    num_cons = 1
    lower_bound = -np.ones((num_design_var, 1))
    upper_bound = np.ones((num_design_var, 1))
    mma_params = mma.MMAParams(
        max_iter=200, kkt_tol=1e-6, step_tol=1e-6, move_limit=1e-2,
        num_design_var=num_design_var, num_cons=num_cons,
        lower_bound=lower_bound, upper_bound=upper_bound)
    mma_state = mma.init_mma(design_var, mma_params)

    while not mma_state.is_converged:
      objective, grad_obj = objective_fn(mma_state.x)
      print(f'{mma_state.epoch} obj, {objective}')
      constr, grad_cons = dummy_constraint(mma_state.x)
      mma_state = mma.update_mma(mma_state, mma_params, objective, grad_obj,
                                 constr, grad_cons)

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

      print(f'epoch {mma_state.epoch} , obj {objective[0]:.2E}')

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
        return (x[0, 0] + 2 * x[1, 0] - 7.0) ** 2 + (
            2 * x[0, 0] + x[1, 0] - 5.0
        ) ** 2

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

      print(f'epoch {mma_state.epoch} , obj {objective[0]:.2E}')

    np.testing.assert_allclose(objective, 0.0, atol=1e-2)
    np.testing.assert_allclose(mma_state.x[0, 0], 1.0, rtol=1e-1)
    np.testing.assert_allclose(mma_state.x[1, 0], 3, rtol=1e-1)


if __name__ == '__main__':
  absltest.main()
