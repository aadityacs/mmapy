"""MMA optimization code.

This is a python implementation of the method of moving asymptopes.

If you use this code, kindly cite the original MMA paper:

Svanberg, K., 1987. The method of moving asymptotes—a new method for
structural optimization. International journal for numerical methods in
engineering, 24(2), pp.359-373.
"""


import dataclasses
from absl import logging

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse

_SUBSOLV_EPSI_FACTOR = 0.9
_SUBSOLV_MAX_INNER_ITER = 200
_SUBSOLV_RESIDUE_NORM_FACTOR = 2
_SUBSOLV_MAX_OUTER_ITER = 50
_SUBSOLV_EPSI_FACTOR = 0.1

_MMASUB_EPSIMIN = 0.0000001
_MMASUB_RAA0 = 0.00001
_MMASUB_ALBEFA = 0.1
_MMASUB_ASY_INIT = 0.5
_MMASUB_ASY_INCR = 1.2
_MMASUB_ASY_DECR = 0.7

_MMA_INIT_DEFAULT_A0 = 1.
_MMA_INIT_DEFAULT_A = 0.
_MMA_INIT_DEFAULT_C = 1000.
_MMA_INIT_DEFAULT_D = 1.


@dataclasses.dataclass
class MMAState:
  """Current state of the MMA optimization.

  Attributes:
    x: Current value of the design variables.
    x_old_1: Value of the design variable at the previous (k-1) iteration.
    x_old_2: Value of the design variable at two iterations before (k-2).
    low: Current lower search bound on each of the design variable
    upp: Current upper search bound on each of the design variable
    is_converged: Boolean indicating if the optimization has converged.
    epoch: current iteration number.
    kkt_norm: Value indication how close optimization is close to convergence.
      Convergence is achieved when kkt_norm <= MMAParams.kkt_tol
    change_design_var: A L2 norm on how much the design variables have changed
      within current and previous iteration. Convergence is achieved when
      change_design_var <= MMAParams.step_tol
  """

  x: np.ndarray
  x_old_1: np.ndarray
  x_old_2: np.ndarray
  low: np.ndarray
  upp: np.ndarray
  is_converged: bool
  epoch: int
  kkt_norm: float
  change_design_var: float

  @classmethod
  def new(cls, num_design_var: int) -> 'MMAState':
    """Returns an `MMAState` with all-zeros fields, for a new optimization."""
    return MMAState(
        x=np.zeros((num_design_var, 1)),
        x_old_1=np.zeros((num_design_var, 1)),
        x_old_2=np.zeros((num_design_var, 1)),
        low=np.zeros((num_design_var, 1)),
        upp=np.ones((num_design_var, 1)),
        is_converged=False,
        epoch=0,
        kkt_norm=1.,
        change_design_var=1.,
    )

  @classmethod
  def from_array(
      cls,
      state_array: np.ndarray,
      num_design_var: int,
  ) -> 'MMAState':
    """Reconstructs an `MMAState` from an array."""
    empty = MMAState.new(num_design_var)
    if empty.to_array().shape != state_array.shape:
      raise ValueError(
          f'`state_array` shape is incompatible with `num_design_var`, got a '
          f'shape of {state_array.shape} but expected {empty.to_array().shape}'
          f'when `num_design_var` is {num_design_var}.')
    n = num_design_var
    return MMAState(x=state_array[0:n].reshape((-1, 1)),
                    x_old_1=state_array[n:2*n].reshape((-1, 1)),
                    x_old_2=state_array[2*n:3*n].reshape((-1, 1)),
                    low=state_array[3*n:4*n].reshape((-1, 1)),
                    upp=state_array[4*n:5*n].reshape((-1, 1)),
                    is_converged=bool(state_array[5*n]),
                    epoch=int(state_array[5*n+1]),
                    kkt_norm=state_array[5*n+2],
                    change_design_var=state_array[5*n+3],)

  def to_array(self) -> np.ndarray:
    """Converts the `MMAState` into a rank-1 array."""
    return np.concatenate(
        [np.array(field).flatten() for field in dataclasses.astuple(self)])


@dataclasses.dataclass
class MMAParams:
  """Parameters that define MMA optimizer.

  Attributes:
    max_iter: maximum number of optimization iterations
    kkt_tol: tolerance for KKT check. Convergence is achieved
      when kkt_norm <= kkt_tol, where kkt_norm is computed during optimization.
    step_tol: tolerance check between successive optimization steps.
      Convergence is achieved when change_design_var <= step_tol, where
      change_design_var is computed during optimization.
    move_limit: the learning rate parameter for MMA. The parameter
      defines the extent of search space for each optimization step.
    num_design_var: number of design variables
    num_cons: number of constraints
    lower_bound: Array of size (num_design_var, 1) which have the lower bound
      (box constraint) for the design variables.
    upper_bound: Array of size (num_design_var, 1) which have the upper bound
      (box constraint) for the design variables.
    move_limit_step: Each design variable in MMA has a search space bound on it
    a0: MMA constant
    a: MMA constant
    c: MMA constant
    d: MMA constant
  """
  max_iter: float
  kkt_tol: float
  step_tol: float
  move_limit: float
  num_design_var: int
  num_cons: int
  lower_bound: np.ndarray
  upper_bound: np.ndarray

  @property
  def move_limit_step(self)-> np.ndarray:
    return self.move_limit * abs(self.upper_bound - self.lower_bound)

  @property
  def a0(self)-> float:
    return _MMA_INIT_DEFAULT_A0

  @property
  def a(self)->np.ndarray:
    return _MMA_INIT_DEFAULT_A*np.ones((self.num_cons, 1))

  @property
  def c(self)-> np.ndarray:
    return _MMA_INIT_DEFAULT_C*np.ones((self.num_cons, 1))

  @property
  def d(self)-> np.ndarray:
    return _MMA_INIT_DEFAULT_D*np.ones((self.num_cons, 1))


def init_mma(init_design_var: np.ndarray,
             mma_params: MMAParams)-> MMAState:
  """Initialize the MMA optimizer.

  Args:
    init_design_var: Array of size (num_design_var, 1) which are the initial
      design variables for the optimizer.
    mma_params: A dataclass of MMAParams that contain all the settings of MMA.

  Returns:
    A dataclass of MMAState that contain all the values that are required
    and change during optimization.
  """

  return MMAState(
      x=init_design_var.copy(),
      x_old_1=init_design_var.copy(),
      x_old_2=init_design_var.copy(),
      low=mma_params.lower_bound.copy(),
      upp=mma_params.upper_bound.copy(),
      is_converged=False,
      epoch=0,
      kkt_norm=1000.,
      change_design_var=1000.,
      )


def update_mma(mma_state: MMAState, mma_params: MMAParams, obj: np.ndarray,
               grad_obj: np.ndarray, cons: np.ndarray,
               grad_cons: np.ndarray)-> MMAState:
  """Call single step of MMA update.

  Args:
    mma_state: Dataclass of type MMAState that contains the current state of
      the optimization.
    mma_params: Dataclass of type MMAParams that contain the parameters
      associated with the MMA optimizer.
    obj: Array of shape (1,) that contain the current objective value.
    grad_obj: Array of shape (num_design_var, 1) that contain the
      gradient of the objective w.r.t to the design variables.
    cons: Array of shape (num_cons, 1) that contain the values of the
      constraints.
    grad_cons: Array of shape (num_cons, num_design_var) that contain the
      gradient of each of the constraints w.r.t each of the design variables

  Returns:
    A MMAState dataclass that contains the updated state of the
    optimization.
  """
  mma_state.epoch += 1
  epoch = mma_state.epoch
  # Impose move limits by modifying lower and upper bounds passed to MMA
  mlb = np.maximum(mma_params.lower_bound,
                   mma_state.x - mma_params.move_limit_step)
  mub = np.minimum(mma_params.upper_bound,
                   mma_state.x + mma_params.move_limit_step)

  # Solve MMA subproblem for current design x
  xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, mma_state.low, mma_state.upp = _mmasub(
      mma_params.num_cons, mma_params.num_design_var, epoch, mma_state.x, mlb,
      mub, mma_state.x_old_1, mma_state.x_old_2, obj, grad_obj, cons, grad_cons,
      mma_state.low, mma_state.upp, mma_params.a0, mma_params.a, mma_params.c,
      mma_params.d, 0.5)

  # Updated design vectors of previous and current iterations
  mma_state.x_old_2, mma_state.x_old_1, mma_state.x = (mma_state.x_old_1,
                                                       mma_state.x, xmma)

  # Compute change in design variables
  # Check only after first iteration
  if epoch > 1:
    mma_state.change_design_var = np.linalg.norm(mma_state.x -
                                                 mma_state.x_old_1)
    if mma_state.change_design_var < mma_params.step_tol:
      logging.info('Design step convergence tolerance satisfied')
      mma_state.is_converged = True

  if epoch == mma_params.max_iter:
    logging.info('Reached maximum number of iterations')
    mma_state.is_converged = True

  # Compute norm of KKT residual vector
  _, mma_state.kktnorm, _ = _kktcheck(
      mma_params.num_cons, mma_params.num_design_var,
      xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, mma_params.lower_bound,
      mma_params.upper_bound, grad_obj, cons, grad_cons, mma_params.a0,
      mma_params.a, mma_params.c, mma_params.d)

  if mma_state.kktnorm < mma_params.kkt_tol:
    logging.info('KKT tolerance satisfied')
    mma_state.is_converged = True

  return mma_state


def _mmasub(m: int, n: int, epoch: int, xval: np.ndarray, xmin: np.ndarray,
            xmax: np.ndarray, xold1: np.ndarray, xold2: np.ndarray,
            f0val: np.ndarray, df0dx: np.ndarray, fval: np.ndarray,
            dfdx: np.ndarray, low: np.ndarray, upp: np.ndarray, a0: float,
            a: np.ndarray, c: np.ndarray, d: np.ndarray, move: float):
  """Solve the MMA sub problem.

  This function mmasub performs one MMA-iteration, aimed at solving the
    nonlinear programming problem:

    Minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                xmin_j <= x_j <= xmax_j,    j = 1,...,n
                z >= 0,   y_i >= 0,         i = 1,...,m
  Args:
    m: The number of general constraints.
    n: The number of variables x_j.
    epoch: Current iteration number ( =1 the first time mmasub is called).
    xval: Column vector with the current values of the variables x_j.
    xmin: Column vector with the lower bounds for the variables x_j.
    xmax: Column vector with the upper bounds for the variables x_j.
    xold1: xval, one iteration ago (provided that iter>1).
    xold2: xval, two iterations ago (provided that iter>2).
    f0val: The value of the objective function f_0 at xval.
    df0dx: Column vector with the derivatives of the objective function f_0 with
      respect to the variables x_j, calculated at xval.
    fval: Column vector with the values of the constraint functions f_i,
      calculated at xval.
    dfdx: (m x n)-matrix with the derivatives of the constraint functions f_i
      with respect to the variables x_j, calculated at xval. dfdx(i,j) = the
      derivative of f_i with respect to x_j.
    low: Column vector with the lower asymptotes from the previous iteration
      (provided that iter>1).
    upp: Column vector with the upper asymptotes from the previous iteration
      (provided that iter>1).
    a0: The constants a_0 in the term a_0*z.
    a: Column vector with the constants a_i in the terms a_i*z.
    c: Column vector with the constants c_i in the terms c_i*y_i.
    d: Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    move: amount to move by during optimization.

  Returns:
    xmma: Column vector with the optimal values of the variables x_j
            in the current MMA subproblem.
    ymma: Column vector with the optimal values of the variables y_i
            in the current MMA subproblem.
    zmma: Scalar with the optimal value of the variable z
            in the current MMA subproblem.
    lam: Lagrange multipliers for the m general MMA constraints.
    xsi: Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
    eta: Lagrange multipliers for the n constraints x_j - beta_j <= 0.
    mu: Lagrange multipliers for the m constraints -y_i <= 0.
    zet: Lagrange multiplier for the single constraint -z <= 0.
    s: Slack variables for the m general MMA constraints.
    low: Column vector with the lower asymptotes, calculated and used
            in the current MMA subproblem.
    upp: Column vector with the upper asymptotes, calculated and used
            in the current MMA subproblem.
  """

  del f0val
  epsimin = _MMASUB_EPSIMIN
  raa0 = _MMASUB_RAA0
  albefa = _MMASUB_ALBEFA
  asyinit = _MMASUB_ASY_INIT
  asyincr = _MMASUB_ASY_INCR
  asydecr = _MMASUB_ASY_DECR
  eeen = np.ones((n, 1))
  eeem = np.ones((m, 1))
  zeron = np.zeros((n, 1))
  # Calculation of the asymptotes low and upp
  if epoch <= 2:
    low = xval - asyinit * (xmax - xmin)
    upp = xval + asyinit * (xmax - xmin)
  else:
    zzz = (xval - xold1) * (xold1 - xold2)
    factor = eeen.copy()
    factor[np.where(zzz > 0)] = asyincr
    factor[np.where(zzz < 0)] = asydecr
    low = xval - factor * (xold1 - low)
    upp = xval + factor * (upp - xold1)
    lowmin = xval - 10 * (xmax - xmin)
    lowmax = xval - 0.01 * (xmax - xmin)
    uppmin = xval + 0.01 * (xmax - xmin)
    uppmax = xval + 10 * (xmax - xmin)
    low = np.maximum(low, lowmin)
    low = np.minimum(low, lowmax)
    upp = np.minimum(upp, uppmax)
    upp = np.maximum(upp, uppmin)
  # Calculation of the bounds alfa and beta
  zzz1 = low + albefa * (xval - low)
  zzz2 = xval - move * (xmax - xmin)
  zzz = np.maximum(zzz1, zzz2)
  alfa = np.maximum(zzz, xmin)
  zzz1 = upp - albefa * (upp - xval)
  zzz2 = xval + move * (xmax - xmin)
  zzz = np.minimum(zzz1, zzz2)
  beta = np.minimum(zzz, xmax)
  # Calculations of p0, q0, P, Q and b
  xmami = xmax - xmin
  xmamieps = 0.00001 * eeen
  xmami = np.maximum(xmami, xmamieps)
  xmamiinv = eeen / xmami
  ux1 = upp - xval
  ux2 = ux1 * ux1
  xl1 = xval - low
  xl2 = xl1 * xl1
  uxinv = eeen / ux1
  xlinv = eeen / xl1
  p0 = zeron.copy()
  q0 = zeron.copy()
  p0 = np.maximum(df0dx, 0)
  q0 = np.maximum(-df0dx, 0)
  pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
  p0 = p0 + pq0
  q0 = q0 + pq0
  p0 = p0 * ux2
  q0 = q0 * xl2
  p_value = np.zeros((m, n))
  q_value = np.zeros((m, n))
  p_value = np.maximum(dfdx, 0)
  q_value = np.maximum(-dfdx, 0)
  pq_value = 0.001 * (p_value + q_value) + raa0 * np.dot(eeem, xmamiinv.T)
  p_value = p_value + pq_value
  q_value = q_value + pq_value
  p_value = (scipy.sparse.diags(ux2.flatten(), 0).dot(p_value.T)).T
  q_value = (scipy.sparse.diags(xl2.flatten(), 0).dot(q_value.T)).T
  b = (np.dot(p_value, uxinv) + np.dot(q_value, xlinv) - fval)

  # Solving the subproblem by a primal-dual Newton method
  xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = _subsolv(
      m, n, epsimin, low, upp, alfa, beta, p0, q0, p_value, q_value, a0, a, b,
      c, d)
  # Return values
  return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def _subsolv(m: int, n: int, epsimin: float, low: np.ndarray, upp: np.ndarray,
             alfa: np.ndarray, beta: np.ndarray, p0: float, q0: float,
             p_value: np.ndarray, q_value: np.ndarray, a0: float,
             a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
  """Solve the MMA or GCMMA sub problem.

  This function subsolv solves the MMA subproblem:

    minimize SUM[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + SUM[ci*yi +
    0.5*di*(yi)^2],

    subject to SUM[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,
        alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
  Args:
    m: Number of constraints.
    n: Number of design variables.
    epsimin: MMA convergence parameter.
    low: Current lower bounds on the design variables.
    upp: Current upper bounds on the design variables.
    alfa: MMA internal paramter.
    beta: MMA internal paramter.
    p0: MMA internal paramter.
    q0: MMA internal paramter.
    p_value: MMA internal paramter.
    q_value: MMA internal paramter.
    a0: MMA internal paramter.
    a: MMA internal paramter.
    b: MMA internal paramter.
    c: MMA internal paramter.
    d: MMA internal paramter.
  Returns:
    Solution of the sub-problem.
  """

  een = np.ones((n, 1))
  eem = np.ones((m, 1))
  epsi = 1
  epsvecn = epsi * een
  epsvecm = epsi * eem

  x = 0.5 * (alfa + beta)
  y = eem.copy()
  z = np.array([[1.0]])
  lam = eem.copy()
  xsi = een / (x - alfa)
  xsi = np.maximum(xsi, een)
  eta = een / (beta - x)
  eta = np.maximum(eta, een)
  mu = np.maximum(eem, 0.5 * c)
  zet = np.array([[1.0]])
  s = eem.copy()

  itera = 0
  # Start while epsi>epsimin
  while epsi > epsimin:
    epsvecn = epsi*een
    epsvecm = epsi*eem

    ux1 = upp - x
    xl1 = x - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    uxinv1 = een / ux1
    xlinv1 = een / xl1

    plam = p0 + np.dot(p_value.T, lam)
    qlam = q0 + np.dot(q_value.T, lam)

    gvec = np.dot(p_value, uxinv1) + np.dot(q_value, xlinv1)
    dpsidx = plam / ux2 - qlam / xl2

    rex = dpsidx - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - np.dot(a.T, lam)

    relam = gvec - a * z - y + s - b
    rexsi = xsi * (x - alfa) - epsvecn
    reeta = eta * (beta - x) - epsvecn
    remu = mu * y - epsvecm
    rezet = zet * z - epsi
    res = lam * s - epsvecm
    residu1 = np.concatenate((rex, rey, rez), axis=0)
    residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
    residu = np.concatenate((residu1, residu2), axis=0)
    residunorm = np.sqrt((np.dot(residu.T, residu)).item())
    residumax = np.max(np.abs(residu))
    ittt = 0
    # Start while (residumax>0.9*epsi) and (ittt<200)

    while ((residumax > _SUBSOLV_EPSI_FACTOR * epsi) and
           (ittt < _SUBSOLV_MAX_INNER_ITER)):
      ittt = ittt + 1
      itera = itera + 1

      ux1 = upp - x
      xl1 = x - low
      ux2 = ux1 * ux1
      xl2 = xl1 * xl1
      ux3 = ux1 * ux2
      xl3 = xl1 * xl2

      uxinv1 = een / ux1
      xlinv1 = een / xl1
      uxinv2 = een / ux2
      xlinv2 = een / xl2

      plam = p0 + np.dot(p_value.T, lam)
      qlam = q0 + np.dot(q_value.T, lam)

      gvec = np.dot(p_value, uxinv1) + np.dot(q_value, xlinv1)
      gg_value = (scipy.sparse.diags(uxinv2.flatten(), 0).dot(p_value.T)).T - (
          scipy.sparse.diags(xlinv2.flatten(), 0).dot(q_value.T)).T
      dpsidx = plam / ux2 - qlam / xl2
      delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
      dely = c + d * y - lam - epsvecm / y
      delz = a0 - np.dot(a.T, lam) - epsi / z
      dellam = gvec - a * z - y - b + epsvecm / lam

      diagx = plam / ux3 + qlam / xl3
      diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
      diagxinv = een / diagx
      diagy = d + mu / y
      diagyinv = eem / diagy
      diaglam = s / lam
      diaglamyi = diaglam + diagyinv
      # Start if m<n
      if m < n:
        blam = dellam + dely / diagy - np.dot(gg_value, (delx / diagx))
        bb = np.concatenate((blam, delz), axis=0)
        alam_value = np.asarray(
            scipy.sparse.diags(diaglamyi.flatten(), 0) +
            (scipy.sparse.diags(diagxinv.flatten(), 0).dot(gg_value.T).T
            ).dot(gg_value.T))
        aar1_value = np.concatenate((alam_value, a), axis=1)
        aar2_value = np.concatenate((a, -zet / z), axis=0).T
        aa_value = np.concatenate((aar1_value, aar2_value), axis=0)
        solut = scipy.linalg.solve(aa_value, bb)
        dlam = solut[0:m]
        dz = solut[m:m + 1]
        dx = -delx / diagx - np.dot(gg_value.T, dlam) / diagx
      else:
        diaglamyiinv = eem / diaglamyi
        dellamyi = dellam + dely / diagy
        axx_value = np.asarray(
            scipy.sparse.diags(diagx.flatten(), 0) +
            (scipy.sparse.diags(diaglamyiinv.flatten(), 0).dot(gg_value).T
            ).dot(gg_value))
        azz = zet / z + np.dot(a.T, (a / diaglamyi))
        axz = np.dot(-gg_value.T, (a / diaglamyi))
        bx = delx + np.dot(gg_value.T, (dellamyi / diaglamyi))
        bz = delz - np.dot(a.T, (dellamyi / diaglamyi))
        aar1_value = np.concatenate((axx_value, axz), axis=1)
        aar2_value = np.concatenate((axz.T, azz), axis=1)
        aa_value = np.concatenate((aar1_value, aar2_value), axis=0)
        bb = np.concatenate((-bx, -bz), axis=0)
        solut = scipy.linalg.solve(aa_value, bb)
        dx = solut[0:n]
        dz = solut[n:n + 1]
        dlam = np.dot(gg_value, dx) / diaglamyi - dz * (
            a / diaglamyi) + dellamyi / diaglamyi
        # End if m<n

      dy = -dely / diagy + dlam / diagy
      dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
      deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
      dmu = -mu + epsvecm / y - (mu * dy) / y
      dzet = -zet + epsi / z - zet * dz / z
      ds = -s + epsvecm / lam - (s * dlam) / lam
      xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
      dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)

      stepxx = -1.01 * dxx / xx
      stmxx = np.max(stepxx)
      stepalfa = -1.01 * dx / (x - alfa)
      stmalfa = np.max(stepalfa)
      stepbeta = 1.01 * dx / (beta - x)
      stmbeta = np.max(stepbeta)
      stmalbe = np.maximum(stmalfa, stmbeta)
      stmalbexx = np.maximum(stmalbe, stmxx)
      stminv = np.maximum(stmalbexx, 1.0)
      steg = 1.0 / stminv

      xold = x.copy()
      yold = y.copy()
      zold = z.copy()
      lamold = lam.copy()
      xsiold = xsi.copy()
      etaold = eta.copy()
      muold = mu.copy()
      zetold = zet.copy()
      sold = s.copy()

      #
      itto = 0
      resinew = _SUBSOLV_RESIDUE_NORM_FACTOR * residunorm
      # Start: while (resinew>residunorm) and (itto<50)
      while ((resinew > residunorm) and (itto < _SUBSOLV_MAX_OUTER_ITER)):
        itto = itto + 1
        x = xold + steg * dx
        y = yold + steg * dy
        z = zold + steg * dz
        lam = lamold + steg * dlam
        xsi = xsiold + steg * dxsi
        eta = etaold + steg * deta
        mu = muold + steg * dmu
        zet = zetold + steg * dzet
        s = sold + steg * ds

        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1

        plam = p0 + np.dot(p_value.T, lam)
        qlam = q0 + np.dot(q_value.T, lam)

        gvec = np.dot(p_value, uxinv1) + np.dot(q_value, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2

        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.dot(a.T, lam)
        relam = gvec - np.dot(a, z) - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = np.dot(zet, z) - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res),
                                 axis=0)
        residu = np.concatenate((residu1, residu2), axis=0)
        resinew = np.sqrt(np.dot(residu.T, residu))
        steg = steg / 2
        # End: while (resinew>residunorm) and (itto<50)
      residunorm = resinew.copy()
      residumax = max(abs(residu))
      steg = 2 * steg
      # End: while (residumax>0.9*epsi) and (ittt<200)
    epsi = _SUBSOLV_EPSI_FACTOR * epsi
    # End: while epsi>epsimin
  xmma = x.copy()
  ymma = y.copy()
  zmma = z.copy()
  lamma = lam
  xsimma = xsi
  etamma = eta
  mumma = mu
  zetmma = zet
  smma = s
  # Return values
  return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma


def _kktcheck(m: int, n: int, x: np.ndarray, y: np.ndarray, z: np.ndarray,
              lam: np.ndarray, xsi: np.ndarray, eta: np.ndarray, mu: np.ndarray,
              zet: np.ndarray, s: np.ndarray, xmin: np.ndarray,
              xmax: np.ndarray, df0dx: np.ndarray, fval: np.ndarray,
              dfdx: np.ndarray, a0: float, a: np.ndarray, c: np.ndarray,
              d: np.ndarray) -> tuple[np.ndarray, float, float]:
  """Checks if KKT condition is satisfied.

  The left hand sides of the KKT conditions for the following nonlinear
  programming problem are calculated.

  Minimize f_0(x) + a_0*z + sum(c_i*y_i + 0.5*d_i*(y_i)^2)
  subject to  f_i(x) - a_i*z - y_i <= 0,   i = 1,...,m
              xmax_j <= x_j <= xmin_j,     j = 1,...,n
              z >= 0,   y_i >= 0,          i = 1,...,m

  Args:
    m: The number of general constraints.
    n: The number of variables x_j.
    x: Current values of the n variables x_j.
    y: Current values of the m variables y_i.
    z: Current value of the single variable z.
    lam: Lagrange multipliers for the m general constraints.
    xsi: Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
    eta: Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
    mu: Lagrange multipliers for the m constraints -y_i <= 0.
    zet: Lagrange multiplier for the single constraint -z <= 0.
    s: Slack variables for the m general constraints.
    xmin: Lower bounds for the variables x_j.
    xmax: Upper bounds for the variables x_j.
    df0dx: Vector with the derivatives of the objective function f_0 with
      respect to the variables x_j, calculated at x.
    fval: Vector with the values of the constraint functions f_i, calculated at
      x.
    dfdx: (m x n)-matrix with the derivatives of the constraint functions f_i
      with respect to the variables x_j, calculated at x. dfdx(i,j) = the
      derivative of f_i with respect to x_j.
    a0: The constants a_0 in the term a_0*z.
    a: Vector with the constants a_i in the terms a_i*z.
    c: Vector with the constants c_i in the terms c_i*y_i.
    d: Vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.

  Returns:
    residu: the residual vector for the KKT conditions.
    residunorm: sqrt(residu'*residu).
    residumax: max(abs(residu)).
  """
  del m, n
  rex = df0dx + np.dot(dfdx.T, lam) - xsi + eta
  rey = c + d * y - mu - lam
  rez = a0 - zet - np.dot(a.T, lam)

  relam = fval - a * z - y + s
  rexsi = xsi * (x - xmin)
  reeta = eta * (xmax - x)
  remu = mu * y
  rezet = zet * z
  res = lam * s

  residu1 = np.concatenate((rex, rey, rez), axis=0)
  residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
  residu = np.concatenate((residu1, residu2), axis=0)
  residunorm = np.sqrt((np.dot(residu.T, residu)).item())
  residumax = np.max(np.abs(residu))

  return residu, residunorm, residumax
