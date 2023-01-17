import functools as ft
from types import SimpleNamespace
from typing import Optional

import diffrax as dfx
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

class LTISystem(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray

def harmonic_oscillator(damping: float = 0.0, time_scaling: float = 1.0) -> LTISystem:
    A = jnp.array([[0.0, time_scaling], [-time_scaling, -2 * damping]])
    B = jnp.array([[0.0], [1.0]])
    C = jnp.array([[0.0, 1.0]])
    return LTISystem(A, B, C)


def interpolate_us(ts, us, B):
    if us is None:
        m = B.shape[-1]
        u_t = SimpleNamespace(evaluate=lambda t: jnp.zeros((m,)))
    else:
        u_t = dfx.LinearInterpolation(ts=ts, ys=us)
    return u_t


def diffeqsolve(
    rhs,
    ts: jnp.ndarray,
    y0: jnp.ndarray,
    solver: dfx.AbstractSolver = dfx.Dopri5(),
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    dt0: float = 0.01,
) -> jnp.ndarray:
    return dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=ts[0],
        t1=ts[-1],
        y0=y0,
        dt0=dt0,
        saveat=dfx.SaveAt(ts=ts),
    ).ys


def simulate_lti_system(
    sys: LTISystem,
    y0: jnp.ndarray,
    ts: jnp.ndarray,
    us: Optional[jnp.ndarray] = None,
    std_measurement_noise: float = 0.0,
    key=jr.PRNGKey(
        1,
    ),
):
    u_t = interpolate_us(ts, us, sys.B)

    def rhs(t, y, args):
        return sys.A @ y + sys.B @ u_t.evaluate(t)

    xs = diffeqsolve(rhs, ts, y0)
    # noisy measurements
    ys = xs @ sys.C.transpose()
    ys = ys + jr.normal(key, shape=ys.shape) * std_measurement_noise
    return xs, ys


class KalmanFilter(eqx.Module):
    """Continuous-time Kalman Filter

    Ref:
        [1] Optimal and robust estimation. 2nd edition. Page 154.
        https://lewisgroup.uta.edu/ee5322/lectures/CTKalmanFilterNew.pdf
    """

    sys: LTISystem
    x0: jnp.ndarray
    P0: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray

    def __call__(self, ts, ys, us: Optional[jnp.ndarray] = None):

        A, B, C = self.sys.A, self.sys.B, self.sys.C

        y_t = dfx.LinearInterpolation(ts=ts, ys=ys)
        u_t = interpolate_us(ts, us, B)

        y0 = (self.x0, self.P0)

        def rhs(t, y, args):
            x, P = y

            # eq 3.22 of Ref [1]
            K = P @ C.transpose() @ jnp.linalg.inv(self.R)

            # eq 3.21 of Ref [1]
            dPdt = (
                A @ P
                + P @ A.transpose()
                + self.Q
                - P @ C.transpose() @ jnp.linalg.inv(self.R) @ C @ P
            )

            # eq 3.23 of Ref [1]
            dxdt = A @ x + B @ u_t.evaluate(t) + K @ (y_t.evaluate(t) - C @ x)

            return (dxdt, dPdt)

        return diffeqsolve(rhs, ts, y0)[0]


def main(
    # evaluate at these timepoints
    ts=jnp.arange(0, 5.0, 0.01),
    # system that generates data
    sys_true=harmonic_oscillator(0.3),
    # initial state of our data generating system
    sys_true_x0=jnp.array([1.0, 0.0]),
    # standard deviation of measurement noise
    sys_true_std_measurement_noise=0.1,
    # our model for system `true`, it's not perfect
    sys_model=harmonic_oscillator(0.7),
    # initial state guess, it's not perfect
    sys_model_x0=jnp.array([0.0, 0.0]),
    # weighs how much we trust our model of the system
    Q=jnp.diag(jnp.ones((2,))) * 0.1,#100000.,
    # weighs how much we trust in the measurements of the system
    R=jnp.diag(jnp.ones((1,))),
    # weighs how much we trust our initial guess
    P0=jnp.diag(jnp.ones((2,))) * 1.0,
    plot=True,
    n_gradient_steps=0,
    print_every=100,
):

    xs, ys = simulate_lti_system(
        sys_true, sys_true_x0, ts, std_measurement_noise=sys_true_std_measurement_noise
    )
    print(P0.shape)
    kmf = KalmanFilter(sys_model, sys_model_x0, P0, Q, R)
    #print(kmf(ts, ys))
    print(f"Initial Q: \n{kmf.Q}\n Initial R: \n{kmf.R}")

    # gradients should only be able to change Q/R parameters
    # *not* the model (well at least not in this example :)
    filter_spec = jtu.tree_map(lambda arr: False, kmf)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.Q, tree.R), filter_spec, replace=(True, True)
    )

    @eqx.filter_jit
    @ft.partial(eqx.filter_value_and_grad, arg=filter_spec)
    def loss_fn(kmf, ts, ys, xs):
        xhats = kmf(ts, ys)
        # minimize error between true state and estimated state
        return jnp.mean((xs - xhats) ** 2)

  
    schedule = optax.exponential_decay(
        init_value=1e-4,
        transition_steps=2000,
        decay_rate=0.99,
        transition_begin=0,
        staircase=False,
        end_value=1e-5
    )
    opt = optax.chain(
        optax.adabelief(learning_rate=schedule)
    )
    opt_state = opt.init(kmf)

    for step in range(n_gradient_steps):
        value, grads = loss_fn(kmf, ts, ys, xs)
        if step % print_every == 0:
            print(f"Current MSE at step {step}: {value}")
        updates, opt_state = opt.update(grads, opt_state)
        kmf = eqx.apply_updates(kmf, updates)

    print(f"Final Q: \n{kmf.Q}\n Final R: \n{kmf.R}")

    if plot:
        xhats = kmf(ts, ys)
        plt.plot(ts, xs[:, 0], label="true position", color="orange")
        plt.plot(
            ts,
            xhats[:, 0],
            label="estimated position",
            color="orange",
            linestyle="dashed",
        )
        plt.plot(ts, xs[:, 1], label="true velocity", color="blue")
        plt.plot(
            ts,
            xhats[:, 1],
            label="estimated velocity",
            color="blue",
            linestyle="dashed",
        )
        plt.plot(ts, ys, label="measured velocity", color="red")
        plt.xlabel("time")
        plt.ylabel("position / velocity")
        plt.grid()
        plt.legend()
        plt.title("Kalman-Filter optimization w.r.t Q/R")
        plt.savefig('kmf-example.png')

main(n_gradient_steps=3000)
