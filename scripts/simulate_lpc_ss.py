"""Performs an open-loop rollout of the L+C state-space dynamics."""

import ipdb
import jax
import jax.random as jr

from jax_guam.lpc_state_space import LiftPlusCruise
from jax_guam.utils.jax_utils import jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger


def main():
    jax_use_cpu()
    jax_use_double()
    set_logger_format()

    logger.info("Constructing dynamics...")
    dynamics = LiftPlusCruise()
    logger.info("Calling dynamics...")

    T = 5000
    x_init = dynamics.get_default_x_init()

    # Perturb the initial state in the x and y directions.
    key0, key1 = jr.split(jr.PRNGKey(0))
    x_init = x_init.at[0:2].set(jr.uniform(key0, (2,), minval=-20.0, maxval=20.0))

    # Generate random controls
    u_traj = jr.uniform(
        key1, (T, dynamics.n_u), minval=dynamics.u_min, maxval=dynamics.u_max
    )

    def forward(x, u):
        x_next = dynamics.forward(x, u)
        return x_next, x_next

    _, x_traj = jax.lax.scan(forward, x_init, u_traj)

    import matplotlib.pyplot as plt

    plt.plot(x_traj[:, 0:3])
    plt.grid(alpha=0.5)
    plt.legend(["x", "y", "z"])
    plt.show()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
