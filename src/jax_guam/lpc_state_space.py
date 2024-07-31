"""Lift+Cruise state space model.

The state is a Vector13, partitioned as follows:
    x = [pos, quat, body_vel, body_omega]
where:
    pos: position in NED inertial frame
    quat: quaternion representing the rotation from inertial to body frame
    body_vel: velocity in body frame
    body_omega: angular velocity in body frame

The control is a Vector14, partitioned as follows:
    u = [left_flaperon, right_flaperon, left_elevator, right_elevator, rudder, 8 lifting rotors, pusher propeller]
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .functional.aero_prop_new import FuncAeroProp
from .functional.vehicle_eom_simple import VehicleEOMSimple
from .subsystems.environment.environment import Environment
from .guam_types import AircraftStateVec, PropAct, SurfAct

# NOTE: the order of the aircraft state is (vel, pqr, pos, quat)
# This is a little less intuitive when working with a SS model, so we'll reorder it to (pos, quat, vel, pqr)


State = Float[Array, "13"]
Control = Float[Array, "14"]


# Indices of the state space vector
pos_inds = jnp.array([0, 1, 2])
quat_inds = jnp.array([3, 4, 5, 6])
vel_inds = jnp.array([7, 8, 9])
omega_inds = jnp.array([10, 11, 12])

ss_to_aircraft_inds = jnp.concatenate([vel_inds, omega_inds, pos_inds, quat_inds])


# Indices of the aircraft state vector
aircraft_vel_inds = jnp.array([0, 1, 2])
aircraft_omega_inds = jnp.array([3, 4, 5])
aircraft_pos_inds = jnp.array([6, 7, 8])
aircraft_quat_inds = jnp.array([9, 10, 11, 12])

aircraft_to_ss_inds = jnp.concatenate(
    [aircraft_pos_inds, aircraft_quat_inds, aircraft_vel_inds, aircraft_omega_inds]
)


# Indices of the control vector
surf_inds = jnp.array([0, 1, 2, 3, 4])
prop_inds = jnp.array([5, 6, 7, 8, 9, 10, 11, 12, 13])


def state_space_to_aircraft_state(x: State) -> AircraftStateVec:
    """Converts the state space vector to an aircraft state vector."""
    return x[ss_to_aircraft_inds]


def aircraft_state_to_state_space(aircraft_state: AircraftStateVec) -> State:
    """Converts an aircraft state vector to the state space vector."""
    return aircraft_state[aircraft_to_ss_inds]


class LiftPlusCruise:
    """Lift+Cruise state space model."""

    def __init__(self, dt: float = 0.005):
        self.dt = dt

        # Initialize the L+C aircraft dynamics model
        self.aero_prop = FuncAeroProp()

        # Initialize equations of motion for the kinematics/dynamics
        self.veh_eom = VehicleEOMSimple()

        # Initialize the environment
        self._environment = Environment()

    def f(self, x: State, u: Control) -> State:
        r"""Computes the continuous-time dynamics `\dot{x} = f(x, u)`."""

        # Convert state to an aircraft state vector
        aircraft_state = state_space_to_aircraft_state(x)

        # Compute environment-specific data based on the altitude, etc.
        sensor, aeroprop_body_data, alt_msl = self.veh_eom.get_sensor_aeroprop_altmsl(
            aircraft_state
        )
        atmosphere = self._environment.get_env_atmosphere(alt_msl)
        env_data = self._environment.Env._replace(Atmosphere=atmosphere)

        # Construct the control data structures
        # TODO: As far as I can tell, only the first element of these tuples is important, double check this
        prop_act = PropAct(
            EngSpeed=u[prop_inds], EngAccel=jnp.zeros(9), Failure_Engines=None
        )
        surf_act = SurfAct(
            CtrlSurfPos=u[surf_inds], CtrlSurfRate=jnp.zeros(5), Failure_Surfaces=None
        )

        # Compute the forces and moments acting on the aircraft
        fm = self.aero_prop.aero_prop(prop_act, surf_act, env_data, aeroprop_body_data)
        fm_total = self.veh_eom.get_fm_with_gravity(aircraft_state, fm)

        # Compute the state derivative
        d_state_aircraft = self.veh_eom.state_deriv(fm_total, aircraft_state)

        # Convert back to state space indexing
        x_dot = aircraft_state_to_state_space(d_state_aircraft)

        return x_dot

    def forward(self, x: State, u: Control) -> State:
        r"""Computes the discrete-time dynamics `x' = F(x, u)`."""
        # Euler integration
        return x + self.f(x, u) * self.dt
