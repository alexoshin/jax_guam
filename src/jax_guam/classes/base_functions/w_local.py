import jax.numpy as jnp

from jax_guam.utils.jax_types import RowVec4, RowVec6, RowVec8


def w_local(x: RowVec6, u: RowVec4, w: RowVec8, ders: bool):
    assert isinstance(ders, bool) and ders is False
    n_points = len(x)
    assert x.shape == (n_points, 6) and u.shape == (n_points, 4) and w.shape == (n_points, 8)
    N = x.shape[0]

    uu = x[:, 0]
    vv = x[:, 1]
    ww = x[:, 2]
    p = x[:, 3]
    q = x[:, 4]
    r = x[:, 5]

    T = u[:, 0]
    To = u[:, 1]
    del_f = u[:, 2]
    i = u[:, 3]

    bx = w[:, 0]
    by = w[:, 1]
    bz = w[:, 2]
    gam = w[:, 7]

    c_i = jnp.cos(i)
    s_i = jnp.sin(i)
    c_gam = jnp.cos(gam)
    s_gam = jnp.sin(gam)

    y = s_i * (uu + q * bz - r * by) + c_gam * c_i * (ww + p * by - q * bx) - c_i * s_gam * (vv - p * bz + r * bx)

    if ders:
        y_x = jnp.column_stack(
            [
                s_i,
                -s_gam * c_i,
                c_gam * c_i,
                c_gam * c_i * by + s_gam * c_i * bz,
                s_i * bz - c_gam * c_i * bx,
                -s_i * by - s_gam * c_i * bx,
            ]
        )

        y_u = jnp.column_stack(
            [
                jnp.zeros(N),
                jnp.zeros(N),
                jnp.zeros(N),
                c_i * (uu + q * bz - r * by)
                - c_gam * s_i * (ww + p * by - q * bx)
                + s_gam * s_i * (vv - p * bz + r * bx),
            ]
        )
    else:
        y_x = jnp.zeros((N, 6))
        y_u = jnp.zeros((N, 4))

    return jnp.expand_dims(y, 1), y_x, y_u
