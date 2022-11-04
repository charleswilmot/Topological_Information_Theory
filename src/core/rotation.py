import jax
from jax.scipy.linalg import expm
import jax.random as random
import jax.numpy as jnp


def anti_symetric(angles, dim):
    ret = jnp.zeros(shape=angles.shape[:-1] + (dim, dim))
    k = 0
    for i in range(1, dim):
        for j in range(i):
            v = angles[..., k]
            ret = ret.at[..., i, j].set(-v)
            ret = ret.at[..., j, i].set(+v)
            k += 1
    return ret


def rotation_matrix(angles, dim):
    return expm(anti_symetric(angles, dim))


v_expm = jax.vmap(expm)


def v_rotation_matrix(angles, dim):
    return v_expm(anti_symetric(angles, dim))


if __name__ == '__main__':
    print(rotation_matrix(jnp.array([jnp.pi, 0, 0]), dim=3))
    print()
    print(v_rotation_matrix(jnp.array([[jnp.pi, 0, 0], [0, jnp.pi, 0]]), dim=3))
