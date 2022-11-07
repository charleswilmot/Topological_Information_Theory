import os
import numpy as np
from jax import random
import jax.numpy as jnp
import jax.lax as lax
import haiku as hk
from . import visualize
import logging
import re


# A logger for this file
log = logging.getLogger(__name__)


class NDShapeBase:
    def __init__(self, name=""):
        self._name = name
        self._str_params = {}
        if name:
            self._str_params["name"] = self._name

    @staticmethod
    def by_name(name):
        if m := re.match('hypercube(?P<dimension>[0-9]+)', name):
            return HyperCube(int(m.group('dimension')))
        if m := re.match('hypersphere(?P<dimension>[0-9]+)', name):
            return HyperSphere(int(m.group('dimension')))
        if m := re.match('hypertorus(?P<dimension>[0-9]+)', name):
            return HyperTorus(int(m.group('dimension')))
        if m := re.match('moebiusstrip', name):
            return MoebiusStrip()
        if m := re.match('kleinbottle', name):
            return KleinTube()
        raise ValueError(f"{name} is not recognized as a NdShape")

    def __mul__(self, other):
        return NDShapeProduct(self, other)

    def __str__(self):
        params = ' '.join(f'{key}={value}' for key, value in self._str_params.items())
        return f'<{self.__class__.__name__} {params}>'

    @staticmethod
    def visualize_samples(fig, samples, color=None):
        embedding_dimension = samples.shape[1]

        if embedding_dimension == 2:
            visualize._2d_samples(fig, samples, color)
        elif embedding_dimension == 3:
            visualize._3d_samples(fig, samples, color)
        elif embedding_dimension == 4:
            visualize._4d_samples(fig, samples, color)
        else:
            raise ValueError(f"Can't visualize this embedding_dimension ({embedding_dimension} > 4)")

    def __hash__(self):
        return self._name.__hash__()


class HyperCube(NDShapeBase):
    def __init__(self, dimension):
        super().__init__(name=f'hypercube{dimension}')
        self._manifold_dimension = dimension
        self._embedding_dimension = dimension
        self._str_params["dimension"] = dimension

    def sample(self, key, n):
        key, = random.split(key, 1)
        return random.uniform(
            key=key,
            minval=-1,
            maxval=1,
            shape=(n, self._manifold_dimension),
            dtype=jnp.float32,
        )

    def mesh(self, n):
        return jnp.mgrid[
            [slice(-1,1,1j * n) for _ in range(self._manifold_dimension)]
        ].T.reshape((n ** self._manifold_dimension, self._manifold_dimension))

    @staticmethod
    def project(points):
        return points, {}

    @staticmethod
    def regularize(z, projection_reg_coef, shape_reg_coef):
        embedding_dimension = z.shape[-1]
        #
        barycenter = jnp.mean(z, axis=0)
        barycenter_squared_dist = jnp.sum(barycenter ** 2)
        clipped_barycenter_squared_dist = jnp.clip(barycenter_squared_dist, a_min=embedding_dimension * 0.15 ** 2)
        #
        return shape_reg_coef * jnp.sum(clipped_barycenter_squared_dist)


class MultivariateGaussian(NDShapeBase):
    def __init__(self, dimension):
        super().__init__(name=f'multivariategaussian{dimension}')
        self._manifold_dimension = dimension
        self._embedding_dimension = dimension
        self._str_params["dimension"] = dimension

    def sample(self, key, n):
        key, = random.split(key, 1)
        return random.normal(
            key=key,
            shape=(n, self._manifold_dimension),
            dtype=jnp.float32,
        )

    def mesh(self, n):
        return jnp.mgrid[
            [slice(-1,1,1j * n) for _ in range(self._manifold_dimension)]
        ].T.reshape((n ** self._manifold_dimension, self._manifold_dimension))

    @staticmethod
    def project(points):
        return points, {}

    @staticmethod
    def regularize(z, projection_reg_coef, shape_reg_coef):
        embedding_dimension = z.shape[-1]
        #
        barycenter = jnp.mean(z, axis=0)
        barycenter_squared_dist = jnp.sum(barycenter ** 2)
        clipped_barycenter_squared_dist = jnp.clip(barycenter_squared_dist, a_min=embedding_dimension * 0.15 ** 2)
        #
        return shape_reg_coef * jnp.sum(clipped_barycenter_squared_dist)


class HyperSphere(NDShapeBase):
    def __init__(self, dimension):
        super().__init__(name=f'hypersphere{dimension}')
        self._manifold_dimension = dimension
        self._embedding_dimension = dimension + 1
        self._str_params["dimension"] = dimension

    def sample(self, key, n):
        key, = random.split(key, 1)
        normal_deviates = random.normal(
            key=key,
            shape=(n, self._embedding_dimension),
            dtype=jnp.float32,
        )
        radius = jnp.sqrt(jnp.sum(normal_deviates ** 2, axis=-1, keepdims=True))
        return normal_deviates / radius

    def mesh(self, n):
        angles = jnp.mgrid[
            [slice(0, np.pi, 1j * n) for _ in range(self._manifold_dimension - 1)] +
            [slice(0, 2 * np.pi, 1j * n)]
        ].T.reshape((n ** self._manifold_dimension, self._manifold_dimension))
        angles_cos = jnp.cos(angles)
        angles_sin = jnp.sin(angles)
        angles_cumsin = jnp.cumprod(angles_sin, axis=-1)
        points = jnp.stack(
            [angles_cos[..., 0]] +
            [angles_cumsin[..., i - 1] * angles_cos[..., i] for i in range(1, self._manifold_dimension)] +
            [angles_cumsin[..., self._manifold_dimension - 1]],
            axis=-1
        )
        return points

    @staticmethod
    def project(points):
        dimension = points.shape[-1]
        squares = points ** 2
        norm = jnp.sqrt(jnp.sum(squares, axis=-1, keepdims=True))
        points = points / norm
        metadata = {}
        # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        cumsum_squares = jnp.cumsum(squares[..., ::-1], axis=-1)
        for i in range(dimension - 1):
            metadata[f'theta{i}'] = jnp.arccos(points[..., i] / jnp.sqrt(cumsum_squares[..., dimension - i]))
        sign = jnp.where(points[..., -1] < 0, -1, 1)
        offset = jnp.where(points[..., -1] < 0, 2 * jnp.pi, 0)
        metadata[f'theta{i}'] = offset + sign * metadata[f'theta{i}']
        return points, metadata

    @staticmethod
    def regularize(z, projection_reg_coef, shape_reg_coef):
        embedding_dimension = z.shape[-1]
        projection, metadata = HyperSphere.project(z)
        #
        loss_per_projection = jnp.mean((projection - z) ** 2, axis=-1)
        #
        barycenter = jnp.mean(z, axis=0)
        barycenter_squared_dist = jnp.sum(barycenter ** 2)
        clipped_barycenter_squared_dist = jnp.clip(barycenter_squared_dist, a_min=embedding_dimension * 0.15 ** 2)
        #
        squared_dist = jnp.sum(z ** 2, axis=-1)
        clipped_squared_dist = jnp.clip(squared_dist, a_max=embedding_dimension * 0.5 ** 2)
        #
        return (
            + projection_reg_coef * jnp.sum(loss_per_projection)
            + shape_reg_coef * jnp.sum(clipped_barycenter_squared_dist)
            - shape_reg_coef * jnp.sum(clipped_squared_dist)
        )


class HyperTorus(NDShapeBase):
    def __init__(self, dimension):
        super().__init__(name=f'hypertorus{dimension}')
        self._manifold_dimension = dimension
        self._embedding_dimension = 2 * dimension
        self._str_params["dimension"] = dimension

    def sample(self, key, n):
        key, = random.split(key, 1)
        angles = random.uniform(
            key=key,
            minval=0,
            maxval=2 * np.pi,
            shape=(n, self._manifold_dimension),
            dtype=jnp.float32,
        )
        return jnp.concatenate([
            jnp.sin(angles),
            jnp.cos(angles),
        ], axis=-1)

    def mesh(self, n):
        angles = jnp.mgrid[
            [slice(0, 2 * np.pi, 1j * n) for _ in range(self._manifold_dimension)]
        ].T.reshape((n ** self._manifold_dimension, self._manifold_dimension))
        return jnp.concatenate([
            jnp.sin(angles),
            jnp.cos(angles),
        ], axis=-1)

    @staticmethod
    def project(points):
        if points.shape[-1] % 2 != 0:
            raise ValueError(f"The embedding has a dimension {points.shape[-1]}, can't project on a torus")
        dimension = points.shape[-1] // 2
        points = points.reshape(points.shape[:-1] + (2, dimension))   # [N, 2, dim]
        norm = jnp.sqrt(jnp.sum(points ** 2, axis=-2, keepdims=True)) # [N, 1, dim]
        points = points / norm
        angle = jnp.arctan2(points[..., 0, :], points[..., 1, :])
        points = points.reshape(points.shape[:-2] + (dimension * 2,))
        metadata = {f"theta{i}": angle[..., i] for i in range(dimension)}
        return points, metadata

    @staticmethod
    def visualize_samples(fig, samples, color=None):
        embedding_dimension = samples.shape[1]
        manifold_dimension = embedding_dimension // 2
        axs = [fig.add_subplot(1, manifold_dimension, i + 1) for i in range(manifold_dimension)]
        for ax in axs: # turn off tick and label
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_ticklabels([])
        for i, ax in enumerate(axs):
            ax.scatter(samples[:, i], samples[:, manifold_dimension + i], c=color)

    @staticmethod
    def regularize(z, projection_reg_coef, shape_reg_coef):
        if z.shape[-1] % 2 != 0:
            raise ValueError(f"The embedding has a dimension {z.shape[-1]}, can't project on a torus")
        manifold_dimension = z.shape[-1] // 2
        projection, metadata = HyperTorus.project(z)
        #
        loss_per_projection = jnp.mean((projection - z) ** 2, axis=-1)
        #
        barycenter = jnp.mean(z, axis=0)
        barycenter_squared_dist = barycenter ** 2 # shape [manifold_dimension * 2]
        barycenter_squared_dist = barycenter_squared_dist.reshape((2, manifold_dimension))
        barycenter_squared_dist = jnp.sum(barycenter_squared_dist, axis=-2)
        clipped_barycenter_squared_dist = jnp.clip(barycenter_squared_dist, a_min=0.15 ** 2)
        return (
            + projection_reg_coef * jnp.sum(loss_per_projection)
            + shape_reg_coef * jnp.sum(clipped_barycenter_squared_dist)
        )


class ReparameterizedShape(NDShapeBase):
    def __init__(self, name=""):
        super().__init__(name=name)
        self._acceptation_rate = 0.5
        self._margin = 1.01
        self._S, self._T = 0, 0

    def sample(self, key, n):
        # https://corybrunson.github.io/2019/02/01/sampling/
        remaining = n
        agg = []
        while remaining > 0:
            key, = random.split(key, 1)
            n_assumed = int(self._margin * remaining / self._acceptation_rate)
            parameters = self._sample_parameters(key, n_assumed)
            jacobians = self._jacobian_determinant(*parameters)
            eta = random.uniform(
                key=key,
                minval=0,
                maxval=self._jacobian_determinant_max,
                shape=(n_assumed,),
                dtype=jnp.float32,
            )
            where = jacobians > eta
            s = np.sum(where)
            t = n_assumed
            self._S += s
            self._T += n_assumed
            self._acceptation_rate = self._S / self._T
            parameters = tuple(p[where][:remaining] for p in parameters)
            agg.append(parameters)
            remaining -= len(parameters[0])
            # log.info(f'found {s=} out of {t=} (acceptation_rate={self._acceptation_rate}), to be found {remaining=}')
        parameters = tuple(
            np.concatenate(tuple(p[i] for p in agg), axis=-1)
            for i in range(len(agg[0]))
        )
        return self.embed(*parameters)


class KleinTube(ReparameterizedShape):
    _R = 0.5

    def __init__(self):
        super().__init__(name='kleinbottle')
        self._manifold_dimension = 2
        self._embedding_dimension = 4
        self._jacobian_determinant_max = self._jacobian_determinant(0, 0)
        self._acceptation_rate = 0.67

    def embed(self, theta, phi):
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        cos_theta_2 = jnp.cos(theta / 2)
        sin_theta_2 = jnp.sin(theta / 2)
        return jnp.stack([
            (1 + KleinTube._R * cos_phi) * cos_theta,
            (1 + KleinTube._R * cos_phi) * sin_theta,
            KleinTube._R * sin_phi * cos_theta_2,
            KleinTube._R * sin_phi * sin_theta_2,
        ], axis=-1)

    def _jacobian_determinant(self, theta, phi):
        return KleinTube._R * jnp.sqrt(
            (1.0 + KleinTube._R * jnp.cos(phi)) ** 2 +
            (0.5 * KleinTube._R * jnp.sin(phi)) ** 2
        )

    def _sample_parameters(self, key, n):
        key1, key2 = random.split(key)
        theta = random.uniform(
            key=key1,
            minval=0,
            maxval=np.pi * 2,
            shape=(n,),
            dtype=jnp.float32,
        )
        phi = random.uniform(
            key=key2,
            minval=0,
            maxval=np.pi * 2,
            shape=(n,),
            dtype=jnp.float32,
        )
        return (theta, phi)

    def mesh(self, n):
        params = jnp.mgrid[
            0:2 * np.pi:1j * n,
            0:2 * np.pi:1j * n,
        ].reshape((self._manifold_dimension, n ** self._manifold_dimension))
        return self.embed(*params)

    @staticmethod
    def project(points):
        if points.shape[-1] != 4:
            raise ValueError(f"The embedding has a dimension {points.shape[-1]}, can't project on a klein bottle")
        theta = jnp.arctan2(points[..., 1], points[..., 0])
        phi = jnp.arctan2(
            jnp.sqrt(points[..., 2] ** 2 + points[..., 3] ** 2),
            jnp.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2) - 1,
        )
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        cos_theta_2, sin_theta_2 = jnp.cos(theta / 2), jnp.sin(theta / 2)
        phi = jnp.where(
            jnp.logical_or(
                jnp.logical_and(cos_theta_2 > 0, points[..., 2] > 0),
                jnp.logical_and(cos_theta_2 < 0, points[..., 2] < 0),
            ),
            phi,
            -phi
        )
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        a = (1 + KleinTube._R * cos_phi) * cos_theta
        b = (1 + KleinTube._R * cos_phi) * sin_theta
        c = KleinTube._R * sin_phi * cos_theta_2
        d = KleinTube._R * sin_phi * sin_theta_2
        points = jnp.stack([a, b, c, d], axis=-1)
        metadata = {}
        metadata["theta"] = theta
        metadata["phi"] = phi
        return points, metadata

    @staticmethod
    def regularize(z, projection_reg_coef, shape_reg_coef):
        projection, metadata = KleinTube.project(z)
        #
        loss_per_projection = jnp.mean((projection - z) ** 2, axis=-1)
        #
        barycenter = jnp.mean(z, axis=0)
        barycenter_squared_dist = barycenter[..., 0] ** 2 + barycenter[..., 1] ** 2
        clipped_barycenter_squared_dist = jnp.clip(barycenter_squared_dist, a_min=0.15 ** 2)
        #
        xy_squared_dist = z[..., 0] ** 2 + z[..., 1] ** 2
        clipped_squared_dist = jnp.clip(xy_squared_dist, a_max=0.5 ** 2)
        #
        return (
            + projection_reg_coef * jnp.sum(loss_per_projection)
            + shape_reg_coef * jnp.sum(clipped_barycenter_squared_dist)
            - shape_reg_coef * jnp.sum(clipped_squared_dist)
        )


class MoebiusStrip(ReparameterizedShape):
    _R = 0.5

    def __init__(self):
        super().__init__(name='moebiusstrip')
        self._embedding_dimension = 3
        self._manifold_dimension = 2
        self._jacobian_determinant_max = self._jacobian_determinant(0, 1)
        self._acceptation_rate = 0.65

    def embed(self, theta, d):
        ret = jnp.empty(shape=theta.shape + (3,), dtype=np.float32)
        coef = 1 + MoebiusStrip._R * d * jnp.cos(theta / 2)
        return jnp.stack([
            coef * jnp.cos(theta),
            coef * jnp.sin(theta),
            MoebiusStrip._R * d * jnp.sin(theta / 2),
        ], axis=-1)

    def _jacobian_determinant(self, theta, d):
        cos_theta_2 = jnp.cos(theta / 2)
        return MoebiusStrip._R * jnp.sqrt(
            d ** 2 * MoebiusStrip._R ** 2 * (cos_theta_2 ** 2 + 0.25) +
            2 * d * MoebiusStrip._R * cos_theta_2 +
            1
        )

    def _sample_parameters(self, key, n):
        key1, key2 = random.split(key)
        theta = random.uniform(
            key=key1,
            minval=0,
            maxval=np.pi * 2,
            shape=(n,),
            dtype=jnp.float32,
        )
        d = random.uniform(
            key=key2,
            minval=-1,
            maxval=1,
            shape=(n,),
            dtype=jnp.float32,
        )
        return (theta, d)

    def mesh(self, n):
        params = jnp.mgrid[
            0:2 * np.pi:1j * n,
            -1:1:1j * n,
        ].reshape((self._manifold_dimension, n ** self._manifold_dimension))
        return self.embed(*params)

    def edge(self, n):
        thetas = jnp.linspace(0, 4 * jnp.pi, n)
        ds = jnp.full(shape=n, fill_value=1.0, dtype=jnp.float32)
        return self.embed(thetas, ds)

    @staticmethod
    def project(points):
        if points.shape[-1] != 3:
            raise ValueError(f"The embedding has a dimension {points.shape[-1]}, can't project on a moebius")
        theta = jnp.arctan2(points[..., 1], points[..., 0])
        d = jnp.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)
        phi = jnp.arctan2(
            points[..., 2],
            d - 1,
        ) - (theta / 2)
        cos_phi = jnp.cos(phi)
        l = jnp.sqrt((d - 1) ** 2 + points[..., 2] ** 2)
        l_cos_phi = l * cos_phi
        alpha = 1 + l_cos_phi * jnp.cos(theta / 2)
        beta  = l_cos_phi * jnp.sin(theta / 2)
        a = alpha * jnp.cos(theta)
        b = alpha * jnp.sin(theta)
        c = beta
        points = jnp.stack([a, b, c], axis=-1)
        metadata = {}
        metadata['theta'] = theta
        metadata['c'] = c
        return points, metadata

    @staticmethod
    def regularize(z, projection_reg_coef, shape_reg_coef):
        projection, metadata = MoebiusStrip.project(z)
        #
        loss_per_projection = jnp.mean((projection - z) ** 2, axis=-1)
        #
        barycenter = jnp.mean(z, axis=0)
        barycenter_squared_dist = barycenter[..., 0] ** 2 + barycenter[..., 1] ** 2
        clipped_barycenter_squared_dist = jnp.clip(barycenter_squared_dist, a_min=0.15 ** 2)
        #
        xy_squared_dist = z[..., 0] ** 2 + z[..., 1] ** 2
        clipped_squared_dist = jnp.clip(xy_squared_dist, a_max=0.5 ** 2)
        #
        return (
            + projection_reg_coef * jnp.sum(loss_per_projection)
            + shape_reg_coef * jnp.sum(clipped_barycenter_squared_dist)
            - shape_reg_coef * jnp.sum(clipped_squared_dist)
        )



class NDShapeProduct(NDShapeBase):
    def __init__(self, *args):
        self._factors = []
        for s in args:
            if isinstance(s, NDShapeProduct):
                self._factors += s._factors
            elif issubclass(s.__class__, NDShapeBase):
                self._factors.append(s)
            else:
                raise ValueError(f'{s.__class__} is not a NDShapeBase')
        self._embedding_dimension = sum(s._embedding_dimension for s in self._factors)
        self._manifold_dimension = sum(s._manifold_dimension for s in self._factors)
        super().__init__(name=''.join(f._name for f in self._factors))

    def sample(self, key, n):
        keys = random.split(key, len(self._factors))
        points_to_concat = tuple(s.sample(key, n) for s, key in zip(self._factors, keys))
        return np.concatenate(points_to_concat, axis=-1)

    def __str__(self):
        return ' X '.join(f.__str__() for f in self._factors)

    def mesh(self, n):
        meshes = tuple(f.mesh(n) for f in self._factors)
        ed = self._embedding_dimension
        md = self._manifold_dimension
        res = np.empty(shape=(n,) * md + (ed,), dtype=np.float32)
        start = 0
        for i, (mesh, f) in enumerate(zip(meshes, self._factors)):
            stop = start + f._embedding_dimension
            shape = (
                (1,) * i +
                (n,) * f._manifold_dimension +
                (1,) * (md - i - f._manifold_dimension) +
                (f._embedding_dimension,)
            )
            res[..., start:stop] = mesh.reshape(shape)
            start = stop
        return res.reshape((n ** md, ed))

    def project(self, points):
        start = 0
        metadata = {}
        projections = []
        for i, f in enumerate(self._factors):
            stop = start + f._embedding_dimension
            p, m = f.project(points[..., start:stop])
            projections.append(p)
            metadata[f'{i}__{f._name}'] = m
        return jnp.concatenate(projections, axis=-1), metadata


class CachedNDShape(NDShapeBase):
    def __init__(self, ndshape):
        super().__init__(ndshape._name)
        if isinstance(ndshape, (NDShapeProduct, ScramblesNDShape)):
            raise ValueError(f"{ndshape.__class__.__name__} can not be cached")
        self._ndshape = ndshape
        self._filepath = f'./cache/{ndshape.__class__.__name__}_md_{ndshape._manifold_dimension}_ed_{ndshape._embedding_dimension}.npy'
        if not os.path.isfile(self._filepath):
            raise ValueError(f'Could not find the file {self._filepath}, this class has not been cached yet?')
        self._data = np.load(self._filepath)

    def __str__(self):
        params = ' '.join(f'{key}={value}' for key, value in self._ndshape._str_params.items())
        return f'<{self._ndshape.__class__.__name__}(cached) {params}>'

    def __getattr__(self, itname):
        return self._ndshape.__getattribute__(itname)

    def sample(self, key, n):
        if n > self._data.shape[0]:
            raise ValueError('Not enough cached samples')
        key, = random.split(key, 1)
        maxval = self._data.shape[0] - n
        index = random.randint(key, minval=0, maxval=maxval, shape=())
        return self._data[index:index + n]


# def sample_unit_simplex(manifold_dimension, n):
#     U = np.random.uniform(0, 1, size=(n, manifold_dimension + 1))
#     E = -np.log(U)                       # (n ,d + 1)
#     S = np.sum(E, axis=1, keepdims=True) # (n, 1)
#     return E / S                         # (n ,d + 1)
#
#
# def project_on_simplexes(points_on_unit_simplex, simplexes):
#     # points_on_unit_simplex has shape (n, manifold_dimension + 1)
#     # simplex has shape (n, manifold_dimension + 1, embedding_dimension)
#     assert points_on_unit_simplex.dims == 2
#     assert simplexes.dims == 3
#     return (points_on_unit_simplex[None] @ simplexes)[0] # (n, embedding_dimension)
#
#
# def simplex_volume(simplexes):
#     # Compute the volume via the Cayley-Menger determinant
#     # compute all edge lengths
#     # based on https://www.programcreek.com/python/?CodeExample=get+vol
#     # simplex has shape (..., manifold_dimension + 1, embedding_dimension)
#     # simplex has shape (..., A, B)
#     simplexes = np.array(simplexes)
#     manifold_dimension = simplexes.shape[-2] - 1
#     mat_dim = simplexes.shape[-2] + 1
#     shape = simplexes.shape[:-2]
#     # get squared distance matrix
#     edges = simplexes[..., None, :] - simplexes[..., None, :, :] # (..., A, A, B)
#     squared_distances = np.sum(edges * edges, axis=-1)           # (..., A, A)
#     # define the Cayley–Menger matrix
#     mat = np.ones(shape + (mat_dim, mat_dim), dtype=np.float32)  # (..., A+1, A+1)
#     mat[..., 0, 0]   = 0.0
#     mat[..., 1:, 1:] = squared_distances
#     # get the Cayley–Menger determinant
#     det = np.abs(np.linalg.det(mat))
#     scale = 2 ** (-0.5 * manifold_dimension) / np.math.factorial(manifold_dimension)
#     return scale * np.sqrt(det)
#
#
# class NDShape(NDShapeBase):
#     def __init__(self, vertices, simplexes, name=""):
#         super().__init__(name=name)
#         self._n_vertices = vertices.shape[0]
#         self._n_simplexes = simplexes.shape[0]
#         self._str_params['n_vertices'] = self._n_vertices
#         self._str_params['n_simplexes'] = self._n_simplexes
#         self._embedding_dimension = vertices.shape[1]
#         self._manifold_dimension = simplexes.shape[1]
#
#         if self._manifold_dimension > self._embedding_dimension:
#             raise ValueError(
# f"""Cannot embed a manifold of dimension {self._manifold_dimension}
# in a space of dimension {self._embedding_dimension}""")
#
#         self._vertices = np.copy(vertices)
#         self._simplexes = np.copy(vertices)
#         self.__simplexes_volumes = None
#         self._name = name
#
#     @property
#     def _simplexes_volumes(self):
#         if self.__simplexes_volumes is None:
#             self.__simplexes_volumes = simplex_volume(self._simplexes)
#         return self.__simplexes_volumes
#
#     @classmethod
#     def from_file(cls, filepath, name=""):
#         with open(filepath, 'rb') as file:
#             # 1 byte for _embedding dimension
#             embedding_dimension = np.frombuffer(file.read(1), dtype=np.uint8)
#             # 1 byte for manifold dimension
#             manifold_dimension = np.frombuffer(file.read(1), dtype=np.uint8)
#             # 4 bytes for number of vertices
#             n_vertices = np.frombuffer(file.read(4), dtype=np.uint32)
#             # 4 bytes for number of simplexes
#             n_simplexes = np.frombuffer(file.read(4), dtype=np.uint32)
#             # write all vertices
#             vertices = np.zeros(shape=(n_vertices, embedding_dimension), dtype=np.float32)
#             for vertice_id in range(n_vertices):
#                 for coord_id in range(embedding_dimension):
#                     # 4 bytes per coordinate
#                     coord = np.frombuffer(file.read(4), dtype=np.float32)
#                     vertices[vertice_id, coord_id] = coord
#             simplexes = np.zeros(shape=(n_simplexes, manifold_dimension + 1), dtype=np.float32)
#             for simplex_id in range(n_simplexes):
#                 for vertice_id_id in range(manifold_dimension + 1):
#                     # 4 bytes per simplex id
#                     vertice_id = np.frombuffer(file.read(4), dtype=np.uint32)
#                     simplexes[simplex_id, vertice_id_id] = vertice_id
#         name = name if name else os.path.splitext(os.path.basename(filepath))[0]
#         return cls(vertices, simplexes, name=name)
#
#     def to_file(self, filepath):
#         with open(filepath, 'wb') as file:
#             # 1 byte for embedding dimension
#             file.write(np.uint8(self._embedding_dimension))
#             # 1 byte for manifold dimension
#             file.write(np.uint8(self._manifold_dimension))
#             # 4 bytes for number of vertices
#             file.write(np.uint32(self._n_vertices))
#             # 4 bytes for number of simplexes
#             file.write(np.uint32(self._n_simplexes))
#             # write all vertices
#             for vertice in self._vertices:
#                 for coord in vertice:
#                     # 4 bytes per coordinate
#                     file.write(np.float32(coord))
#             for simplex in self._simplexes:
#                 for vertice_id in simplex:
#                     # 4 bytes per simplex id
#                     file.write(np.uint32(vertice_id))
#
#     def sample(self, n):
#         # select n simplex at random (with weighted probs)
#         simplexes_indices = np.random.choice(
#             self._n_simplexes,
#             size=n,
#             p=self._simplexes_volumes
#         )
#         # select n points uniformly at random
#         unit_points = sample_unit_simplex(self._manifold_dimension, n)
#         points = project_on_simplexes(unit_points, self._simplexes[simplexes_indices])
#         return points


class ScramblesNDShape(NDShapeBase):
    def __init__(self, key, ndshape, depth, dilation_factor=10):
        super().__init__(name=f'scrambled{ndshape._name}')
        self._ndshape = ndshape
        self._dilation_factor = dilation_factor
        self._depth = depth
        self._manifold_dimension = self._ndshape._manifold_dimension
        self._embedding_dimension = self._ndshape._embedding_dimension * dilation_factor
        self._str_params["dilation_factor"] = dilation_factor
        self._str_params["depth"] = depth
        self.__input_mean = None
        self.__input_std = None
        self.__output_mean = None
        self.__output_std = None

        def __network(samples):
            output = hk.Sequential([
                hk.nets.MLP([self._embedding_dimension] * self._depth),
                jnp.tanh,
            ])(samples)
            return output

        self._network = hk.without_apply_rng(hk.transform(__network))
        key, subkey = random.split(key)
        self._network_params = self._network.init(
            subkey,
            np.zeros(shape=(1, self._ndshape._embedding_dimension), dtype=np.float32),
        )

    def _initialize_mean_stds(self, n):
        key = random.PRNGKey(0)
        samples = self._ndshape.sample(key, n)
        self.__input_mean = jnp.mean(samples, axis=0, keepdims=True)
        self.__input_std = jnp.std(samples, axis=0, keepdims=True)
        output = self.scramble(samples, normalize_output=False)
        self.__output_mean = jnp.mean(output, axis=0, keepdims=True)
        self.__output_std = jnp.std(output, axis=0, keepdims=True)

    def scramble(self, samples, normalize_output=True):
        samples = (samples - self._input_mean) / self._input_std
        output = self._network.apply(self._network_params, samples)
        if normalize_output:
            output = (output - self._output_mean) / self._output_std
        return output

    def mesh(self, n):
        return self.scramble(self._ndshape.mesh(n))

    @property
    def _input_mean(self):
        if self.__input_mean is None:
            self._initialize_mean_stds(1000)
        return self.__input_mean

    @property
    def _input_std(self):
        if self.__input_std is None:
            self._initialize_mean_stds(1000)
        return self.__input_std

    @property
    def _output_mean(self):
        if self.__output_mean is None:
            self._initialize_mean_stds(1000)
        return self.__output_mean

    @property
    def _output_std(self):
        if self.__output_std is None:
            self._initialize_mean_stds(1000)
        return self.__output_std

    def sample(self, key, n):
        return self.scramble(self._ndshape.sample(key, n))

    def project(self, points):
        return self._ndshape.project(points)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # key = random.PRNGKey(0)
    #
    # ndshapes = (
    #     # HyperCube(2),
    #     # HyperTorus(1),
    #     # HyperSphere(1),
    #     # HyperSphere(2),
    #     # MultivariateGaussian(2),
    #     # HyperCube(3),
    #     HyperTorus(2),
    #     # HyperSphere(3),
    #     # MultivariateGaussian(3),
    #     # HyperCube(1) * HyperTorus(1),
    #     # HyperCube(1) * MoebiusStrip(),
    #     # MoebiusStrip(),
    #     # KleinTube(),
    # )
    #
    #
    # for ndshape in ndshapes:
    #     print(ndshape)
    #     # samples = ndshape.sample(key, 2000)
    #     samples = ndshape.mesh(10)
    #     fig = plt.figure()
    #     ndshape.visualize_samples(fig, samples)
    #     plt.show()

    shp = HyperCube(3)
    points = shp.mesh(10)
    projections, metadata = MoebiusStrip.project(points)
    projections_2, metadata = MoebiusStrip.project(projections)
    # fig = plt.figure()
    # visualize._3d_samples(fig, projections, color=np.full(shape=(points.shape[0],), fill_value=1))
    # fig = plt.figure()
    # visualize._3d_samples(fig, projections_2, color=np.full(shape=(points.shape[0],), fill_value=1))
    # plt.show()
    # for x in jnp.abs(projections - projections_2):
    #     print(x > 0.1)
    # print("max err:", jnp.max(jnp.abs(projections - projections_2)))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    root_losses_per_samples = jnp.sqrt(jnp.sum((projections - projections_2) ** 2, axis=-1))
    c = root_losses_per_samples / jnp.max(root_losses_per_samples)
    # Repeat for each body line and two head lines
    c = jnp.concatenate((c, jnp.repeat(c, 2)))
    # Colormap
    c = plt.cm.viridis(c)

    q = ax.quiver(
        projections[:, 0],
        projections[:, 1],
        projections[:, 2],
        projections_2[:, 0] - projections[:, 0] + 0.01,
        projections_2[:, 1] - projections[:, 1] + 0.01,
        projections_2[:, 2] - projections[:, 2] + 0.01,
        colors=c,
    )
    # q.set_array(root_losses_per_samples / jnp.max(root_losses_per_samples))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.show()
