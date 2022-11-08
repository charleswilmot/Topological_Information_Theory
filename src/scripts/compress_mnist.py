import logging
from ..core import ndshape as nds
from ..core.rotation import *
import hydra
import jax
from jax import random
import jax.numpy as jnp
import tensorflow_datasets as tfds
import os
from tensorboardX import SummaryWriter
import haiku as hk
from scipy.optimize import differential_evolution
import optax
import re
import numpy as np
import matplotlib.pyplot as plt


# A logger for this file
log = logging.getLogger(__name__)


def end(fig, save, filename):
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


class Model:
    def __init__(self, shp, network_depth, network_width, rotate_every, rotate_stop, lr, alpha, differential_evolution, barycenter_reg_coef, projection_reg_coef, seed):
        self.shp = shp
        self.network_depth = network_depth
        self.network_width = network_width
        self.bottleneck_dimension = self.shp._embedding_dimension
        self.embedding_dimension = 28 * 28
        self.key = random.PRNGKey(seed)
        self.barycenter_estimate = jnp.mean(self.shp.sample(self.key, 2 ** 11), axis=0)
        self.rotate_every = rotate_every
        self.rotate_stop = rotate_stop
        self.root = os.path.join('experiments', 'CompressMNIST')
        os.makedirs(self.root, exist_ok=True)
        count = sum(1 for f in os.listdir(self.root) if re.match("[0-9]+", f))
        self.root = os.path.join(self.root, f'{count:04d}')
        os.makedirs(self.root, exist_ok=True)
        tensorboard_logdir = os.path.join(self.root, 'tensorboard', self.shp._name)
        self.tensorboard = SummaryWriter(logdir=tensorboard_logdir)
        self.iteration = 0
        self.lr = lr
        self.alpha = alpha
        self.differential_evolution = differential_evolution
        self.barycenter_reg_coef = barycenter_reg_coef
        self.projection_reg_coef = projection_reg_coef

        def forward():
            encode = hk.nets.MLP(
                [self.network_width] * self.network_depth +
                [self.bottleneck_dimension],
                activation=jnp.tanh,
                name='encode',
            )
            rotate = hk.Linear(self.bottleneck_dimension, with_bias=False, name="rotate")
            decode = hk.nets.MLP(
                [self.network_width] * (self.network_depth - 1) +
                [self.embedding_dimension],
                activation=jnp.tanh,
                name='decode',
            )

            def init(x):
                z = encode(x)
                r = rotate(z)
                reconstruction = decode(z)
                return reconstruction

            return init, (encode, rotate, decode)

        self.network = hk.without_apply_rng(hk.multi_transform(forward))
        self.encode, self.rotate, self.decode = self.network.apply
        self.unrotate = lambda network_params, inp: inp @ jnp.linalg.inv(network_params["rotate"]["w"])

        @jax.jit
        def reconstruction_loss_per_sample(network_params, samples):
            z = self.encode(network_params, samples)
            r = self.rotate(network_params, z) + self.barycenter_estimate
            p, _ = self.shp.project(r)
            u = self.unrotate(network_params, p - self.barycenter_estimate)
            reconstructions = self.decode(network_params, u)
            return jnp.mean((samples - reconstructions) ** 2, axis=-1)

        @jax.jit
        def reconstruction_loss(trainable_params, non_trainable_params, alpha, samples):
            network_params = hk.data_structures.merge(trainable_params, non_trainable_params)
            z = self.encode(network_params, samples)
            r = self.rotate(network_params, z) + self.barycenter_estimate
            p, _ = self.shp.project(r)
            u = self.unrotate(network_params, p - self.barycenter_estimate)
            blend = alpha * z + (1 - alpha) * u
            reconstructions = self.decode(network_params, blend)
            reconstruction_term = jnp.sum(jnp.mean((samples - reconstructions) ** 2, axis=-1))
            return reconstruction_term

        @jax.jit
        def barycenter_loss(trainable_params, non_trainable_params, samples):
            network_params = hk.data_structures.merge(trainable_params, non_trainable_params)
            z = self.encode(network_params, samples)
            barycenter = jnp.mean(z, axis=0)
            loss = jnp.sum(barycenter ** 2)
            return loss * self.barycenter_reg_coef

        @jax.jit
        def reduce_loss(trainable_params, non_trainable_params, samples):
            network_params = hk.data_structures.merge(trainable_params, non_trainable_params)
            z = self.encode(network_params, samples)
            r = self.rotate(network_params, z) + self.barycenter_estimate
            p, _ = self.shp.project(r)
            projection_term = jnp.sum(jnp.mean((r - p) ** 2, axis=-1))
            return projection_term * self.projection_reg_coef

        @jax.jit
        def total_loss(trainable_params, non_trainable_params, alpha, samples):
            return (
                reconstruction_loss(trainable_params, non_trainable_params, alpha, samples) +
                reduce_loss(trainable_params, non_trainable_params, samples)
            )

        @jax.jit
        def gradients(network_params, alpha, samples):
            rotation_grad = {'rotate': {"w": jnp.zeros_like(network_params['rotate']['w'])}}

            barycenter_params, non_trainable_params = hk.data_structures.partition(lambda m, n, v:
                m.startswith("encode") and m.endswith(f"_{self.network_depth - 1}") and n == 'b',
                network_params,
            )
            barycenter_grad = jax.grad(barycenter_loss)(barycenter_params, non_trainable_params, samples)

            reconstruction_params, non_trainable_params = hk.data_structures.partition(lambda m, n, v:
                not m.startswith("rotate") and
                not (m.startswith("encode") and m.endswith(f"_{self.network_depth - 1}") and n == 'b'),
                network_params,
            )
            reconstruction_grad = jax.grad(total_loss)(reconstruction_params, non_trainable_params, alpha, samples)

            network_grad = hk.data_structures.merge(rotation_grad, barycenter_grad, reconstruction_grad)
            return network_grad

        self.reconstruction_loss_per_sample = reconstruction_loss_per_sample
        self.gradients = gradients
        self.optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=optax.exponential_decay(
                init_value=self.lr['decay_init_value'],
                transition_steps=self.lr['decay_transition_steps'],
                decay_rate=self.lr['decay_decay_rate'],
                transition_begin=self.lr['decay_transition_begin'],
                staircase=self.lr['decay_staircase'],
                end_value=self.lr['decay_end_value'],
            )
        )
        self.alpha = optax.exponential_decay(
            init_value=self.alpha['decay_init_value'],
            transition_steps=self.alpha['decay_transition_steps'],
            decay_rate=self.alpha['decay_decay_rate'],
            transition_begin=self.alpha['decay_transition_begin'],
            staircase=self.alpha['decay_staircase'],
            end_value=self.alpha['decay_end_value'],
        )

        dummy = jnp.zeros(shape=(1, self.embedding_dimension), dtype=jnp.float32)
        self.network_params = self.network.init(self.key, dummy)
        self.learner_state = self.optimizer.init(self.network_params)

    def train(self, samples):
        if self.iteration and self.iteration % self.rotate_every == 0 and self.iteration <= self.rotate_stop:
            log.info(f"Searching best rotation matrix (iteration = {self.iteration})")
            rot = self.search_rotation_matrix(samples)
            self.network_params["rotate"]["w"] = rot
        dloss_dtheta = self.gradients(self.network_params, self.alpha(self.iteration), samples)
        updates, self.learner_state = self.optimizer.update(dloss_dtheta, self.learner_state)
        self.network_params = optax.apply_updates(self.network_params, updates)
        self.iteration += 1

    def search_rotation_matrix(self, samples):
        bounds = self.differential_evolution['bounds']
        tol = self.differential_evolution['tol']
        maxiter = self.differential_evolution['maxiter']
        popsize = self.differential_evolution['popsize']
        mutation = self.differential_evolution['mutation']
        recombination = self.differential_evolution['recombination']
        seed = 1
        z = self.encode(self.network_params, samples)
        n_angles = self.bottleneck_dimension * (self.bottleneck_dimension - 1) // 2
        x0 = jnp.zeros(shape=(n_angles,))
        res = differential_evolution(
            func=self.minimization_target,
            x0=x0,
            args=(z,),
            disp=True,
            bounds=[(-bounds, bounds)] * n_angles,
            vectorized=True,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            seed=seed,
            recombination=recombination,
            maxiter=maxiter,
        )
        log.info(f'angles={res.x}')
        log.info(f'final MRSE={res.fun}')
        log.info(f'{res.nfev=}')
        log.info(f'{res.nit=}')
        log.info(f'{res.success=}')
        return rotation_matrix(res.x, self.bottleneck_dimension)

    def minimization_target(self, angles, z):
        if angles.ndim == 1:
            angles = angles[..., None]
        angles = angles.T
        rot = v_rotation_matrix(angles, self.bottleneck_dimension)
        rotated = z @ rot + self.barycenter_estimate
        projection, _ = self.shp.project(rotated)
        dist = jnp.mean(jnp.sqrt(jnp.sum((projection - rotated) ** 2, axis=-1)), axis=-1)
        return dist

    def log(self, samples):
        ### log to tensorboard
        log.info(f'logging @ {self.iteration}')
        mse_loss_per_sample = self.reconstruction_loss_per_sample(self.network_params, samples)
        mean = np.asarray(jnp.mean(jnp.sqrt(mse_loss_per_sample)))
        maxi = np.asarray(jnp.sqrt(jnp.max(mse_loss_per_sample)))
        mini = np.asarray(jnp.sqrt(jnp.min(mse_loss_per_sample)))
        self.tensorboard.add_scalar('batch_MeanRSE', mean, self.iteration)
        self.tensorboard.add_scalar('batch_MaxRSE', maxi, self.iteration)
        self.tensorboard.add_scalar('batch_MinRSE', mini, self.iteration)
        self.tensorboard.add_scalar('learning_rate', self.learner_state.hyperparams['learning_rate'], self.iteration)
        log.info(f'batch_MeanRSE: {mean}')
        log.info(f'batch_MaxRSE : {maxi}')

    def plot(self, samples):
        log.info(f'plotting @ {self.iteration}')

    def plot(self, samples, save=True):
        log.info(f'plotting for iteration {self.iteration}')
        z = self.encode(self.network_params, samples)
        r = self.rotate(self.network_params, z) + self.barycenter_estimate
        p, _ = self.shp.project(r)
        u = self.unrotate(self.network_params, p - self.barycenter_estimate)
        reconstructions = self.decode(self.network_params, u)
        root_losses_per_samples = jnp.sqrt(jnp.mean((samples - reconstructions) ** 2, axis=-1))
        # fig 1
        for i, (spl, rec) in enumerate(zip(samples[:10], reconstructions[:10])):
            fig = plt.figure()
            spl = jnp.reshape(spl, (28, 28))
            rec = jnp.reshape(rec, (28, 28))
            ax = fig.add_subplot(111)
            ax.imshow(jnp.concatenate([spl, rec], axis=1))
            end(fig, save, os.path.join(self.root, 'reconstruction', f'{self.iteration:06d}_{i:02d}.png'))
        # fig 2
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, z, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'latent', f'{self.iteration:06d}.png'))
        # fig 3
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, r, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'rotated', f'{self.iteration:06d}.png'))
        # fig 3 bis
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, u, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'unrotated', f'{self.iteration:06d}.png'))
        # fig 4
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, p, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'projection', f'{self.iteration:06d}.png'))


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="compress_mnist")
def main(cfg):
    train_dataset, = tfds.load(
        'mnist',
        split=['train'],
        as_supervised=True,
        with_info=False,
    )


    def normalize(images):
        images = jnp.float32(images) / (255. / 2) - 1.
        return jnp.reshape(images, (-1, 28 * 28))

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.repeat(1000)
    train_dataset = train_dataset.shuffle(5000)
    train_dataset = train_dataset.batch(cfg.batch_size)
    # train_dataset = train_dataset.as_numpy_iterator()

    model = Model(
        shp=nds.NDShapeBase.by_name(cfg.ndshape_name),
        network_depth=cfg.network_depth,
        network_width=cfg.network_width,
        rotate_every=cfg.rotate_every,
        rotate_stop=cfg.rotate_stop,
        lr=cfg.lr,
        alpha=cfg.alpha,
        differential_evolution=cfg.differential_evolution,
        barycenter_reg_coef=cfg.barycenter_reg_coef,
        projection_reg_coef=cfg.projection_reg_coef,
        seed=cfg.seed,
    )

    for i, (batch, _) in enumerate(train_dataset):
        batch = normalize(batch)
        if i % 100 == 0:
            model.log(batch)
            model.plot(batch)
        model.train(batch)


if __name__ == '__main__':
    main()
