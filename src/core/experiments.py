import sqlalchemy as sql
import sqlalchemy.orm as orm
from sqlalchemy.ext.hybrid import hybrid_property
from dataclasses import MISSING, fields
import yaml
import re
from .database import field, ExperimentTableMeta
from . import ndshape as nds
import logging
import sys, inspect
import os
import shutil
from tensorboardX import SummaryWriter
import jax
from jax import random
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib.pyplot as plt
import pickle


# A logger for this file
log = logging.getLogger(__name__)


def logarithmic_freq(iteration, base=10):
    if iteration == 0:
        return True
    smallest_exponent = get_smallest_exponent(iteration, base=base)
    return iteration % (base ** smallest_exponent) == 0


def get_smallest_exponent(iteration, base=10):
    smallest_exponent = 0
    while base ** smallest_exponent <= iteration: smallest_exponent += 1
    return smallest_exponent - 1


def get_next_checkpoint_it(iteration, base=10):
    if iteration == 0:
        return 1
    smallest_exponent = get_smallest_exponent(iteration, base=base)
    smallest_pow = base ** smallest_exponent
    below = smallest_pow * (iteration // smallest_pow)
    above = below + smallest_pow
    return  above


def reset_default_experiment_conf_files():
    module = sys.modules["src.core.experiments"]
    predicate = lambda cls: inspect.isclass(cls) and issubclass(cls, Experiment) and cls is not Experiment
    experiment_classes = inspect.getmembers(module, predicate)
    for name, cls in experiment_classes:
        log.info(f'Checking config file for experiment {name}')
        filename = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        filepath = f"./conf/experiment/{filename}.yaml"
        filepath_backup = f"./conf/experiment/.{filename}.backup.yaml"
        default_conf = cls.defaults_as_dict()
        default_conf['_target_'] = cls.__module__ + '.' + cls.__name__
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                previous_conf = yaml.load(f, Loader=yaml.SafeLoader)
            same = True
            same = same and (set(previous_conf.keys()) == set(default_conf.keys()))
            same = same and all(default_conf[key] == value for key, value in previous_conf.items())
            if same:
                log.info(f'experiment config file {filename}.yaml has not changed')
                continue
            # copy previous file -> backup
            shutil.copy(filepath, filepath_backup)
            log.info(f'experiment config file {filename}.yaml has changed, making a backup (.{filename}.yaml)')
        # overwrite / create file with default
        with open(filepath, 'w') as f:
            f.write(yaml.dump(default_conf))
            log.info(f'experiment config file {filename}.yaml overwritten')


def end(fig, save, filename):
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


class Experiment(metaclass=ExperimentTableMeta):
    n_repetitions: int = field(default=1, sql=sql.Column(sql.Integer))

    @hybrid_property
    def path(self):
        return os.path.join(self.type, f'{self.id:04d}')

    @classmethod
    def defaults_as_dict(cls):
        return {
            f.name: '???' if f.default is MISSING else f.default
            for f in fields(cls)
            if isinstance(f.metadata[cls.__sa_dataclass_metadata_key__], sql.Column) and f.init
        }

    def configure(self, database, repetition):
        database_name = database.url.split('/')[-1].rstrip('.sqlite')
        database_root = os.path.join('experiments', database_name)
        experiment_root = os.path.join('experiments', database_name, self.path)
        repetition_root = os.path.join('experiments', database_name, self.path, repetition.path)
        self.root = repetition_root
        must_init = not os.path.isdir(self.root)
        log.info(f'resume experiment from {experiment_root}')
        os.makedirs(database_root, exist_ok=True)
        os.makedirs(experiment_root, exist_ok=True)
        os.makedirs(repetition_root, exist_ok=True)
        if must_init: self.init(repetition)

    def latest_checkpoint(self):
        # search latest checkpoint, restore
        return int(min((x for x in os.listdir(self.root) if re.match('[0-9]+', x)), key=int))

    def list_checkpoints(self):
        return list(int(x) for x in sorted(os.listdir(self.root), key=int) if re.match('[0-9]+', x))


class ExperimentType1(Experiment):
    ndshape_name: str = field(kw_only=True, sql=sql.Column(sql.String(128)))
    batch_size: int = field(default=1024, sql=sql.Column(sql.Integer))
    lr_decay_init_value: float = field(default=2e-3, sql=sql.Column(sql.Float(precision=8)))
    lr_decay_transition_steps: int = field(default=1000, sql=sql.Column(sql.Integer))
    lr_decay_decay_rate: float = field(default=0.5, sql=sql.Column(sql.Float(precision=8)))
    lr_decay_transition_begin: int = field(default=1000, sql=sql.Column(sql.Integer))
    lr_decay_staircase: bool = field(default=False, sql=sql.Column(sql.Boolean()))
    lr_decay_end_value: float = field(default=1e-5, sql=sql.Column(sql.Float(precision=8)))
    n_iterations: int = field(default=5000, sql=sql.Column(sql.Integer))
    network_depth: int = field(default=2, sql=sql.Column(sql.Integer))
    dilation_factor: int = field(default=10, sql=sql.Column(sql.Integer))
    log_base: int = field(default=10, sql=sql.Column(sql.Integer))

    def init_infrastructure(self):
        log.info('init_infrastructure')
        self.init_ndshape()
        self.init_optimizer()
        self.init_tensorboard()
        self.init_networks()

    def init_ndshape(self):
        self.shp = nds.NDShapeBase.by_name(self.ndshape_name)

    def init_optimizer(self):
        self.optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=optax.exponential_decay(
                init_value=self.lr_decay_init_value,
                transition_steps=self.lr_decay_transition_steps,
                decay_rate=self.lr_decay_decay_rate,
                transition_begin=self.lr_decay_transition_begin,
                staircase=self.lr_decay_staircase,
                end_value=self.lr_decay_end_value,
            )
        )

    def init_tensorboard(self):
        self.tensorboard = SummaryWriter(logdir=f'{self.root}/tensorboard/{self.shp._name}')

    def init(self, repetition):
        log.info(f'init repetition {repetition.repetition_id}')
        self.iteration = 0
        self.init_infrastructure()
        self.key = random.PRNGKey(repetition.seed)
        dummy = jnp.zeros(shape=(1, self.embedding_dimension), dtype=jnp.float32)
        self.network_params = self.network.init(self.key, dummy)
        self.learner_state = self.optimizer.init(self.network_params)
        self.checkpoint()

    def restore(self, checkpoint):
        log.info(f'restoring {checkpoint=}')
        checkpoint_str = checkpoint if isinstance(checkpoint, str) else f'{checkpoint:06d}'
        path = os.path.join(self.root, checkpoint_str)
        self.iteration = int(checkpoint)
        self.init_infrastructure()
        with open(os.path.join(path, 'key.pkl'), 'rb') as f:
            self.key = pickle.load(f)
        with open(os.path.join(path, 'network_params.pkl'), 'rb') as f:
            self.network_params = pickle.load(f)
        with open(os.path.join(path, 'learner_states.pkl'), 'rb') as f:
            self.learner_state = pickle.load(f)

    def checkpoint(self):
        log.info(f'checkpointing {self.iteration=}')
        if not os.path.isdir(self.root):
            raise RuntimeError(f"Directory is missing: {self.root}")
        path = os.path.join(self.root, f'{self.iteration:06d}')
        os.makedirs(path)
        with open(os.path.join(path, 'key.pkl'), 'wb') as f:
            pickle.dump(self.key, f)
        with open(os.path.join(path, 'network_params.pkl'), 'wb') as f:
            pickle.dump(self.network_params, f)
        with open(os.path.join(path, 'learner_states.pkl'), 'wb') as f:
            pickle.dump(self.learner_state, f)

    def finished(self):
        return self.iteration == self.n_iterations

    def log(self):
        ### log to tensorboard
        log.info('logging data')
        samples = self.shp.sample(self.key, 1024)
        mse_loss_per_sample = self.reconstruction_loss_per_sample(self.network_params, samples)
        self.tensorboard.add_scalar('batch_MeanSE', jnp.mean(mse_loss_per_sample), self.iteration)
        self.tensorboard.add_scalar('batch_MaxRSE', jnp.sqrt(jnp.max(mse_loss_per_sample)), self.iteration)
        self.tensorboard.add_scalar('batch_MinSE', jnp.min(mse_loss_per_sample), self.iteration)
        self.tensorboard.add_scalar('learning_rate', self.learner_state.hyperparams['learning_rate'], self.iteration)
        ### generate plots
        self.plot()

    def to_next_checkpoint(self):
        next_checkpoint_it = get_next_checkpoint_it(self.iteration, base=self.log_base)
        log.info(f'training: {self.iteration} --> {next_checkpoint_it}')
        next_checkpoint_it = get_next_checkpoint_it(self.iteration, base=self.log_base)
        for iteration in range(self.iteration, next_checkpoint_it):
            self.key, = random.split(self.key, 1)
            samples = self.shp.sample(self.key, self.batch_size)
            log.debug(f'Iteration {iteration}')
            dloss_dtheta = jax.grad(self.loss)(self.network_params, samples)
            updates, self.learner_state = self.optimizer.update(dloss_dtheta, self.learner_state)
            self.network_params = optax.apply_updates(self.network_params, updates)
        self.iteration = next_checkpoint_it


class DimensionCollapse(ExperimentType1):
    def init_networks(self):
        self.bottleneck_dimension = self.shp._manifold_dimension
        self.embedding_dimension = self.shp._embedding_dimension

        def forward():
            encode = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * self.network_depth +
                [self.bottleneck_dimension],
                activation=jnp.tanh,
                activate_final=True,
            )
            decode = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * (self.network_depth - 1) +
                [self.embedding_dimension],
                activation=jnp.tanh,
            )

            def init(x):
                z = encode(x)
                xx = decode(z)
                return xx

            return init, (encode, decode)

        self.network = hk.without_apply_rng(hk.multi_transform(forward))
        self.encode, self.decode = self.network.apply

        @jax.jit
        def reconstruction_loss_per_sample(network_params, samples):
            z = self.encode(network_params, samples)
            reconstructions = self.decode(network_params, z)
            return jnp.mean((samples - reconstructions) ** 2, axis=-1)

        @jax.jit
        def loss(network_params, samples):
            return jnp.sum(self.reconstruction_loss_per_sample(network_params, samples))

        self.reconstruction_loss_per_sample = reconstruction_loss_per_sample
        self.loss = loss

    def plot(self, save=True):
        log.info(f'plotting for iteration {self.iteration}')
        samples = self.shp.mesh(100)
        z = self.encode(self.network_params, samples)
        reconstructions = self.decode(self.network_params, z)
        root_losses_per_samples = jnp.sqrt(jnp.mean((samples - reconstructions) ** 2, axis=-1))
        # fig 1
        fig = plt.figure()
        self.shp.visualize_samples(fig, reconstructions, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'reconstruction', f'{self.iteration:06d}.png'))
        # fig 2
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, z, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'latent', f'{self.iteration:06d}.png'))


class DimensionProject(ExperimentType1):
    projection_reg_coef: float = field(default=1, sql=sql.Column(sql.Float(precision=8)))
    shape_reg_coef: float = field(default=10, sql=sql.Column(sql.Float(precision=8)))

    def init_networks(self):
        self.bottleneck_dimension = self.shp._embedding_dimension
        self.embedding_dimension = self.shp._embedding_dimension

        def forward():
            encode = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * self.network_depth +
                [self.bottleneck_dimension],
                activation=jnp.tanh,
            )
            decode = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * (self.network_depth - 1) +
                [self.embedding_dimension],
                activation=jnp.tanh,
            )

            def init(x):
                z = encode(x)
                projection, metadata = self.shp.project(z)
                xx = decode(z)
                return xx

            return init, (encode, decode)

        self.network = hk.without_apply_rng(hk.multi_transform(forward))
        self.encode, self.decode = self.network.apply

        @jax.jit
        def reconstruction_loss_per_sample(network_params, samples):
            z = self.encode(network_params, samples)
            projection, metadata = self.shp.project(z)
            reconstructions = self.decode(network_params, z)
            return jnp.mean((samples - reconstructions) ** 2, axis=-1)

        @jax.jit
        def loss(network_params, samples):
            z = self.encode(network_params, samples)
            reconstructions = self.decode(network_params, z)
            reconstruction_term = jnp.sum(jnp.mean((samples - reconstructions) ** 2, axis=-1))
            regularization_term = self.shp.regularize(z, self.projection_reg_coef, self.shape_reg_coef)
            return reconstruction_term + regularization_term

        self.reconstruction_loss_per_sample = reconstruction_loss_per_sample
        self.loss = loss

    def plot(self, save=True):
        log.info(f'plotting for iteration {self.iteration}')
        samples = self.shp.mesh(100)
        z = self.encode(self.network_params, samples)
        projection, metadata = self.shp.project(z)
        reconstructions = self.decode(self.network_params, z)
        root_losses_per_samples = jnp.sqrt(jnp.mean((samples - reconstructions) ** 2, axis=-1))
        # fig 1
        fig = plt.figure()
        self.shp.visualize_samples(fig, reconstructions, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'reconstruction', f'{self.iteration:06d}.png'))
        # fig 2
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, z, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'latent', f'{self.iteration:06d}.png'))
        # fig 3
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, projection, color=root_losses_per_samples)
        end(fig, save, os.path.join(self.root, 'projection', f'{self.iteration:06d}.png'))

