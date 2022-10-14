import sqlalchemy as sql
import sqlalchemy.orm as orm
from dataclasses import dataclass, MISSING, fields
import yaml
import re
from .database import mapper_registry, field, ExperimentTableMeta
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
from functools import partial

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


class Experiment(metaclass=ExperimentTableMeta):
    id: int = field(init=False, sql=sql.Column(sql.Integer, primary_key=True))
    type: str = field(init=False, sql=sql.Column(sql.String(64)))
    path: str = field(init=False, sql=sql.Column(sql.String(256)))
    n_repetitions: int = field(default=1, sql=sql.Column(sql.Integer))

    __table_args__ = (
        sql.UniqueConstraint("id", name="conf_unicity"),
    )
    __mapper_args__ = {
        "polymorphic_identity": "Experiment",
        "polymorphic_on": "type",
    }

    @classmethod
    def defaults_as_dict(cls):
        return {
            f.name: '???' if f.default is MISSING else f.default
            for f in fields(cls)
            if isinstance(f.metadata[cls.__sa_dataclass_metadata_key__], sql.Column) and f.init
        }

    def exists_in_db(self, session):
        conf_unicity = next(filter(lambda c: c.name == 'conf_unicity', self.__table_args__))
        unique = {column.name: self.__getattribute__(column.name) for column in conf_unicity}
        statement = sql.select(self.__class__.id).filter_by(**unique).limit(1)
        res = session.execute(statement).all()
        return len(res) == 1

    def configure(self, repetition):
        self.current_repetition = repetition
        self.root = f'experiments{self.path}{self.current_repetition.path}'
        log.info(f'resume experiment from {self.root}')
        if not os.path.isdir(self.path) or not os.listdir(self.path):
            os.makedirs(self.path, exist_ok=True)
            self.init()

    def latest_checkpoint(self):
        # search latest checkpoint, restore
        latest = "-1"
        for dirname in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, dirname)) and re.match('[0-9]+', dirname):
                latest = latest if int(dirname) < int(latest) else dirname
        return os.path.join(self.root, latest)

    def list_checkpoints(self):
        res = []
        for dirname in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, dirname)) and re.match('[0-9]+', dirname):
                res.append(os.path.join(self.root, dirname))
        res.sort(key=lambda x: int(x.split('/')[-1]))
        return res


class ExperimentType1(Experiment):
    id: int = field(init=False, sql=sql.Column(sql.ForeignKey('experiments.id'), primary_key=True))
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
    __table_args__ = (
        sql.UniqueConstraint(
            'batch_size',
            'ndshape_name',
            'lr_decay_init_value',
            'lr_decay_transition_steps',
            'lr_decay_decay_rate',
            'lr_decay_transition_begin',
            'lr_decay_staircase',
            'lr_decay_end_value',
            'network_depth',
            'dilation_factor',
            'log_base',
            name='conf_unicity',
    ),)

    def init_infrastructure(self):
        log.info('init_infrastructure')
        self.init_shape()
        self.init_optimizer()
        self.init_tensorboard()
        self.init_networks()

    def init_shape(self):
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

    def init(self):
        log.info(f'init repetition {self.current_repetition.repetition_id}')
        self.iteration = 0
        self.init_infrastructure()
        self.key = random.PRNGKey(self.current_repetition.seed)
        dummy = jnp.zeros(shape=(1, self.embedding_dimension), dtype=jnp.float32)
        self.network_params = self.network.init(self.key, dummy)
        self.learner_state = self.optimizer.init(self.network_params)
        self.checkpoint()

    def restore(self, path):
        log.info(f'restoring {path=}')
        self.iteration = int(path.split('/')[-1])
        self.init_infrastructure()
        with open(f'{path}/key.pkl', 'rb') as f:
            self.key = pickle.load(f)
        with open(f'{path}/network_params.pkl', 'rb') as f:
            self.network_params = pickle.load(f)
        with open(f'{path}/learner_states.pkl', 'rb') as f:
            self.learner_state = pickle.load(f)

    def checkpoint(self):
        log.info(f'checkpointing {self.iteration=}')
        if not os.path.isdir(self.root):
            raise RuntimeError(f"Directory is missing: {self.root}")
        path = f'{self.root}/{self.iteration:06d}'
        os.makedirs(path)
        with open(f'{path}/key.pkl', 'wb') as f:
            pickle.dump(self.key, f)
        with open(f'{path}/network_params.pkl', 'wb') as f:
            pickle.dump(self.network_params, f)
        with open(f'{path}/learner_states.pkl', 'wb') as f:
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
            log.info(f'Iteration {iteration}')
            mse_loss, dloss_dtheta = jax.value_and_grad(self.loss)(self.network_params, samples)
            updates, self.learner_state = self.optimizer.update(dloss_dtheta, self.learner_state)
            self.network_params = optax.apply_updates(self.network_params, updates)
        self.iteration = next_checkpoint_it


class DimensionCollapse(ExperimentType1):
    def init_networks(self):
        self.bottleneck_dimension = self.shp._manifold_dimension
        self.embedding_dimension = self.shp._embedding_dimension

        def forward():
            encoder = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * self.network_depth +
                [self.bottleneck_dimension],
                activation=jnp.tanh,
                activate_final=True,
            )
            decoder = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * (self.network_depth - 1) +
                [self.embedding_dimension],
                activation=jnp.tanh,
            )

            def init(x):
                z = encoder(x)
                xx = decoder(z)
                return xx

            return init, (encoder, decoder)

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

    def plot(self):
        log.info(f'plotting for iteration {self.iteration}')
        samples = self.shp.mesh(100)
        z = self.encode(self.network_params, samples)
        reconstructions = self.decode(self.network_params, z)
        losses_per_samples = jnp.mean((samples - reconstructions) ** 2, axis=-1)
        # fig 1
        fig = plt.figure()
        self.shp.visualize_samples(fig, reconstructions, color=losses_per_samples)
        path = f'{self.root}/reconstruction'
        os.makedirs(path, exist_ok=True)
        fig.savefig(f'{path}/{self.iteration:06d}.png')
        plt.close(fig)
        # fig 2
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, z, color=losses_per_samples)
        path = f'{self.root}/latent'
        os.makedirs(path, exist_ok=True)
        fig.savefig(f'{path}/{self.iteration:06d}.png')
        plt.close(fig)


class DimensionProject(ExperimentType1):
    def init_networks(self):
        self.bottleneck_dimension = self.shp._embedding_dimension
        self.embedding_dimension = self.shp._embedding_dimension

        def forward():
            encoder = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * self.network_depth +
                [self.bottleneck_dimension],
                activation=jnp.tanh,
            )
            decoder = hk.nets.MLP(
                [self.embedding_dimension * self.dilation_factor] * (self.network_depth - 1) +
                [self.embedding_dimension],
                activation=jnp.tanh,
            )

            def init(x):
                z = encoder(x)
                projection, metadata = self.shp.project(z)
                xx = decoder(projection)
                return xx

            return init, (encoder, decoder)

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
            projection, metadata = self.shp.project(z)
            reconstructions = self.decode(network_params, z)
            loss_per_sample = jnp.mean((samples - reconstructions) ** 2, axis=-1)
            loss_per_projection = jnp.mean((projection - z) ** 2, axis=-1)
            return jnp.sum(loss_per_sample) + 10 * jnp.sum(loss_per_projection)

        self.reconstruction_loss_per_sample = reconstruction_loss_per_sample
        self.loss = loss

    def plot(self, save=True):
        def end(fig):
            if save:
                fig.savefig(f'{path}/{self.iteration:06d}.png')
            else:
                plt.show()
            plt.close()

        log.info(f'plotting for iteration {self.iteration}')
        samples = self.shp.mesh(100)
        complement = self.shp.edge(10000)
        samples = jnp.concatenate([samples, complement], axis=0)
        z = self.encode(self.network_params, samples)
        projection, metadata = self.shp.project(z)
        reconstructions = self.decode(self.network_params, z)
        root_losses_per_samples = jnp.sqrt(jnp.mean((samples - reconstructions) ** 2, axis=-1))
        # tmp fig
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        c = root_losses_per_samples[::10] / jnp.max(root_losses_per_samples[::10])
        # Repeat for each body line and two head lines
        c = jnp.concatenate((c, jnp.repeat(c, 2)))
        # Colormap
        c = plt.cm.viridis(c)

        q = ax.quiver(
            reconstructions[::10, 0],
            reconstructions[::10, 1],
            reconstructions[::10, 2],
            samples[::10, 0] - reconstructions[::10, 0],
            samples[::10, 1] - reconstructions[::10, 1],
            samples[::10, 2] - reconstructions[::10, 2],
            colors=c,
        )
        # q.set_array(root_losses_per_samples[::10] / jnp.max(root_losses_per_samples[::10]))
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        path = f'{self.root}/quiver'
        os.makedirs(path, exist_ok=True)
        end(fig)
        # tmp fig
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        q = ax.quiver(
            z[::10, 0],
            z[::10, 1],
            z[::10, 2],
            projection[::10, 0] - z[::10, 0],
            projection[::10, 1] - z[::10, 1],
            projection[::10, 2] - z[::10, 2],
            colors=c,
        )
        # q.set_array(root_losses_per_samples[::10] / jnp.max(root_losses_per_samples[::10]))
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        path = f'{self.root}/quiver2'
        os.makedirs(path, exist_ok=True)
        end(fig)
        # fig 1
        fig = plt.figure()
        self.shp.visualize_samples(fig, reconstructions, color=root_losses_per_samples)
        path = f'{self.root}/reconstruction'
        os.makedirs(path, exist_ok=True)
        end(fig)
        # fig 2
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, z, color=root_losses_per_samples)
        path = f'{self.root}/latent'
        os.makedirs(path, exist_ok=True)
        end(fig)
        # fig 3
        fig = plt.figure()
        nds.NDShapeBase.visualize_samples(fig, projection, color=root_losses_per_samples)
        path = f'{self.root}/projection'
        os.makedirs(path, exist_ok=True)
        end(fig)
