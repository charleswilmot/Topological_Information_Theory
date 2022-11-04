import logging
from ..core import database
from ..core.experiments import *
from ..core.repetitions import *
import hydra
import os
import jax.numpy as jnp
from jax import random


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="dump_latent")
def main(cfg):
    engine = database.get_engine(cfg.database)
    with orm.Session(engine, future=True) as session:
        query = sql.select(Repetition).where(sql.and_(
            Repetition.experiment_id == cfg.experiment_id,
            Repetition.repetition_id == cfg.repetition_id,
        ))
        repetition, = session.execute(query).first()
        experiment = repetition.experiment
        experiment.configure(cfg.database, repetition)
        experiment.restore(cfg.checkpoint)

        # script starts here...
        if cfg.mesh:
            data = experiment.shp.mesh(cfg.mesh_size)
        else:
            key = random.PRNGKey(0)
            data = experiment.shp.sample(key, cfg.sample_size)
        latents = experiment.encode(experiment.network_params, data)
        dirname = os.path.join(experiment.root, 'latent_dumps')
        filename = os.path.join(dirname, f'{cfg.checkpoint:06d}.npy')
        os.makedirs(dirname, exist_ok=True)
        jnp.save(filename, latents)



if __name__ == '__main__':
    main()
