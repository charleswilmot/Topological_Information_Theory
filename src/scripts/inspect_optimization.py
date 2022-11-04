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


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="inspect_optimization")
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
        experiment.search_rotation_matrix(
            bounds=cfg.bounds,
            maxiter=cfg.maxiter,
            popsize=cfg.popsize,
            mutation=cfg.mutation,
            recombination=cfg.recombination,
            seed=cfg.seed,
            log_plots=True,
            log_plots_name=cfg.log_plots_name,
        )


if __name__ == '__main__':
    main()
