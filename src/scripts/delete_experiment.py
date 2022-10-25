from genericpath import isdir
import logging
from ..core import database
from ..core.experiments import *
from ..core.repetitions import *
import hydra
import shutil
import os


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="delete_experiment")
def main(cfg):
    engine = database.get_engine(cfg.database)
    with orm.Session(engine, future=True) as session:
        query = sql.select(Experiment).where(Experiment.id == cfg.experiment_id)
        experiment = session.execute(query).scalar()
        if experiment is None:
            raise ValueError(f"No experiment with id {cfg.experiment_id}")

        database_name = cfg.database.url.split('/')[-1].rstrip('.sqlite')
        experiment_root = os.path.join('experiments', database_name, experiment.path)
        if os.path.isdir(experiment_root):
            shutil.rmtree(experiment_root)

        cls = experiment.__class__
        while cls != object:
            query = sql.delete(cls.__table__).where(cls.id == cfg.experiment_id)
            session.execute(query)
            cls = cls.__base__

        query = sql.delete(Repetition).where(Repetition.experiment_id == cfg.experiment_id)
        session.execute(query)
        session.commit()


if __name__ == '__main__':
    main()
