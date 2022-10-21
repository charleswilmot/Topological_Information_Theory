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
        if os.path.isdir(experiment.path):
            shutil.rmtree(experiment.path)
        table = experiment.__table__
        query = sql.delete(table).where(table.id == cfg.experiment_id)
        session.execute(query)
        query = sql.delete(Repetition).where(Repetition.experiment_id == cfg.experiment_id)
        session.execute(query)
        session.commit()


if __name__ == '__main__':
    main()
