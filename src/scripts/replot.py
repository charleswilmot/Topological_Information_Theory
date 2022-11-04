import logging
from ..core import database
from ..core.experiments import *
from ..core.repetitions import *
import hydra


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="replot")
def main(cfg):
    engine = database.get_engine(cfg.database)
    with orm.Session(engine, future=True) as session:
        query = sql.select(Repetition).join(Experiment).where(Experiment.type == cfg.type)
        for repetition, in session.execute(query).all():
            experiment = repetition.experiment
            experiment.configure(cfg.database, repetition)
            for checkpoint in experiment.list_checkpoints():
                experiment.restore(checkpoint)
                experiment.plot()


if __name__ == '__main__':
    main()
