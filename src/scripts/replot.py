import logging
from ..core import database
from ..core.experiments import *
from ..core.repetitions import *
import omegaconf
import hydra


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="replot")
def main(cfg):
    engine = database.get_engine(cfg.database)
    with orm.Session(engine, future=True) as session:
        query = sql.select(Repetition).join(Repetition.experiment).where(Experiment.type == cfg.type)
        for repetition, in session.execute(query).all():
            experiment = repetition.experiment
            experiment.configure(repetition)
            for checkpoint_path in experiment.list_checkpoints():
                experiment.restore(checkpoint_path)
                experiment.plot()


if __name__ == '__main__':
    main()
