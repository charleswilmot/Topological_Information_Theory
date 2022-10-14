import logging
from ..core import database
from ..core.experiments import *
from ..core.repetitions import *
import omegaconf
import hydra


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="checkpoint_interactive_plot")
def main(cfg):
    engine = database.get_engine(cfg.database)
    with orm.Session(engine, future=True) as session:
        query = sql.select(Repetition).where(sql.and_(
            Repetition.experiment_id == cfg.experiment_id,
            Repetition.repetition_id == cfg.repetition_id,
        ))
        repetition, = session.execute(query).first()
        experiment = repetition.experiment
        experiment.configure(repetition)
        experiment.restore(f'experiments{experiment.path}{repetition.path}/{cfg.checkpoint:06d}')
        experiment.plot(save=False)


if __name__ == '__main__':
    main()
