import logging
from ..core import database
from ..core.experiments import *
from ..core.repetitions import *
import omegaconf
import hydra


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="worker")
def main(cfg):
    engine = database.get_engine(cfg.database)
    with orm.Session(engine, future=True) as session:
        while there_is_work(session):
            repetition = get_repetition(session)
            repetition.running = True
            session.commit()
            experiment = repetition.experiment
            try:
                experiment.configure(repetition)
                checkpoint_path = experiment.latest_checkpoint()
                experiment.restore(checkpoint_path)
                while repetition.is_most_urgent(session) and not experiment.finished():
                    experiment.log()
                    experiment.to_next_checkpoint()
                    experiment.checkpoint()
                    repetition.current_step = experiment.iteration
                    session.commit()
                if experiment.finished():
                    repetition.done = True
                    session.commit()
                    experiment.log()
            except:
                log.critical('An exception has been raised...')
                raise
            finally:
                repetition.running = False
                session.commit()


if __name__ == '__main__':
    main()
