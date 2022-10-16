import hydra
import logging
from ..core import experiments
from ..core import repetitions
from ..core import database
import omegaconf
import sqlalchemy as sql
import sqlalchemy.orm as orm


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="configure_experiment")
def main(cfg):
    engine = database.get_engine(cfg.database)
    experiment = hydra.utils.instantiate(cfg.experiment)
    with orm.Session(engine, future=True) as session:
        log.info('Adding new experiment to the database')
        session.add(experiment)
        session.commit()
        log.info(f'Adding {experiment.n_repetitions} new repetitions to the database')
        for i in range(experiment.n_repetitions):
            query = sql.select(sql.func.count()) \
                .select_from(repetitions.Repetition) \
                .filter(repetitions.Repetition.experiment_id == experiment.id)
            repetition_id = session.execute(query).scalar()
            repetition = repetitions.Repetition(
                experiment_id=experiment.id,
                repetition_id=repetition_id,
                seed=i,
                priority=0 if i else 100
            )
            session.add(repetition)
        session.commit()
    experiments.reset_default_experiment_conf_files()



if __name__ == '__main__':
    main()
