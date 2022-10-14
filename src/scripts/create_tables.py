from ..core import experiments
from ..core import repetitions
from ..core import database
import hydra
import logging


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="create_tables")
def main(cfg):
    engine = database.get_engine(cfg.database)
    database.metadata.create_all(engine)


if __name__ == '__main__':
    main()
