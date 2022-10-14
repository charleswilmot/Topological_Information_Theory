import hydra
import logging
from ..core import experiments


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="reset_default_experiment_conf_files")
def main(cfg):
    experiments.reset_default_experiment_conf_files()


if __name__ == '__main__':
    main()
