import logging
from re import template
import hydra


# A logger for this file
log = logging.getLogger(__name__)


python_template = """import logging
from ..core import database
from ..core.experiments import *
from ..core.repetitions import *
import hydra


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="{script_name}")
def main(cfg):
    engine = database.get_engine(cfg.database)
    with orm.Session(engine, future=True) as session:
        pass

if __name__ == '__main__':
    main()
"""


yaml_template = """

defaults:
  - hydra: default
"""


@hydra.main(version_base="1.2", config_path="../../conf/", config_name="new_script")
def main(cfg):
    script_name = cfg.script_name
    with open(f"src/scripts/{script_name}.py", "w") as f:
        f.write(python_template.format(script_name=script_name))

    with open(f"conf/{script_name}.yaml", "w") as f:
        for arg in cfg.script_args:
            f.write(f"{arg}: ???\n")
        f.write(yaml_template)


if __name__ == '__main__':
    main()
