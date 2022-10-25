import sqlalchemy as sql
import sqlalchemy.orm as orm
from sqlalchemy.ext.hybrid import hybrid_property
from .database import field, TableMeta
from . import experiments
import logging


# A logger for this file
log = logging.getLogger(__name__)


class Repetition(metaclass=TableMeta):
    id: int = field(init=False, sql=sql.Column(sql.Integer, primary_key=True))
    experiment_id: int = field(sql=sql.Column(sql.ForeignKey("experiments.id")))
    repetition_id: int = field(sql=sql.Column(sql.Integer))
    seed: int = field(sql=sql.Column(sql.Integer))
    current_step: int = field(default_factory=int, init=False, sql=sql.Column(sql.Integer))
    priority: int = field(default_factory=int, sql=sql.Column(sql.Integer))
    done: bool = field(default_factory=bool, sql=sql.Column(sql.Boolean))
    running: bool = field(default_factory=bool, sql=sql.Column(sql.Boolean))
    experiment: experiments.Experiment = field(init=False, sql=orm.relationship("Experiment"))

    @hybrid_property
    def path(self):
        return f'{self.repetition_id:02d}'

    def is_most_urgent(self, session):
        return self.priority >= get_max_priority(session)


def there_is_work(session):
    query = sql.select(Repetition.id).filter(sql.and_(
        Repetition.done == False,
        Repetition.running == False,
    ))
    res = session.execute(query).all()
    return len(res) > 0


def get_max_priority(session):
    max_priority_query = sql.select(sql.func.max(Repetition.priority)).filter(sql.and_(
        Repetition.done == False,
        Repetition.running == False,
    ))
    max_priority, = session.execute(max_priority_query).first()
    if max_priority is None:
        return 0
    return max_priority


def get_repetition(session):
    max_priority = get_max_priority(session)
    repetitions_query = sql.select(Repetition).filter(sql.and_(
        Repetition.priority == max_priority,
        Repetition.done == False,
        Repetition.running == False,
    )).limit(1)
    repetition = session.execute(repetitions_query).first()
    return repetition[0]
