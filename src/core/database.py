import logging
import sqlalchemy as sql
import sqlalchemy.orm as orm
from dataclasses import dataclass, field as dataclasses_field
import re


# A logger for this file
log = logging.getLogger(__name__)
mapper_registry = orm.registry()
metadata = mapper_registry.metadata


class TableMeta(type):
    def __new__(mcs, name, bases, namespace):
        # this is called when the 'class Hero(YAMLTable):...' is interpreted
        name, bases, namespace = mcs.__before_sql_decorator__(mcs, name, bases, namespace)
        cls = super().__new__(mcs, name, bases, namespace)
        return mapper_registry.mapped(dataclass(cls))

    def __before_sql_decorator__(mcs, name, bases, namespace):
        if '__tablename__' not in namespace:
            namespace['__tablename__'] = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower() + 's'
        if '__sa_dataclass_metadata_key__' not in namespace:
            namespace['__sa_dataclass_metadata_key__'] = 'sa'
        return name, bases, namespace


class ExperimentTableMeta(TableMeta):
    def __before_sql_decorator__(mcs, name, bases, namespace):
        super().__before_sql_decorator__(mcs, name, bases, namespace)
        if '__mapper_args__' not in namespace:
            namespace['__mapper_args__'] = {}
        if '__annotations__' not in namespace:
            namespace['__annotations__'] = {}
        if '__table_args__' not in namespace:
            namespace['__table_args__'] = ()
        if any(map(lambda c: c.name == 'conf_unicity', namespace['__table_args__'])):
            raise ValueError(f"class {name} already has a constraint with name conf_unicity in its __table_args__")

        # configure the polymorphic properties
        namespace['__mapper_args__']['polymorphic_identity'] = name
        namespace['__mapper_args__']['polymorphic_on'] = 'type'

        if not len(bases) or bases[-1] is object: # root experiment class
            # add the 'id' column to the root class
            id_col = sql.Column(sql.Integer, primary_key=True)
            namespace['__annotations__']['id'] = int
            namespace['id'] = field(init=False, sql=id_col)
            # add the 'type' column to the root class
            type_col = sql.Column(sql.String(64))
            namespace['__annotations__']['type'] = str
            namespace['type'] = field(init=False, sql=type_col)
        else:              # some subclass of the root class
            # add the 'id' column to the subclass
            id_col = sql.Column(sql.Integer, sql.ForeignKey(f'{bases[-1].__tablename__}.id', ondelete='cascade'), primary_key=True)
            namespace['__annotations__']['id'] = int
            namespace['id'] = field(init=False, sql=id_col)
        return name, bases, namespace


def field(*args, **kwargs):
    if 'metadata' in kwargs:
        if 'sa' in kwargs['metadata']:
            raise ValueError("Can not use 'sa' as a key in the metadata of this field, this is reserved.")
    if 'sql' in kwargs:
        if 'metadata' not in kwargs:
            kwargs['metadata'] = {}
        kwargs['metadata']['sa'] = kwargs.pop('sql')
    return dataclasses_field(*args, **kwargs)


def get_engine(cfg):
    engine = sql.create_engine(cfg.url, echo=cfg.echo, future=cfg.future)
    connection = engine.connect()
    return engine
