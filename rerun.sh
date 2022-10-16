#!/usr/bin/env bash
rm -r experiments/*
rm databases/debug.sqlite
python -m src.scripts.create_tables
python -m src.scripts.reset_default_experiment_conf_files
python -m src.scripts.configure_experiment -m \
  experiment=dimension_collapse \
  experiment.ndshape_name=hypercube2,hypersphere2,hypertorus2,moebiusstrip,kleinbottle \
  experiment.network_depth=2 \
  experiment.n_iterations=2 \
  experiment.n_repetitions=1
python -m src.scripts.configure_experiment -m \
  experiment=dimension_project_type1,dimension_project_type2 \
  +experiment.ndshape_name=moebiusstrip \
  ++experiment.network_depth=2 \
  ++experiment.n_iterations=2 \
  ++experiment.n_repetitions=4


# python -m src.scripts.worker
