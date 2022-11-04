#!/usr/bin/env bash
rm -r experiments/debug

rm databases/debug.sqlite

python -m src.scripts.create_tables

python -m src.scripts.reset_default_experiment_conf_files

python -m src.scripts.configure_experiment -m \
  experiment=dimension_collapse \
  experiment.ndshape_name=hypercube2,hypersphere2,hypertorus2,moebiusstrip,kleinbottle \

python -m src.scripts.configure_experiment -m \
  experiment=dimension_project \
  +experiment.ndshape_name=hypercube2,hypersphere2,hypertorus2,moebiusstrip,kleinbottle \
  ++experiment.n_repetitions=4

python -m src.scripts.configure_experiment -m \
  experiment=dimension_project2 \
  +experiment.ndshape_name=hypercube2,hypersphere2,hypertorus2,moebiusstrip,kleinbottle \
  ++experiment.projection_reg_coef=0.01 \
  ++experiment.n_repetitions=4 \
  ++experiment.n_iterations=2000 \
  ++experiment.rotate_every=100 \
  ++experiment.rotate_stop=500 \
  ++experiment.log_start=100
