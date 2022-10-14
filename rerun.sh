#!/usr/bin/env bash
rm -r experiments/*
rm databases/debug.sqlite
python -m src.scripts.create_tables
python -m src.scripts.configure_experiment -m \
  experiment=dimension_project \
  experiment.ndshape_name=moebiusstrip \
  experiment.network_depth=2 \
  experiment.n_iterations=5000 \
  experiment.n_repetitions=4
python -m src.scripts.worker
