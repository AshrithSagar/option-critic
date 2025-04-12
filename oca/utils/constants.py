"""
oca/utils/constants.py \n
Constants, Paths
"""

import os

base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
exps_dir = os.path.join(base_dir, "experiments")
models_dir = os.path.join(exps_dir, "models")
runs_dir = os.path.join(exps_dir, "runs")

for dir_ in [exps_dir, models_dir, runs_dir]:
    os.makedirs(dir_, exist_ok=True)
