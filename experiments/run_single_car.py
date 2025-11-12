"""Simple runner script for the single car experiment.
Usage:
    cd experiments
    python run_single_car.py
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

experiments_dir = os.path.dirname(os.path.abspath(__file__))
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)

from flow.core.experiment import Experiment
from single_car import flow_params

if __name__ == "__main__":
    exp = Experiment(flow_params)
    
    print("Starting simulation with one car...")
    print("The SUMO GUI should open shortly.")
    exp.run(1)

