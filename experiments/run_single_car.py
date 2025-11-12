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
    # Check for --no_render flag
    import sys
    if "--no_render" in sys.argv:
        flow_params['sim'].render = False
        print("Running without GUI (headless mode)")
    else:
        print("Starting simulation with GUI...")
        print("The SUMO GUI should open shortly.")
    
    exp = Experiment(flow_params)
    print(f"Simulating with {flow_params['veh'].num_vehicles} total vehicles")
    print(f"- {flow_params['veh'].num_rl_vehicles} RL vehicles (red)")
    print(f"- {flow_params['veh'].num_vehicles - flow_params['veh'].num_rl_vehicles} human vehicles (blue)")
    exp.run(1)

