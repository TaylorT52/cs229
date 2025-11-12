"""Minimal setup: one car on a simple highway with GUI visualization."""

from flow.controllers import IDMController, RLController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.envs import TestEnv
from flow.networks import HighwayNetwork

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=40,  # 40 human-driven vehicles
    color="0,100,255"  # Blue color (RGB format: "R,G,B")
)

# RL-trained vehicles (red) - use RLController for RL policy control
vehicles.add(
    veh_id="rl_vehicle",
    acceleration_controller=(RLController, {}),  # This marks them as RL vehicles
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=10,  # 10 RL vehicles
    color="255,0,0"  # Red color to distinguish from human-driven
)

# Network parameters - simple straight highway
additional_net_params = {
    "length": 600, #defines highway length
    "lanes": 3, 
    "speed_limit": 30,  # 30 m/s speed limit
    "num_edges": 1,     # single edge
    "use_ghost_edge": False, 
    "ghost_speed_limit": 25,
    "boundary_cell_length": 500
}

flow_params = dict(
    exp_tag='single_car',
    env_name=TestEnv,
    network=HighwayNetwork,
    simulator='traci',

    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    env=EnvParams(
        horizon=1500, # Run for 150 seconds (1500 steps * 0.1s)
    ),

    net=NetParams(
        additional_params=additional_net_params
    ),

    veh=vehicles,
    initial=InitialConfig(),
)

