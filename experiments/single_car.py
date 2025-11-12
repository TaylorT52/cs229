"""Minimal setup: one car on a simple highway with GUI visualization."""

from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs import TestEnv
from flow.networks import HighwayNetwork

vehicles = VehicleParams()
vehicles.add(
    veh_id="car",
    acceleration_controller=(IDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=50
)

# Network parameters - simple straight highway
additional_net_params = {
    "length": 10000, #defines highway length
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

