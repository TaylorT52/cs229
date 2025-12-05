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
    num_vehicles=20,
    color="0,100,255"
)

vehicles.add(
    veh_id="rl_vehicle",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=2,
    color="255,0,0"
)

additional_net_params = {
    "length": 1000,
    "lanes": 4,
    "speed_limit": 30, 
    "num_edges": 1,
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
        horizon=1500,
    ),

    net=NetParams(
        additional_params=additional_net_params
    ),

    veh=vehicles,
    initial=InitialConfig(
        spacing="uniform",
        perturbation=1.0,
        lanes_distribution=float('inf'), 
        shuffle=False,
    ),
)

