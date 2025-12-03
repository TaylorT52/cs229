"""Flow configuration for the platooning experiment."""

from flow.controllers import IDMController, RLController, ContinuousRouter
from flow.core.params import (
    SumoParams,
    EnvParams,
    InitialConfig,
    NetParams,
    VehicleParams,
)
from flow.networks import HighwayNetwork

from platoon_env import PlatoonEnv

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=40,  # More vehicles for realistic traffic density
    color="0,100,255",
)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=2,
    color="255,0,0",
)

additional_net_params = dict(
    length=2000,  # Longer highway for more realistic flow
    lanes=4,
    speed_limit=30.0,  # Enforce speed limit (30 m/s = ~108 km/h)
    num_edges=1,
    use_ghost_edge=False,
    ghost_speed_limit=30.0,
    boundary_cell_length=500,
)

flow_params = dict(
    exp_tag="platoon_independent",
    env_name=PlatoonEnv,
    network=HighwayNetwork,
    simulator="traci",
    sim=SumoParams(render=False, sim_step=0.1, restart_instance=True),
    env=EnvParams(horizon=1500),
    net=NetParams(additional_params=additional_net_params),
    veh=vehicles,
    initial=InitialConfig(
        spacing="uniform",
        perturbation=1.0,
        lanes_distribution=float("inf"),
        shuffle=False,
    ),
)
