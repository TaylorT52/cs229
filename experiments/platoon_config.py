"""Flow configuration for the platooning experiment."""

from flow.controllers import IDMController, RLController, ContinuousRouter, SimLaneChangeController
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
    num_vehicles=8,  # Reduced from 18 for better density
    color="0,100,255",
)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    # Enable RL-controlled lane changes
    lane_change_controller=(SimLaneChangeController, {}),
    num_vehicles=2,
    color="255,0,0",
)

additional_net_params = dict(
    length=1000,
    lanes=4,  # Increased from 2 to 4 lanes
    speed_limit=30.0,
    num_edges=1,
    use_ghost_edge=False,
    ghost_speed_limit=30.0,
    boundary_cell_length=500,
)

# Note: Collision parameters are set via TraCI in the visualization script
# This prevents SUMO from teleporting vehicles on collisions/overlaps
# which corrupts RL state and rewards

# Environment parameters with optional lane change support
env_params = EnvParams(
    horizon=3000,  # Increased from 1500 for longer episodes (better for delayed lane-change rewards)
    additional_params={
        # Enable lane-changing: CTDE policy controls both acceleration and lane changes
        "lane_change_enabled": True,
        # Cooldown between lane changes (seconds) - increased to reduce twitchiness
        "lane_change_duration": 5.0,
    }
)

flow_params = dict(
    exp_tag="platoon_independent",
    env_name=PlatoonEnv,
    network=HighwayNetwork,
    simulator="traci",
    sim=SumoParams(render=False, sim_step=0.1, restart_instance=True),
    env=env_params,
    net=NetParams(additional_params=additional_net_params),
    veh=vehicles,
    initial=InitialConfig(
        spacing="uniform",
        perturbation=2.0,  # Increased from 1.0 to spread vehicles more
        lanes_distribution=float("inf"),  # Allow vehicles to start in any lane
        shuffle=True,  # Shuffle to randomize initial positions
    ),
)
