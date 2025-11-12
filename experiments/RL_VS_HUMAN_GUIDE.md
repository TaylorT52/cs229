# Guide: Distinguishing RL vs Human-Driven Vehicles in Flow

This guide explains how to set up and distinguish between RL-trained vehicles and human-driven vehicles in your Flow simulation.

## Key Concepts

1. **RL Vehicles**: Use `RLController` as the acceleration controller
2. **Human Vehicles**: Use car-following controllers like `IDMController`, `SimCarFollowingController`, etc.
3. **Visual Distinction**: Set different colors for each vehicle type
4. **Programmatic Access**: Use `get_rl_ids()` and `get_human_ids()` methods

## Setup

### 1. Define Vehicle Types with Different Controllers

```python
from flow.controllers import IDMController, RLController, ContinuousRouter
from flow.core.params import VehicleParams

vehicles = VehicleParams()

# Human-driven vehicles (blue)
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),  # Human controller
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=40,
    color="0,100,255"  # Blue (RGB format: "R,G,B")
)

# RL-trained vehicles (red)
vehicles.add(
    veh_id="rl_vehicle",
    acceleration_controller=(RLController, {}),  # RL controller - THIS IS KEY!
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=10,
    color="255,0,0"  # Red
)
```

### 2. Access Vehicle Types in Your Environment

```python
from flow.envs.base import Env

class MyEnv(Env):
    def additional_command(self):
        # Get RL vehicle IDs
        rl_ids = self.k.vehicle.get_rl_ids()
        print(f"RL vehicles: {rl_ids}")
        
        # Get human vehicle IDs
        human_ids = self.k.vehicle.get_human_ids()
        print(f"Human vehicles: {human_ids}")
        
        # Apply actions only to RL vehicles
        if len(rl_ids) > 0:
            # Your RL policy actions here
            actions = ...  # Get from your RL policy
            self.k.vehicle.apply_acceleration(rl_ids, actions)
        
        # Human vehicles are controlled automatically by their controllers
```

## Visual Distinction

### In Configuration (Recommended)

Set colors when defining vehicles:

```python
vehicles.add(
    veh_id="human",
    color="0,100,255",  # Blue
    ...
)

vehicles.add(
    veh_id="rl_vehicle",
    color="255,0,0",  # Red
    ...
)
```

### Dynamically in Environment

Change colors during simulation:

```python
def additional_command(self):
    # Set RL vehicles to red
    for rl_id in self.k.vehicle.get_rl_ids():
        self.k.kernel_api.vehicle.setColor(rl_id, (255, 0, 0, 255))  # Red
    
    # Set human vehicles to blue
    for human_id in self.k.vehicle.get_human_ids():
        self.k.kernel_api.vehicle.setColor(human_id, (0, 100, 255, 255))  # Blue
```

## Common Use Cases

### 1. Track Statistics Separately

```python
def additional_command(self):
    rl_ids = self.k.vehicle.get_rl_ids()
    human_ids = self.k.vehicle.get_human_ids()
    
    # Average speed of RL vehicles
    if len(rl_ids) > 0:
        rl_speeds = [self.k.vehicle.get_speed(vid) for vid in rl_ids]
        avg_rl_speed = np.mean(rl_speeds)
        print(f"RL avg speed: {avg_rl_speed:.2f} m/s")
    
    # Average speed of human vehicles
    if len(human_ids) > 0:
        human_speeds = [self.k.vehicle.get_speed(vid) for vid in human_ids]
        avg_human_speed = np.mean(human_speeds)
        print(f"Human avg speed: {avg_human_speed:.2f} m/s")
```

### 2. Apply RL Actions Only to RL Vehicles

```python
def _apply_rl_actions(self, rl_actions):
    """Apply actions only to RL vehicles."""
    if rl_actions is None:
        return
    
    rl_ids = sorted(self.k.vehicle.get_rl_ids())
    if len(rl_ids) > 0:
        # Clip actions to valid range
        clipped_actions = np.clip(rl_actions, -3.0, 3.0)
        # Apply to RL vehicles only
        self.k.vehicle.apply_acceleration(rl_ids, clipped_actions)
    
    # Human vehicles are controlled automatically by IDMController
```

### 3. Get State Observations for RL Vehicles Only

```python
def get_state(self):
    """Get state for RL vehicles only."""
    rl_ids = sorted(self.k.vehicle.get_rl_ids())
    
    states = []
    for rl_id in rl_ids:
        pos = self.k.vehicle.get_x_by_id(rl_id)
        speed = self.k.vehicle.get_speed(rl_id)
        # Add other observations as needed
        states.extend([pos, speed])
    
    return np.array(states)
```

### 4. Compute Rewards Based on RL Vehicle Performance

```python
def compute_reward(self, rl_actions, **kwargs):
    """Compute reward based on RL vehicles only."""
    rl_ids = self.k.vehicle.get_rl_ids()
    
    if len(rl_ids) == 0:
        return 0.0
    
    # Example: reward based on speed maintenance
    rl_speeds = [self.k.vehicle.get_speed(vid) for vid in rl_ids]
    target_speed = 20.0  # m/s
    avg_speed = np.mean(rl_speeds)
    
    reward = -abs(avg_speed - target_speed) / target_speed
    return reward
```

## Important Notes

1. **RLController is Required**: Vehicles with `RLController` are automatically added to `get_rl_ids()`. All other vehicles are in `get_human_ids()`.

2. **Action Space**: The action space should match the number of RL vehicles:
   ```python
   @property
   def action_space(self):
       num_rl = self.initial_vehicles.num_rl_vehicles
       return gym.spaces.Box(low=-3.0, high=3.0, shape=(num_rl,))
   ```

3. **Human Vehicles are Automatic**: Human-driven vehicles (with `IDMController`, etc.) are controlled automatically by Flow. You don't need to provide actions for them.

4. **Color Format**: Colors in Flow use RGB format as strings: `"R,G,B"` (e.g., `"255,0,0"` for red) or as tuples in TraCI: `(R, G, B, alpha)`.

## Example Files

- `single_car.py`: Configuration with mixed RL and human vehicles
- `mixed_env.py`: Custom environment showing how to distinguish and handle each type

## Running the Simulation

```bash
cd experiments
python run_single_car.py
```

In the SUMO GUI, you should see:
- **Red vehicles**: RL-trained vehicles
- **Blue vehicles**: Human-driven vehicles

You can verify in the console output that the counts match your configuration.

