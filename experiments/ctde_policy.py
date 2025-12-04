"""Custom model for CTDE (Centralized Training Decentralized Execution) with centralized critic.

The critic uses global state (concatenation of all agents' observations),
while the policy uses only local observations (decentralized execution).
"""

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gym.spaces import Box, Dict


class CTDEModel(TorchModelV2, nn.Module):
    """Model with centralized critic for CTDE.
    
    Policy network: Uses local observation (decentralized execution)
    Critic network: Uses global observation (centralized training)
    """
    
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs
    ):
        """Initialize the CTDE model.
        
        Args:
            obs_space: Observation space (should be Dict with "local" and "global" keys)
            action_space: Action space
            num_outputs: Number of action outputs
            model_config: Model configuration dict with:
                - custom_model_config:
                    - global_obs_dim: Size of global observation (num_agents * obs_dim)
                    - local_obs_dim: Size of local observation per agent
            name: Model name
        """
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        
        # Extract configuration
        custom_config = model_config.get("custom_model_config", {})
        # Default dimensions (will be overridden by actual obs space if available)
        # For lane changes: 14 features per agent, otherwise 5
        default_local_dim = custom_config.get("local_obs_dim", 14)  # Default: 14 features (with lane info + relative speeds)
        default_global_dim = custom_config.get("global_obs_dim", 28)  # Default: 2 agents * 14 features
        
        # Try to infer from observation space if it's a Dict
        if isinstance(obs_space, Dict):
            if "local" in obs_space.spaces:
                self.local_obs_dim = obs_space.spaces["local"].shape[0]
            else:
                self.local_obs_dim = default_local_dim
            if "global" in obs_space.spaces:
                self.global_obs_dim = obs_space.spaces["global"].shape[0]
            else:
                self.global_obs_dim = default_global_dim
        else:
            self.local_obs_dim = default_local_dim
            self.global_obs_dim = default_global_dim
        
        # Policy network: uses local observation only
        # Create a Box space for local observations
        local_obs_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.local_obs_dim,),
            dtype=np.float32
        )
        
        self.policy_net = FullyConnectedNetwork(
            local_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_policy"
        )
        
        # Critic network: uses global observation (centralized)
        # Build critic network manually
        critic_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "tanh")
        
        if activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "relu":
            activation_fn = nn.ReLU
        else:
            activation_fn = nn.Tanh
        
        # Critic network layers
        critic_layers = []
        prev_size = self.global_obs_dim
        for size in critic_hiddens:
            critic_layers.append(nn.Linear(prev_size, size))
            critic_layers.append(activation_fn())
            prev_size = size
        # Output layer (value function)
        critic_layers.append(nn.Linear(prev_size, 1))
        
        self.critic_net = nn.Sequential(*critic_layers)
        
        # Value function output
        self._value_out = None
        
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the model.
        
        Args:
            input_dict: Dictionary containing:
                - "obs": Dict with "local" and "global" keys
            state: RNN state (not used)
            seq_lens: Sequence lengths (not used)
        
        Returns:
            Policy logits and state
        """
        # Extract observations - must be a dict with "local" and "global" keys
        obs = input_dict["obs"]
        
        if not isinstance(obs, dict):
            raise ValueError(
                f"CTDE model expects dict observation with 'local' and 'global' keys, "
                f"got {type(obs)}. Check that use_ctde_obs=True in env config."
            )
        
        if "local" not in obs or "global" not in obs:
            raise ValueError(
                f"CTDE model expects observation dict with 'local' and 'global' keys, "
                f"got keys: {list(obs.keys())}"
            )
        
        local_obs = obs["local"]
        global_obs = obs["global"]
        
        # Convert to tensors if needed
        if isinstance(local_obs, np.ndarray):
            local_obs = torch.from_numpy(local_obs).float()
        if isinstance(global_obs, np.ndarray):
            global_obs = torch.from_numpy(global_obs).float()
        
        # Ensure proper batching: must be (B, feature_dim)
        if len(local_obs.shape) == 1:
            local_obs = local_obs.unsqueeze(0)
        if len(global_obs.shape) == 1:
            global_obs = global_obs.unsqueeze(0)
        
        # Sanity checks - fail early if shapes are wrong
        assert local_obs.shape[-1] == self.local_obs_dim, (
            f"Local obs shape mismatch: got {local_obs.shape[-1]}, "
            f"expected {self.local_obs_dim}"
        )
        assert global_obs.shape[-1] == self.global_obs_dim, (
            f"Global obs shape mismatch: got {global_obs.shape[-1]}, "
            f"expected {self.global_obs_dim}. "
            f"Check that env returns correct global observation (concat of all local obs)."
        )
        
        # Policy network: uses local observation (decentralized execution)
        # FullyConnectedNetwork expects {"obs": tensor} format
        policy_out, state = self.policy_net({"obs": local_obs}, state, seq_lens)
        
        # Critic network: uses global observation (centralized training)
        # No gradient-breaking operations - use global_obs as-is
        self._value_out = self.critic_net(global_obs).squeeze(-1)
        
        return policy_out, state
    
    def value_function(self):
        """Return the value function output."""
        return self._value_out

