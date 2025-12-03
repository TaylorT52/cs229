"""CTDE (Centralized Training with Decentralized Execution) model for RLlib.

This module provides a PyTorch-based RLlib custom model that implements a
decentralized actor (policy) which acts from local observations and a
centralized critic (value function) which can condition on a global
observation/state during training.

Design / usage
- The model expects the environment to provide observations as a Dict space
  with two entries: ``{"local": local_obs, "global": global_obs}``.
  - ``local``: per-agent observation used by the policy (shape e.g. (5,))
  - ``global``: centralized observation/state used by the value network
               (shape depends on your environment; e.g. full concatenated
               state from all agents)

If your environment cannot return a dict-shaped observation, you can still
use this model by passing the global observation size via
``model_config['custom_model_config']['global_obs_dim']`` and supplying the
global information through RLlib's observation Dict wrapper or by returning
an observation that concatenates local+global and splitting inside the
environment. See the README in this project for examples.

The model exposes a policy network (produces action logits / outputs)
and a separate value network that consumes the centralized/global input.

This file only implements the model. To use it with RLlib's PPOTrainer,
set in the trainer config:

config['model'] = {
    'custom_model': 'ctde_torch_model',
    'custom_model_config': {
        'global_obs_dim': <int>,  # optional if your obs is a dict
    }
}

And register the model with ModelCatalog:

from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model('ctde_torch_model', CTDEModel)

Notes / limitations
- This implementation uses PyTorch and RLlib's TorchModelV2 API.
- The environment must provide a usable global observation during training
  for the centralized critic to be meaningful. At execution time the policy
  only uses the local observation.
"""

from typing import Dict, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.framework import try_import_torch

try:
    torch, nn
except Exception:
    torch = None


class CTDEModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self._hidden_layers: Sequence[int] = tuple(
            model_config.get("fcnet_hiddens", [64, 64])
        )
        self._activation = model_config.get("fcnet_activation", "relu")

        # Determine local / global observation sizes
        # IMPORTANT: Check custom_model_config FIRST because RLlib may flatten Dict spaces
        self._local_dim = None
        self._global_dim = None
        
        custom_cfg = model_config.get("custom_model_config", {})
        if "local_obs_dim" in custom_cfg:
            self._local_dim = int(custom_cfg["local_obs_dim"])
        if "global_obs_dim" in custom_cfg:
            self._global_dim = int(custom_cfg["global_obs_dim"])

        # If not in config, try to infer from obs_space
        # Note: RLlib may flatten Dict, so obs_space.shape might be wrong
        if self._local_dim is None or self._global_dim is None:
            try:
                from gym.spaces import Dict as GymDict

                if isinstance(obs_space, GymDict):
                    # Dict space: extract from sub-spaces
                    if "local" in obs_space.spaces and self._local_dim is None:
                        self._local_dim = int(np.prod(obs_space.spaces["local"].shape))
                    if "global" in obs_space.spaces and self._global_dim is None:
                        self._global_dim = int(np.prod(obs_space.spaces["global"].shape))
            except Exception:
                # Fall through - will use defaults or raise error
                pass

        if self._local_dim is None:
            # As a last resort, try to infer from obs_space shape
            # But be careful - if obs_space is Dict, shape might be flattened
            try:
                # If it's a Dict space, don't use the overall shape
                # (that would be the flattened size)
                if isinstance(obs_space, GymDict):
                    # Already handled above, but if we get here, try first space
                    if len(obs_space.spaces) > 0:
                        first_space = list(obs_space.spaces.values())[0]
                        self._local_dim = int(np.prod(first_space.shape))
                    else:
                        raise ValueError("Empty Dict observation space")
                else:
                    self._local_dim = int(np.prod(obs_space.shape))
            except Exception:
                raise ValueError(
                    "CTDEModel: could not infer local observation dimension. "
                    "Provide a Dict observation space with 'local' and 'global' keys,"
                    " or set custom_model_config['local_obs_dim'] and "
                    "['global_obs_dim'].")

        if self._global_dim is None:
            self._global_dim = self._local_dim

        # Debug: Print dimensions to verify
        print(f"CTDEModel initialized:")
        print(f"  - Local dim: {self._local_dim}")
        print(f"  - Global dim: {self._global_dim}")
        print(f"  - Obs space type: {type(obs_space)}")
        if hasattr(obs_space, 'shape'):
            print(f"  - Obs space shape: {obs_space.shape}")

        #build policy (actor) network from local observations -> action logits 
        #NOTE THIS IS AS ALL LOCAL OBSERVATIONS
        policy_layers = []
        in_dim = self._local_dim
        for size in self._hidden_layers:
            policy_layers.append(nn.Linear(in_dim, size))
            policy_layers.append(self._get_activation())
            in_dim = size
        policy_layers.append(nn.Linear(in_dim, num_outputs))
        self.policy_net = nn.Sequential(*policy_layers)

        # Build value (critic) network from centralized/global observations
        #NOTE THIS IS USES GLOBAL OBSERVATIONS 
        value_layers = []
        in_dim = self._global_dim
        for size in self._hidden_layers:
            value_layers.append(nn.Linear(in_dim, size))
            value_layers.append(self._get_activation())
            in_dim = size
        value_layers.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*value_layers)

        self._last_value = None

    def _get_activation(self):
        if self._activation == "tanh":
            return nn.Tanh()
        elif self._activation == "relu":
            return nn.ReLU()
        elif self._activation == "leaky_relu":
            return nn.LeakyReLU(0.01)
        else:
            return nn.ReLU()

    def forward(self, input_dict: Dict[str, TensorType], state, seq_lens):
        obs = input_dict["obs"]

        local_obs = None
        global_obs = None

        # Handle Dict observation (RLlib may pass as dict or flattened tensor)
        if isinstance(obs, dict):
            # Dict format: {"local": tensor, "global": tensor}
            if "local" in obs:
                local_obs = obs["local"].float()
            else:
                # Try first key if "local" not found
                first_key = list(obs.keys())[0]
                local_obs = obs[first_key].float()
            if "global" in obs:
                global_obs = obs["global"].float()
        else:
            # RLlib may flatten Dict observations - need to split them
            # If obs is a tensor, it might be flattened Dict
            obs_tensor = obs.float()
            
            # Check if this is a flattened Dict (local + global concatenated)
            # Expected: local (5) + global (10) = 15 total for 2 agents
            if obs_tensor.shape[-1] == self._local_dim + self._global_dim:
                # Split into local and global
                local_obs = obs_tensor[..., :self._local_dim]
                global_obs = obs_tensor[..., self._local_dim:]
            elif obs_tensor.shape[-1] == self._local_dim:
                # Only local observation provided
                local_obs = obs_tensor
            else:
                # Fallback: treat entire tensor as local
                local_obs = obs_tensor

        # Ensure correct shape (batch, features)
        if local_obs is not None:
            if len(local_obs.shape) > 2:
                local_obs = local_obs.view(local_obs.size(0), -1)
            elif len(local_obs.shape) == 1:
                local_obs = local_obs.unsqueeze(0)

        if global_obs is not None:
            if len(global_obs.shape) > 2:
                global_obs = global_obs.view(global_obs.size(0), -1)
            elif len(global_obs.shape) == 1:
                global_obs = global_obs.unsqueeze(0)

        # Policy forward (local only)
        pi_out = self.policy_net(local_obs)

        # Critic forward (global when available, otherwise local)
        critic_in = global_obs if global_obs is not None else local_obs
        value = self.value_net(critic_in)
        # value shape: [B, 1] -> store as [B]
        self._last_value = value.view(-1)

        return pi_out, state

    def value_function(self) -> TensorType:
        assert self._last_value is not None, "must call forward() before value_function()"
        return self._last_value


__all__ = ["CTDEModel"]
