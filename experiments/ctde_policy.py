import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gym.spaces import Box, Dict

class CTDEModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        

        custom_config = model_config.get("custom_model_config", {})
        default_local_dim = custom_config.get("local_obs_dim", 14)
        default_global_dim = custom_config.get("global_obs_dim", 28) 
        
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

        critic_hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "tanh")
        
        if activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "relu":
            activation_fn = nn.ReLU
        else:
            activation_fn = nn.Tanh
        
        critic_layers = []
        prev_size = self.global_obs_dim
        for size in critic_hiddens:
            critic_layers.append(nn.Linear(prev_size, size))
            critic_layers.append(activation_fn())
            prev_size = size
        critic_layers.append(nn.Linear(prev_size, 1))
        
        self.critic_net = nn.Sequential(*critic_layers)
        self._value_out = None
        
    def forward(self, input_dict, state, seq_lens):
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
        
        if isinstance(local_obs, np.ndarray):
            local_obs = torch.from_numpy(local_obs).float()
        if isinstance(global_obs, np.ndarray):
            global_obs = torch.from_numpy(global_obs).float()
        
        if len(local_obs.shape) == 1:
            local_obs = local_obs.unsqueeze(0)
        if len(global_obs.shape) == 1:
            global_obs = global_obs.unsqueeze(0)
        
        assert local_obs.shape[-1] == self.local_obs_dim, (
            f"Local obs shape mismatch: got {local_obs.shape[-1]}, "
            f"expected {self.local_obs_dim}"
        )
        assert global_obs.shape[-1] == self.global_obs_dim, (
            f"Global obs shape mismatch: got {global_obs.shape[-1]}, "
            f"expected {self.global_obs_dim}. "
            f"Check that env returns correct global observation (concat of all local obs)."
        )

        policy_out, state = self.policy_net({"obs": local_obs}, state, seq_lens)
        self._value_out = self.critic_net(global_obs).squeeze(-1)
        
        return policy_out, state
    
    def value_function(self):
        return self._value_out

