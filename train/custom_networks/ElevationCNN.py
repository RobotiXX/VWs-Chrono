import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ElevationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=16):
        super(ElevationExtractor, self).__init__(observation_space, features_dim=features_dim)
        
        input_size = observation_space.shape[0]

        # Define a simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),  # Adjust the size as per your requirement
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process observations through the network
        return self.network(observations)