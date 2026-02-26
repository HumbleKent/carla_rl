import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

# Define the continuous action space for PPO
continuous_action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

class CustomExtractor_PPO_Cone(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        image_dim = 512  # Dimensionality of the CNN features
        rest_dim = 128   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(observation_space, features_dim=features_dim)
        self.action_dim = continuous_action_space.shape[0]  # Dimensionality of the action space

        # Custom CNN for processing the RGB data
        self.image_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to get a fixed-size feature vector
        ) 

        # Define the neural network architecture for processing the rest of the input
        # Input: 23 (2 nav + 4 kinematics + 2 action + 15 cones)
        self.rest_model = nn.Sequential(
            nn.Linear(23, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        image_features = self.image_model(rgb_input)
        image_features = torch.flatten(image_features, 1)

        rest_output = self.rest_model(rest_input)
                
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)

        combined_features = torch.cat((image_features, rest_output), dim=1)
        return combined_features

    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        rgb_data = rgb_data / 255.0  # Normalize the pixel values to be in the range [0, 1]

        return (rgb_data.float().to(self.device), observations['rest'])
