import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Define the continuous action space for PPO
continuous_action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

class CustomExtractor_PPO_End2end(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        image_dim = 512  # Dimensionality of the CNN features
        rest_dim = 128   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.AdaptiveAvgPool2d(1)  
        ) 

        # Corrected to 8 features to match environment
        self.rest_model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        # Image Processing
        rgb_data = observations['rgb_data']
        if rgb_data.dtype == torch.uint8:
            rgb_data = rgb_data.float() / 255.0
        if rgb_data.shape[-1] == 3 and len(rgb_data.shape) == 4:
            rgb_data = rgb_data.permute(0, 3, 1, 2)
        
        image_features = self.image_model(rgb_data)
        image_features = torch.flatten(image_features, 1)

        rest_output = self.rest_model(observations['rest'])
        
        combined_features = torch.cat((image_features, rest_output), dim=1)
        return combined_features


class CustomExtractor_PPO_Modular(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        image_dim = 1280  
        rest_dim = 256   
        features_dim = image_dim + rest_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load EfficientNet-B0 from torchvision (Stable/Offline)
        self.efficientnet_base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet = self.efficientnet_base.features
        
        for param in self.efficientnet.parameters():
            param.requires_grad = False  
        self.efficientnet.eval()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Corrected to 8 features to match environment
        self.rest_model = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, observations):
        # 1. Image Processing
        rgb_data = observations['rgb_data']
        if rgb_data.dtype == torch.uint8:
            rgb_data = rgb_data.float() / 255.0
        if rgb_data.shape[-1] == 3 and len(rgb_data.shape) == 4:
            rgb_data = rgb_data.permute(0, 3, 1, 2)
        if rgb_data.shape[-2:] != (224, 224):
            rgb_data = F.interpolate(rgb_data, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        with torch.no_grad():
            image_features = self.efficientnet(rgb_data)
        image_features = self.global_avg_pool(image_features)
        image_features = torch.flatten(image_features, 1)

        # 2. Rest Processing
        rest_output = self.rest_model(observations['rest'])
                
        combined_features = torch.cat((image_features, rest_output), dim=1)
        return combined_features
