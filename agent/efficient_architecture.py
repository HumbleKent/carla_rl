import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

# Define the action space for PPO
action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

class CustomExtractor_PPO_EfficientNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # Image features from EfficientNet-B1 (after Global Average Pooling) is 1280 (for B0) 
        # B0: 1280, B1: 1280.
        image_dim = 1280  
        rest_dim = 128   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(observation_space, features_dim=features_dim)
        
        # 1. EfficientNet-B0 for Image Feature Extraction
        # We use weights=EfficientNet_B0_Weights.DEFAULT for pretrained features
        # which usually generalize better for visual tasks.
        self.efficientman = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the classification head (classifier), we only want the features
        self.image_model = self.efficientman.features
        
        # Add a Global Average Pooling layer to get a fixed-size vector (1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 2. MLP for processing the "Rest" vector (23 features)
        # Input: 23 (2 nav + 4 kinematics + 2 action + 15 cones)
        self.rest_model = nn.Sequential(
            nn.Linear(23, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        # Extract image features
        image_features = self.image_model(rgb_input)
        image_features = self.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)

        # Extract rest features
        rest_output = self.rest_model(rest_input)
        
        # Ensure batch dimension for rest_output if needed (though SB3 handles this)
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)

        # Concatenate both feature sets
        combined_features = torch.cat((image_features, rest_output), dim=1)
        return combined_features

    def process_observations(self, observations):
        # 1. Image Processing
        rgb_data = observations['rgb_data']
        
        # If input is uint8 (0-255), convert to float and normalize to [0, 1]
        # This acts as a safety layer if SB3 hasn't normalized it yet
        if rgb_data.dtype == torch.uint8:
            rgb_data = rgb_data.float() / 255.0

        # Ensure (Batch, Channels, H, W) format
        # If input is (Batch, H, W, Channels), transpose it
        if rgb_data.shape[-1] == 3 and len(rgb_data.shape) == 4:
            rgb_data = rgb_data.permute(0, 3, 1, 2)

        # Resize to EfficientNet's expected input size (224x224)
        rgb_data = F.interpolate(rgb_data, size=(224, 224), mode='bilinear', align_corners=False)
        
        return (rgb_data.to(self.device), observations['rest'].to(self.device))
