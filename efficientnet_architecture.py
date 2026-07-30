import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

# Define the action space for PPO
continuous_action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

class CustomExtractor_PPO_EfficientNet(BaseFeaturesExtractor):
    """
    Improved Feature Extractor for PPO with EfficientNet-B0.
    
    Refinements:
    - Robust input processing: handles uint8/float and NHWC/NCHW formats.
    - Standardized backbone: switched to torchvision.models for better stability.
    - Efficiency: added torch.no_grad() for the frozen EfficientNet backbone.
    - Consistency: aligned with CustomExtractor_PPO_Modular patterns.
    """
    def __init__(self, observation_space: spaces.Dict):
        # Image features from EfficientNet-B0 is 1280
        image_dim = 1280  
        rest_dim = 256   # Dimensionality of the MLP output
        features_dim = image_dim + rest_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = continuous_action_space.shape[0]

        # Use weights=DEFAULT for the latest stable EfficientNet-B0 weights from torchvision
        self.efficientnet_base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet = self.efficientnet_base.features
        
        # Freeze the backbone parameters
        for param in self.efficientnet.parameters():
            param.requires_grad = False  

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # MLP for processing the non-visual "rest" vector (Vel, AngVel, Error, etc.)
        n_rest_features = observation_space['rest'].shape[0]
        self.rest_model = nn.Sequential(
            nn.Linear(n_rest_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, observations):
        # 1. Process RGB Data
        rgb_data = observations['rgb_data']
        
        # Scale uint8 images [0, 255] to float [0, 1]
        if rgb_data.dtype == torch.uint8:
            rgb_data = rgb_data.float() / 255.0
            
        # Standard ImageNet normalization for pre-trained EfficientNet
        # Mean and Std values for RGB channels
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_data.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_data.device).view(1, 3, 1, 1)
        rgb_data = (rgb_data - mean) / std
            
        # Handle NHWC (B, H, W, C) -> NCHW (B, C, H, W) conversion if necessary
        if rgb_data.shape[-1] == 3 and len(rgb_data.shape) == 4:
            rgb_data = rgb_data.permute(0, 3, 1, 2)
            
        # Ensure image size is 224x224 (EfficientNet standard)
        if rgb_data.shape[-2:] != (224, 224):
            rgb_data = F.interpolate(rgb_data, size=(224, 224), mode='bilinear', align_corners=False)

        # Extract features (using torch.no_grad to save memory for frozen backbone)
        with torch.no_grad():
            image_features = self.efficientnet(rgb_data)
        
        image_features = self.global_avg_pool(image_features)
        image_features = torch.flatten(image_features, 1)

        # 2. Process scalar "rest" features
        rest_output = self.rest_model(observations['rest'])

        # 3. Concatenate and return the combined feature vector
        return torch.cat((image_features, rest_output), dim=1)
