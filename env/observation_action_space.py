from gymnasium import spaces
import numpy as np

# Standard Observation Shapes
observation_shapes = {
    'rgb_data': (224, 224, 3),
    'position': (3,),
    'target_position': (3,),
    'next_waypoint_position': (3,),
    'speed': (1,)
}



# Observation Space (8 features in 'rest')
REST_DIM = 8

observation_space = spaces.Dict({
    'rgb_data': spaces.Box(low=0, high=255, shape=observation_shapes['rgb_data'], dtype=np.uint8),
    'rest': spaces.Box(low=-np.inf, high=np.inf, shape=(REST_DIM,), dtype=np.float32)
})

# Action Space
# For continuous actions (steering [-1.0, 1.0], throttle/brake [-1.0, 1.0])
action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
