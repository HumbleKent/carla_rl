from gymnasium import spaces
import numpy as np

# Standard Observation Shapes
observation_shapes = {
    'rgb_data': (360, 640, 3),
    'position': (3,),
    'target_position': (3,),
    'next_waypoint_position': (3,),
    'speed': (1,),
    'num_of_situations': 4
}

situations_map = {
    "Road": 0,
    "Roundabout": 1,
    "Junction": 2,
    "Tunnel": 3
}

# Observation Space (23 features in 'rest')
# 2 nav features (dist to target, dist to next wp) 
# + 4 kinematics (yaw_err, ang_z, fwd_v, lat_v) 
# + 2 action history 
# + 15 cones (5 * 3) = 23
REST_DIM = 2 + 4 + 2 + (5 * 3)

observation_space = spaces.Dict({
    'rgb_data': spaces.Box(low=0, high=255, shape=observation_shapes['rgb_data'], dtype=np.uint8),
    'rest': spaces.Box(low=-np.inf, high=np.inf, shape=(REST_DIM,), dtype=np.float32)
})

# Action Space
# For continuous actions (steering [-1.0, 1.0], throttle/brake [-1.0, 1.0])
action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
