# Pygame Window
IM_WIDTH                = 1920
IM_HEIGHT               = 1080
NUM_COLS                = 2  # Number of columns in the grid
NUM_ROWS                = 2  # Number of rows in the grid
MARGIN                  = 30
BORDER_WIDTH            = 5

# Vehicle and Sensors attributes
SENSOR_FPS              = 30
VERBOSE                 = False
VEHICLE_SENSORS_FILE    = 'env/vehicle_config/train_sensors.json'
VEHICLE_PHYSICS_FILE    = 'env/vehicle_config/default_vehicle_physics.json'
VEHICLE_MODEL           = "vehicle.tesla.model3"

# Simulation attributes
import os
SIM_HOST                = '127.0.0.1'
SIM_PORT                = int(os.getenv('CARLA_PORT', 2000))
SIM_TIMEOUT             = 100.0
SIM_LOW_QUALITY         = False
SIM_OFFSCREEN_RENDERING = False
SIM_DELTA_SECONDS       = 0.05
SIM_FPS                 = 20

# Environment attributes
ENV_SCENARIOS_FILE      = 'env/vehicle_spawn.json'
ENV_MAX_STEPS           = 2400 # Max number of steps per episode. I suggest running the helpfull-scipts/check_max_num_steps.py script to get your number
ENV_WAYPOINT_SPACING    = 2.0
ENV_CONE_LAYOUT_FILE    = 'env/cone_layout.json'
ENV_CONE_THRESHOLD      = 2.2
