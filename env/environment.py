from networkx.generators import spectral_graph_forge
import numpy as np
import json
import time
import random
import carla
import gymnasium as gym
from gymnasium.envs.registration import register
import configuration as config

from src.world import World
from src.server import CarlaServer
from src.vehicle import Vehicle
from src.display import Display
from env.reward import Reward
from env.observation_action_space import (
    observation_space, 
    action_space
)
from env.pre_processing import PreProcessing
from src.debug_manager import DebugManager
from src.route_planner import AdvancedRoutePlanner

register(
    id="carla-rl-gym",
    entry_point="env.environment:CarlaEnv",
    max_episode_steps=config.ENV_MAX_STEPS,
)

# Unified Environment for CARLA Training
class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": config.SIM_FPS}
    
    def __init__(self, scenarios=[], time_limit=60, initialize_server=False, 
                 random_weather=False, random_traffic=False, synchronous_mode=True, 
                 show_sensor_data=False, has_traffic=True, verbose=True, 
                 spawn_cones=True, port=None, is_eval=False,
                 camera=config.CAMERA_VIEW,
                 weather_override=None,
                 force_lights=False,
                 debug_features=[]):
        super().__init__()
        
        self.__port = port if port is not None else config.SIM_PORT
        self.__random_weather = random_weather
        self.__weather_override = weather_override
        self.__force_lights = force_lights
        self.__random_traffic = random_traffic
        self.__synchronous_mode = synchronous_mode
        self.__show_sensor_data = show_sensor_data
        self.__has_traffic = has_traffic
        self.__verbose = verbose
        self.__automatic_server_initialization = initialize_server
        self.__cones_spawned = False
        self.__spawn_cones = spawn_cones 
        self.__is_eval = is_eval
        
        # Debugging
        self.debug_manager = DebugManager(debug_features, tag=f"Port {self.__port}")

        # 1. Server initialization
        if self.__automatic_server_initialization:
            self.__server_process = CarlaServer.initialize_server(
                low_quality=config.SIM_LOW_QUALITY, 
                offscreen_rendering=config.SIM_OFFSCREEN_RENDERING
            )
        
        # 2. Setup World, Vehicle, and Processing
        self.__world = World(synchronous_mode=self.__synchronous_mode, port=self.__port)
        self.__get_situations(scenarios)
        self.__vehicle = Vehicle(self.__world.get_world(), camera=camera)
        self.pre_processing = PreProcessing()
        self.__reward_calculator = Reward()

        self.__reward_func = self.__reward_calculator # Default
        
        # 3. Spaces
        self.observation_space = observation_space
        self.action_space = action_space
        
        # 4. State variables
        self._active_scenario_name = None
        self._active_scenario_dict = None
        self._waypoints = None

        self.__episode_number = 0
        self.__truncated = False
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.current_cone_data = []
        self.num_cones_to_track = 5
        
        self.__cone_config = self.__load_cone_config()
        self.__route_planner = AdvancedRoutePlanner(
            self.__world.get_world(), 
            sampling_res=config.ENV_WAYPOINT_SPACING,
            cone_threshold=config.ENV_CONE_THRESHOLD,
            verbose=False
        )
        if self.__cone_config:
            self.__route_planner.set_cones(self.__cone_config)

    def __load_cone_config(self):
        try:
            with open(config.ENV_CONE_LAYOUT_FILE, 'r') as f:
                return json.load(f)
        except:
            return None

    def reset(self, seed=None, options=None):
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # Select Scenario
        if options and 'scenario_name' in options:
            self._active_scenario_name = options['scenario_name']
        else:
            self._active_scenario_name = self.__chose_situation(seed)
            
        if self.__verbose:
            print(f"[Port {self.__port}] Loading scenario {self._active_scenario_name}...")
        
        # Select correct reward logic based on scenario
        if "Pothole" in self._active_scenario_name:
            self.__reward_func = self.__reward_calculator
            if self.__verbose: print(f"[ENV] Switched to Pothole Reward logic.")
        else:
            self.__reward_func = self.__reward_calculator
            if self.__verbose: print(f"[ENV] Switched to Standard Reward logic (Restored).")
            
        # Ensure world is clean before loading new scenario
        self.clean_scenario()
        
        self.load_scenario(self._active_scenario_name, seed)
        
        # Initialize target position for debugging and rewards
        self._reward_target_pos = np.array([
            self._active_scenario_dict['target_position']['x'], 
            self._active_scenario_dict['target_position']['y'], 
            self._active_scenario_dict['target_position']['z']
        ], dtype=np.float32)
        
        self.place_spectator_above_vehicle()
        
        # Pathfinding
        s_pos = self._active_scenario_dict['initial_position']
        start_loc = carla.Location(x=s_pos['x'], y=s_pos['y'], z=s_pos['z'])
        waypoints_locations = self.get_path_waypoints(spacing=config.ENV_WAYPOINT_SPACING, start_loc=start_loc)
        if self.__verbose:
            print(f"[Port {self.__port}] Generated {len(waypoints_locations)} waypoints.")
        
        self._waypoints = [np.array([w.x, w.y, w.z]) for w in waypoints_locations]
        
        # Draw full planned route at episode start (long lifetime so whole path is visible)
        self.debug_manager.draw_waypoints(self.__world.get_world(), self._waypoints, life_time=30.0)
        self.debug_manager.draw_target(self.__world.get_world(), self._reward_target_pos, life_time=30.0)
        
        # Wait for sensors
        waited = 0
        while not self.__vehicle.sensors_ready() and waited < 50:
            self.__world.tick()
            time.sleep(0.1)
            waited += 1
            
        self._update_observation()
        self.__reward_func.reset(self._waypoints)

        self.__episode_number += 1
        self.__start_timer()
        self.number_of_steps = 0
        
        # Metrics tracking
        self.__initial_dist_to_target = np.linalg.norm(self._reward_current_pos - self._reward_target_pos)
        self.__cumulative_speed = 0.0
        self.__cumulative_steer_jerk = 0.0
        self.__cumulative_throttle_jerk = 0.0
        self.__prev_steer = 0.0
        self.__prev_throttle = 0.0
        
        # Spawn cones AFTER potential map reloads in load_scenario
        if self.__spawn_cones:
            self.__spawn_all_cones()
            
        info = {
            'scenario_name': self._active_scenario_name, 
            'waypoints': self._waypoints,
            'initial_dist': self.__initial_dist_to_target
        }
        return self.__observation, info

    def render(self, mode='human'):
        if mode == 'human':
            self.__world.tick()
            if hasattr(self, 'display'):
                self.display.play_window_tick()
        else:
            raise NotImplementedError("This mode is not implemented yet")

    def step(self, action):
        if self.__synchronous_mode:
            self.__world.tick()
            
        # Self-healing: Ensure cones haven't been wiped mid-episode by a parallel process
        if self.__spawn_cones and self.__episode_number > 0 and self.number_of_steps % 10 == 0:
            if self.__world.get_cone_count() == 0 and self.__cone_config is not None:
                if self.__verbose: print(f"[Port {self.__port}] Cones disappeared mid-episode! Respawning...")
                self.__spawn_all_cones()

        self.number_of_steps += 1
        
        # Control
        action_array = np.array(action)
        self.__vehicle.control_vehicle(action_array)

        # Update
        self._update_observation()
        self.place_spectator_above_vehicle()

        # Metrics tracking
        curr_steer = self.__vehicle.get_steering()
        curr_throttle = self.__vehicle.get_throttle_brake()
        if self.number_of_steps > 1:
            self.__cumulative_steer_jerk += abs(curr_steer - self.__prev_steer)
            self.__cumulative_throttle_jerk += abs(curr_throttle - self.__prev_throttle)
        self.__prev_steer = curr_steer
        self.__prev_throttle = curr_throttle
        self.__cumulative_speed += self._reward_speed
        
        # Reward
        reward = self.__reward_func.calculate_reward(
            self.__vehicle, 
            self._reward_current_pos, 
            self._reward_target_pos, 
            self._reward_next_waypoint_pos, 
            speed=self._reward_speed,
            yaw_error=self.__current_yaw_error,
            cte=self.__current_cte,
            cone_data=self.current_cone_data,
            step_count=self.number_of_steps,
            verbose=self.__verbose,
            debug_manager=self.debug_manager
        )
        
        terminated = self.__reward_func.get_terminated()
        self._waypoints = self.__reward_func.get_waypoints()
        
        # Dynamic rendering of remaining waypoints + target with short lifetime
        # This refreshes every step so waypoints "consume" as the agent progresses
        if self._waypoints:
            self.debug_manager.draw_waypoints(self.__world.get_world(), self._waypoints, life_time=0.2)
        self.debug_manager.draw_target(self.__world.get_world(), self._reward_target_pos, life_time=0.2)
        
        # Use exact step count instead of wall-clock time for reproducible RL episodes
        self.__truncated = (self.number_of_steps >= config.ENV_MAX_STEPS)
        
        if terminated or self.__truncated:
            reason = "SUCCESS (Goal Reached)" if terminated else "TIMEOUT (Max Steps)"
            if self.__vehicle.collision_occurred(): reason = "CRASH (Collision)"
            elif self.__vehicle.lane_invasion_occurred(): reason = "OFF-ROAD (Lane Invasion)"
            
            # If reward function has a more specific reason, use it
            reward_reason = self.__reward_func.get_termination_reason()
            if reward_reason != "None":
                reason = reward_reason

            # Use debug_manager to log termination and summary
            # We log the summary FIRST so the monitor script captures the Reason from the End line following it
            if hasattr(self.__reward_func, 'get_episode_summary'):
                self.debug_manager.log("term", self.__reward_func.get_episode_summary())
            msg = f"Episode {self.__episode_number} End | Reason: {reason} | Total Score: {self.__reward_func.get_total_ep_reward():.2f}"
            self.debug_manager.log("term", msg)
            self.clean_scenario()
        else:
            reason = "None"
        
        self.last_action = np.array(action, dtype=np.float32)
        
        # Reset per-step sensor flags AFTER reward calculation so this step's
        # collision/lane-invasion events are consumed, not carried into the next step.
        self.__vehicle.reset_step()
        
        info = {
            'scenario_name': self._active_scenario_name, 
            'waypoints': self._waypoints, 
            'port': self.__port,
            'termination_reason': reason
        }

        if terminated or self.__truncated:
            final_dist = np.linalg.norm(self._reward_current_pos - self._reward_target_pos)
            # Ensure it's not negative and doesn't exceed 100%
            completion_pct = max(0.0, min(100.0, (1.0 - (final_dist / self.__initial_dist_to_target)) * 100.0)) if self.__initial_dist_to_target > 0.1 else 0.0
            if terminated and "Goal Reached" in reason: completion_pct = 100.0

            info.update({
                'completion_pct': completion_pct,
                'avg_speed': self.__cumulative_speed / self.number_of_steps if self.number_of_steps > 0 else 0,
                'steer_jerk': self.__cumulative_steer_jerk / self.number_of_steps if self.number_of_steps > 0 else 0,
                'throttle_jerk': self.__cumulative_throttle_jerk / self.number_of_steps if self.number_of_steps > 0 else 0
            })
        
        return self.__observation, reward, terminated, self.__truncated, info

    def _update_observation(self):
        obs_dict = self.__vehicle.get_observation_data()
        if 'rgb_data' not in obs_dict: return
            
        rgb_image = obs_dict['rgb_data']
        vehicle_actor = self.__vehicle.get_vehicle()
        if vehicle_actor is None: return
            
        vehicle_loc = vehicle_actor.get_location()
        current_position = np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
        target_position = np.array([
            self._active_scenario_dict['target_position']['x'], 
            self._active_scenario_dict['target_position']['y'], 
            self._active_scenario_dict['target_position']['z']
        ])
        
        if self._waypoints and len(self._waypoints) > 0:
            next_waypoint_position = self._waypoints[0]
            second_waypoint_position = self._waypoints[1] if len(self._waypoints) > 1 else next_waypoint_position
        else:
            next_waypoint_position = target_position
            second_waypoint_position = target_position

        velocity = vehicle_actor.get_velocity()
        ang_vel = vehicle_actor.get_angular_velocity()
        transform = vehicle_actor.get_transform()
        
        # Raw Data for Processing
        raw_obs = {
            'rgb_data': np.uint8(rgb_image),
            'position': current_position,
            'target_position': target_position,
            'next_waypoint_position': next_waypoint_position,
            'velocity': np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32),
            'angular_velocity': np.array([ang_vel.x, ang_vel.y, ang_vel.z], dtype=np.float32),
            'rotation': np.array([transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll], dtype=np.float32),
            'second_waypoint_position': second_waypoint_position
        }
        
        # Cone Detection
        active_cones = self.__world.get_active_cones()
        self.current_cone_data = []
        nearest_cone_loc = None
        if active_cones:
            cones_with_dist = []
            for cone in active_cones:
                if not cone.is_alive: continue
                dist = vehicle_loc.distance(cone.get_location())
                cones_with_dist.append((cone, dist))
            
            cones_with_dist.sort(key=lambda x: x[1])
            if len(cones_with_dist) > 0:
                nearest_cone_loc = cones_with_dist[0][0].get_location()

            for cone, dist in cones_with_dist[:self.num_cones_to_track]:
                loc = cone.get_location()
                self.current_cone_data.append({
                    'rel_x': loc.x - vehicle_loc.x, 
                    'rel_y': loc.y - vehicle_loc.y, 
                    'dist': dist
                })
        
        # Processed Observation
        # Now returns the observation and raw error terms.
        self.__observation, self.__current_yaw_error, self.__current_cte, self.__current_wp2_yaw_error = self.pre_processing.preprocess_data(
            raw_obs, 
            cone_data=self.current_cone_data, 
            last_action=self.last_action
        )
        
        # Debugging
        self.debug_manager.show_nn_input(
            raw_obs['rgb_data'], 
            self.__observation['rest'],
            self.last_action
        )
        
        # self.debug_manager.draw_dynamic_distances(
        #     self.__world.get_world(),
        #     vehicle_loc,
        #     target_position,
        #     nearest_cone_loc
        # )
        
        # Refresh the Pygame sensor window every step
        if self.__show_sensor_data and hasattr(self, 'display'):
            self.display.play_window_tick()
        
        # Reward Aux
        self._reward_target_pos = target_position
        self._reward_current_pos = current_position
        self._reward_next_waypoint_pos = next_waypoint_position
        self._reward_speed = 3.6 * velocity.length()

    def close(self):
        """Clean up the environment by destroying all actors."""
        if self.__verbose:
            print(f"[Port {self.__port}] Closing environment and destroying actors...")
            
        try:
            if hasattr(self, '_CarlaEnv__vehicle') and self.__vehicle:
                self.__vehicle.destroy_vehicle()
        except Exception as e:
            print(f"Error destroying vehicle on port {self.__port}: {e}")
            
        try:
            if hasattr(self, '_CarlaEnv__world') and self.__world:
                # Destroy traffic, pedestrians, and cones
                self.__world.destroy_world()
        except Exception as e:
            print(f"Error destroying world actors on port {self.__port}: {e}")
            
        if self.__show_sensor_data and hasattr(self, 'display'):
            try:
                self.display.close_window()
            except:
                pass

        # If we initialized the server automatically, we should close it
        if hasattr(self, '_CarlaEnv__automatic_server_initialization') and self.__automatic_server_initialization:
            if hasattr(self, '_CarlaEnv__server_process') and self.__server_process:
                CarlaServer.close_server(self.__server_process)

    # --- Scenario & World Helpers ---
    def load_scenario(self, scenario_name, seed=None):
        try:
            scenario_dict = self.situations_dict[scenario_name]
        except:
            scenario_name = self.__choose_random_situation(seed)
            scenario_dict = self.situations_dict[scenario_name]
            
        self._active_scenario_name = scenario_name
        self._active_scenario_dict = scenario_dict
        
        self.__load_world(scenario_dict['map_name'])
        self.__world.update_traffic_map()
        time.sleep(2.0)
        
        # Weather
        self.__load_weather(scenario_dict['weather_condition'])

        # Ego vehicle
        self.__spawn_vehicle(scenario_dict)
            
        # Create the Display window only once (guard against re-creation every episode)
        if self.__show_sensor_data and not hasattr(self, 'display'):
            self.display = Display('Sensor Feed', self.__vehicle)

        # Traffic
        if self.__has_traffic:
            self.__spawn_traffic(seed=seed)
        
        self.__world.set_all_traffic_lights_green()

    def clean_scenario(self):
        """Clean up scenario-specific actors like the ego vehicle, traffic, and cones."""
        try:
            self.__vehicle.destroy_vehicle()
        except:
            pass
            
        try:
            self.__world.destroy_world() # This destroys traffic, pedestrians, AND cones
        except:
            pass
        
        self.__cones_spawned = False

    def print_all_scenarios(self):
        for idx, i in enumerate(self.situations_list):
            print(idx, ": ", i)
    
    def __load_world(self, name):
        self.__world.set_active_map(name)
        # Refresh all internal handles (World, Controllers, and Mapper)
        self.__world.sync_world_handles()
        if hasattr(self, '__route_planner'):
            self.__route_planner.update_map(self.__world.get_world())

    def __spawn_vehicle(self, s_dict):
        location = (s_dict['initial_position']['x'], s_dict['initial_position']['y'], s_dict['initial_position']['z'])
        rotation = (s_dict['initial_rotation']['pitch'], s_dict['initial_rotation']['yaw'], s_dict['initial_rotation']['roll'])
        try:
            self.__vehicle.spawn_vehicle(location, rotation)
        except Exception as e:
            if self.__verbose:
                print(f"[Port {self.__port}] Spawn area occupied: {e}. Clearing area and retrying...")
            
            # Instead of map reload (which kills cones), just try to clear spawn point actors
            all_actors = self.__world.get_world().get_actors()
            for actor in all_actors.filter('vehicle.*'):
                if actor.get_location().distance(carla.Location(x=location[0], y=location[1], z=location[2])) < 5.0:
                    try: actor.destroy()
                    except: pass
            
            time.sleep(1.0)
            try:
                self.__vehicle.spawn_vehicle(location, rotation)
            except Exception as e2:
                if self.__verbose: print(f"[Port {self.__port}] Failed second spawn, doing fallback map reload...")
                self.__world.reload_map()
                self.__cones_spawned = False
                time.sleep(2.0)
                self.__vehicle.spawn_vehicle(location, rotation)

    def __toggle_lights(self):
        if self.__force_lights:
            self.__world.toggle_lights(lights_on=True)
            self.__vehicle.toggle_lights(lights_on=True)
            return

        weather = self.__world.get_active_weather().lower()
        if "night" in weather:
            self.__world.toggle_lights(lights_on=True)
            self.__vehicle.toggle_lights(lights_on=True)
        else:
            self.__world.toggle_lights(lights_on=False)
            self.__vehicle.toggle_lights(lights_on=False)
        
    def __load_weather(self, weather_name):
        if self.__weather_override:
            self.__world.set_active_weather_preset(self.__weather_override)
        elif self.__random_weather:
            self.__world.set_random_weather()
        else:
            self.__world.set_active_weather_preset(weather_name)

    def __spawn_traffic(self, seed):
        if not self.__random_traffic and self._active_scenario_dict.get('traffic_density') == 'None':
            return
        
        if not self.__random_traffic:
            random.seed(self._active_scenario_name)
            seed = self._active_scenario_name
        
        if seed is not None:
            random.seed(seed)
        
        if not self.__random_traffic:
            density = self._active_scenario_dict.get('traffic_density', 'Low')
            num_vehicles = random.randint(1, 5) if density == 'Low' else random.randint(10, 20)
        else:
            num_vehicles = random.randint(1, 20)
        
        self.__world.spawn_vehicles_around_ego(self.__vehicle.get_vehicle(), radius=100, num_vehicles_around_ego=num_vehicles, seed=seed)

    def __spawn_all_cones(self):
        if self.__cone_config is None: return
        
        # Self-healing: Check if cones actually exist in the world
        world_cone_count = self.__world.get_cone_count()
        if world_cone_count >= len(self.__cone_config):
            self.__cones_spawned = True
            return

        if self.__verbose:
            print(f"[Port {self.__port}] Cones missing or map reloaded. Spawning {len(self.__cone_config)} cones...")
            
        for cone in self.__cone_config:
            self.__world.spawn_cones_from_positions([{'x': cone['x'], 'y': cone['y'], 'z': cone.get('z', 0.0)}])
        
        self.__cones_spawned = True
        self.__world.tick()

    def get_path_waypoints(self, spacing=2.0, start_loc=None):
        """
        Plans a route that avoids cones using the AdvancedRoutePlanner.
        """
        if start_loc is None:
            start_loc = self.__vehicle.get_location()
        
        target_loc = carla.Location(
            x=self._active_scenario_dict['target_position']['x'], 
            y=self._active_scenario_dict['target_position']['y'], 
            z=self._active_scenario_dict['target_position']['z']
        )
        
        # Freshly check if cones are available to avoid them in planning
        # We also pass the cone configuration to the planner in case it changed
        if self.__cone_config:
            self.__route_planner.set_cones(self.__cone_config)
            
        waypoints = self.__route_planner.plan_route(
            start_loc, 
            target_loc, 
            avoid_cones=True,
            verbose=self.__verbose
        )
        
        if self.__verbose:
            print(f"[Port {self.__port}] Planner returned {len(waypoints)} points.")
            
        # Return as locations list
        if len(waypoints) > 1:
            return waypoints[1:] # Skip the first one as it's at the car's current pose
        return waypoints

    def get_vehicle(self):
        return self.__vehicle

    def place_spectator_above_vehicle(self):
        self.__world.place_spectator_above_location(self.__vehicle.get_location())

    def output_all_waypoints(self, spacing=5):
        waypoints = self.__world.get_map().generate_waypoints(distance=spacing)
        for w in waypoints:
            self.__world.get_world().debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)

    def draw_waypoints(self, waypoints, life_time=10.0):
        for w in waypoints:
            self.__world.get_world().debug.draw_string(w, 'O', draw_shadow=False,
                                                    color=carla.Color(r=255, g=0, b=0), life_time=life_time,
                                                    persistent_lines=True)

    def __start_timer(self):
        self.start_time = time.time()

    def __get_situations(self, scenarios):
        with open(config.ENV_SCENARIOS_FILE, 'r') as f:
            full_situations = json.load(f)
        
        if scenarios:
            # Case-insensitive substring matching for robustness
            filtered_dict = {}
            for k, v in full_situations.items():
                situation_type = v.get('situation', '').lower()
                scenario_name = k.lower()
                
                match = False
                for s in scenarios:
                    s_low = s.lower()
                    # Match if the input is a substring of the scenario name or type
                    if s_low in scenario_name or s_low in situation_type:
                        match = True
                        break
                    # Also try matching without spaces for things like "LaneGuidance" -> "Lane Guidance"
                    if s_low.replace(" ", "") in scenario_name.replace(" ", ""):
                        match = True
                        break
                
                if match:
                    filtered_dict[k] = v
            
            if not filtered_dict:
                print(f"WARNING: No scenarios matched '{scenarios}'. Available: {list(full_situations.keys())}")
                print("Falling back to all scenarios.")
                self.situations_dict = full_situations
            else:
                self.situations_dict = filtered_dict
        else:
            self.situations_dict = full_situations

        self.situations_list = list(self.situations_dict.keys())

    def __choose_random_situation(self, seed=None):
        if seed: np.random.seed(seed)
        return np.random.choice(self.situations_list)

    def __chose_situation(self, seed):
        return self.__choose_random_situation(seed)
