import carla
import numpy as np
import pygame
import cv2

class DebugManager:
    """
    Centralized manager for all debugging features in the CARLA-RL project.
    Toggles features like waypoint drawing, target visualization, sensor reset logging,
    and neural network input inspection.
    """
    
    # Debug Categories
    WAYPOINTS = "waypoints"   # Draw the path waypoints
    TARGET = "target"         # Draw the finish line
    SENSORS = "sensors"       # Log sensor transgressions (line crossing, collision)
    NN_INPUT = "nn"           # Show 224x224 NN input and Rest vector
    REWARD = "reward"         # Detailed reward breakdown per step
    TERMINATION = "term"      # Detailed reason for episode end
    
    def __init__(self, debug_list=[]):
        """
        Initialize with a list of active debug categories.
        Example: debug_list=["waypoints", "target", "term"]
        """
        self.active_features = set(debug_list)
        self.nn_window_initialized = False
        self.screen = None
        self.clock = None
        self.fonts = {}

    def is_active(self, feature):
        return feature in self.active_features or "all" in self.active_features

    def log(self, feature, message):
        if self.is_active(feature):
            print(f"[DEBUG:{feature.upper()}] {message}")

    # --- CARLA World Visuals ---
    
    def draw_waypoints(self, world, waypoints, life_time=10.0):
        if not self.is_active(self.WAYPOINTS) or not waypoints:
            return
        
        for i, w in enumerate(waypoints):
            # Handle both carla.Location objects and numpy arrays
            loc = w
            if isinstance(w, np.ndarray):
                loc = carla.Location(x=float(w[0]), y=float(w[1]), z=float(w[2]))
            
            # Small cyan dot for each breadcrumb waypoint
            world.debug.draw_point(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.25),
                size=0.04,
                color=carla.Color(r=0, g=200, b=255),
                life_time=life_time
            )

    def draw_target(self, world, target_pos, life_time=120.0):
        if not self.is_active(self.TARGET) or target_pos is None:
            return
            
        loc = carla.Location(x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2]))
        
        # Large bright green point at ground level
        world.debug.draw_point(loc, size=0.4, color=carla.Color(r=0, g=255, b=100), life_time=life_time)
        
        # Vertical post: line from ground up to 3m
        top = carla.Location(x=loc.x, y=loc.y, z=loc.z + 3.0)
        world.debug.draw_line(loc, top, thickness=0.08, color=carla.Color(r=0, g=255, b=100), life_time=life_time)
        
        # Label above the post
        world.debug.draw_string(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 3.5),
            '★ GOAL',
            draw_shadow=True,
            color=carla.Color(r=0, g=255, b=100),
            life_time=life_time,
            persistent_lines=False
        )

    def draw_dynamic_distances(self, world, ego_loc, target_pos, nearest_cone_loc):
        if not self.is_active(self.TARGET) and not self.is_active(self.WAYPOINTS):
            return
            
        start_loc = carla.Location(x=ego_loc.x, y=ego_loc.y, z=ego_loc.z + 1.5)
        
        # Final destination
        if target_pos is not None:
            t_loc = carla.Location(x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2]))
            dist_to_target = start_loc.distance(t_loc)
            
            # Highlight final destination
            world.debug.draw_point(t_loc + carla.Location(z=1.5), size=0.3, color=carla.Color(0, 255, 0), life_time=0.1)
            
            # Line to target
            world.debug.draw_line(start_loc, t_loc + carla.Location(z=1.5), thickness=0.05, color=carla.Color(0, 255, 0), life_time=0.1)
            
            # Draw distance centrally
            mid_t = carla.Location(x=(start_loc.x+t_loc.x)/2, y=(start_loc.y+t_loc.y)/2, z=(start_loc.z+t_loc.z)/2 + 2.0)
            world.debug.draw_string(mid_t, f'Target: {dist_to_target:.1f}m', draw_shadow=True, color=carla.Color(0, 255, 0), life_time=0.1)
            
        # Nearest cone
        if nearest_cone_loc is not None:
            c_loc = carla.Location(x=nearest_cone_loc.x, y=nearest_cone_loc.y, z=nearest_cone_loc.z)
            dist_to_cone = start_loc.distance(c_loc)
            
            # Highlight nearest cone
            world.debug.draw_point(c_loc + carla.Location(z=1.5), size=0.3, color=carla.Color(255, 0, 0), life_time=0.1)
            
            # Line to cone
            world.debug.draw_line(start_loc, c_loc + carla.Location(z=1.5), thickness=0.05, color=carla.Color(255, 0, 0), life_time=0.1)
            
            # Draw distance centrally
            mid_c = carla.Location(x=(start_loc.x+c_loc.x)/2, y=(start_loc.y+c_loc.y)/2, z=(start_loc.z+c_loc.z)/2 + 2.0)
            world.debug.draw_string(mid_c, f'Cone: {dist_to_cone:.1f}m', draw_shadow=True, color=carla.Color(255, 0, 0), life_time=0.1)

    # --- NN Input Inspection (Pygame Window) ---
    
    def show_nn_input(self, raw_rgb, rest_vector, last_action=None):
        """Visualizes exactly what the Neural Network sees."""
        if not self.is_active(self.NN_INPUT):
            return

        if not self.nn_window_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("NN Debug View")
            self.clock = pygame.time.Clock()
            self.fonts['small'] = pygame.font.SysFont("Courier New", 12)
            self.fonts['bold'] = pygame.font.SysFont("Courier New", 14, bold=True)
            self.nn_window_initialized = True

        # Process Image (Resize to 224x224 like the architecture does)
        nn_image = cv2.resize(raw_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Render
        self.screen.fill((20, 20, 25))
        
        # 1. Raw Image
        raw_surf = pygame.surfarray.make_surface(raw_rgb.swapaxes(0, 1))
        raw_surf = pygame.transform.scale(raw_surf, (320, 180))
        self.screen.blit(raw_surf, (20, 40))
        self.screen.blit(self.fonts['bold'].render("RAW CAMERA", True, (200, 200, 200)), (20, 20))
        
        # 2. NN Input (224x224)
        nn_surf = pygame.surfarray.make_surface(nn_image.swapaxes(0, 1))
        self.screen.blit(nn_surf, (360, 40))
        pygame.draw.rect(self.screen, (0, 255, 0), (358, 38, 228, 228), 1)
        self.screen.blit(self.fonts['bold'].render("NN INPUT (224x224)", True, (0, 255, 0)), (360, 20))
        
        # 3. Rest Vector textual display
        self.screen.blit(self.fonts['bold'].render("REST VECTOR (23 Features)", True, (255, 150, 0)), (20, 240))
        
        feature_labels = [
            "Dist Target", "Dist Waypoint", "Yaw Error", "Ang Vel Z", 
            "Fwd Vel", "Lat Vel", "Last Steer", "Last Throttle"
        ]
        for i in range(5): feature_labels.extend([f"Cone{i}X", f"Cone{i}Y", f"Cone{i}D"])
        
        for i, val in enumerate(rest_vector):
            col = i // 12
            row = i % 12
            label = feature_labels[i] if i < len(feature_labels) else f"F{i}"
            txt = self.fonts['small'].render(f"{label:<12}: {val:>7.3f}", True, (220, 220, 220))
            self.screen.blit(txt, (20 + col * 380, 270 + row * 18))

        pygame.display.flip()
        
        # Handle Pygame events to prevent window freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.active_features.remove(self.NN_INPUT)
                pygame.quit()
                self.nn_window_initialized = False

    def close(self):
        if self.nn_window_initialized:
            pygame.quit()
            self.nn_window_initialized = False
