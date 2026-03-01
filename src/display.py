'''
Display Module:
    It provides the functionality to display the sensor data in a window using Pygame.
'''
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import configuration


class Display:
    def __init__(self, title, vehicle):
        # Store a live reference to the Vehicle object, NOT a snapshot of the sensor dict.
        # Vehicle.destroy_vehicle() does `del self.__sensor_dict` then replaces it with a new
        # empty dict, so any snapshot taken at construction would become a dangling reference
        # after the first episode ends. By storing Vehicle, we always call get_sensor_dict()
        # at render time to get the current (post-respawn) sensors.
        self.__vehicle = vehicle
        self.__non_displayable_sensors = ['gnss', 'imu', 'collision', 'lane_invasion']
        self.__sensor_window_dict = {}
        self.__main_screen = self.__initialize_pygame_window(title)
        self.__clock = pygame.time.Clock()

    def __initialize_pygame_window(self, title):
        pygame.init()
        pygame.display.set_caption(title)

        # CRITICAL: pygame.display.set_mode() MUST be called before any pygame.Surface()
        # or pygame.event.get(). Creating a Surface before set_mode() leaves the display
        # subsystem in an inconsistent state, causing "video system not initialized" errors.
        main_screen = pygame.display.set_mode((configuration.IM_WIDTH, configuration.IM_HEIGHT))

        # Now it is safe to build the per-sensor sub-surfaces
        self.__rebuild_sensor_windows()

        return main_screen

    def __rebuild_sensor_windows(self):
        """Rebuild Surface objects from the vehicle's current sensor dict.

        Called at init and automatically when a sensor dict change is detected
        (which happens every time the vehicle is destroyed and respawned, because
        Vehicle.destroy_vehicle() does `del self.__sensor_dict` followed by
        `self.__sensor_dict = {}`, creating a brand-new dict object).
        """
        sensor_dict = self.__vehicle.get_sensor_dict()
        displayable = [s for s in sensor_dict if s not in self.__non_displayable_sensors]
        n = max(len(displayable), 1)
        w = max(configuration.IM_WIDTH // n, 64)
        h = max(configuration.IM_HEIGHT // 2, 64)

        self.__sensor_window_dict = {}
        for sensor in displayable:
            self.__sensor_window_dict[sensor] = pygame.Surface((w, h))

    def refresh_sensors(self):
        """Manually trigger a sensor surface rebuild (e.g. after a vehicle respawn)."""
        self.__rebuild_sensor_windows()

    # --------------------------------------------------------------------------
    # play_window: Standalone loop (not suitable for use inside a training loop)
    # --------------------------------------------------------------------------
    def play_window(self):
        clock = pygame.time.Clock()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.__render()
                clock.tick(configuration.SENSOR_FPS)
        finally:
            pygame.quit()
            print('Display window closed!')

    # --------------------------------------------------------------------------
    # play_window_tick: Single-frame render — call this from the training loop
    # --------------------------------------------------------------------------
    def play_window_tick(self):
        # Safety guard: if pygame display subsystem was quit for any reason, bail out
        if not pygame.display.get_init():
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Auto-detect sensor dict replacement (after vehicle respawn) by comparing
        # current displayable sensor names against what we have surfaces for
        sensor_dict = self.__vehicle.get_sensor_dict()
        current_displayable = {s for s in sensor_dict if s not in self.__non_displayable_sensors}
        if current_displayable != set(self.__sensor_window_dict.keys()):
            self.__rebuild_sensor_windows()

        self.__render()
        self.__clock.tick(configuration.SENSOR_FPS)

    # --------------------------------------------------------------------------
    # Internal: draw one frame onto the main screen and flip
    # --------------------------------------------------------------------------
    def __render(self):
        sensor_dict = self.__vehicle.get_sensor_dict()
        self.__main_screen.fill((127, 127, 127))

        for idx, sensor in enumerate(self.__sensor_window_dict):
            sub_surface = self.__sensor_window_dict[sensor]
            w, h = sub_surface.get_size()

            row_idx = idx // configuration.NUM_COLS
            col_idx = idx % configuration.NUM_COLS

            x = configuration.MARGIN + col_idx * (w + configuration.MARGIN)
            y = configuration.MARGIN + row_idx * (h + configuration.MARGIN)

            # Border
            pygame.draw.rect(
                self.__main_screen, (50, 50, 50),
                (x - configuration.BORDER_WIDTH, y - configuration.BORDER_WIDTH,
                 w + 2 * configuration.BORDER_WIDTH,
                 h + 2 * configuration.BORDER_WIDTH),
                configuration.BORDER_WIDTH
            )

            # Background tile
            self.__main_screen.blit(sub_surface, (x, y))

            # Live camera / sensor frame
            if sensor in sensor_dict and sensor_dict[sensor].get_last_data() is not None:
                try:
                    frame = pygame.surfarray.make_surface(
                        sensor_dict[sensor].get_last_data().swapaxes(0, 1)
                    )
                    self.__main_screen.blit(frame, (x, y))
                except Exception:
                    pass  # Skip bad frames rather than crashing

            # Label
            font = pygame.font.Font(None, 24)
            label = font.render(sensor.capitalize(), True, (255, 255, 255))
            self.__main_screen.blit(label, (x + 10, y + h - 30))

        # GNSS overlay
        if 'gnss' in sensor_dict and sensor_dict['gnss'].get_last_data() is not None:
            gnss = sensor_dict['gnss'].get_last_data()
            text = (f"GNSS: Lat {gnss.latitude:.6f}  "
                    f"Lon {gnss.longitude:.6f}  "
                    f"Alt {gnss.altitude:.6f}")
            surf = pygame.font.Font(None, 24).render(text, True, (255, 255, 255))
            self.__main_screen.blit(surf, (configuration.MARGIN, configuration.IM_HEIGHT - configuration.MARGIN))

        # IMU overlay
        if 'imu' in sensor_dict and sensor_dict['imu'].get_last_data() is not None:
            imu = sensor_dict['imu'].get_last_data()
            text = (f"IMU Accel: {imu.accelerometer.x:.3f}, {imu.accelerometer.y:.3f}, {imu.accelerometer.z:.3f}  "
                    f"Gyro: {imu.gyroscope.x:.3f}, {imu.gyroscope.y:.3f}, {imu.gyroscope.z:.3f}  "
                    f"Compass: {imu.compass:.3f}")
            surf = pygame.font.Font(None, 24).render(text, True, (255, 255, 255))
            rect = surf.get_rect()
            rect.topleft = (configuration.IM_WIDTH - rect.width - configuration.MARGIN,
                            configuration.IM_HEIGHT - configuration.MARGIN)
            self.__main_screen.blit(surf, rect)

        pygame.display.flip()

    def close_window(self):
        pygame.quit()
        print('Display window closed!')