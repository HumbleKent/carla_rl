import carla

import re
import random

'''
This module provides functions to control weather in the simulator

Currently available weather presets:

'Clear Night'
'Clear Noon'
'Clear Sunset'
'Cloudy Night'
'Cloudy Noon'
'Cloudy Sunset'
'Default'
'Dust Storm'
'Hard Rain Night'
'Hard Rain Noon'
'Hard Rain Sunset'
'Mid Rain Sunset'
'Mid Rainy Night'
'Mid Rainy Noon'
'Soft Rain Night'
'Soft Rain Noon'
'Soft Rain Sunset'
'Wet Cloudy Night'
'Wet Cloudy Noon'
'Wet Cloudy Sunset'
'Wet Night'
'Wet Noon'
'Wet Sunset'
'''

class WeatherControl:
    def __init__(self, world):
        self.__weather_list = self.__get_all_weather_presets()
        self.__active_weather = "Default"
        self.__world = world

    def __get_all_weather_presets(self):
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    
    # The output is a tuple (carla.WeatherPreset, Str: name of the weather preset)
    def get_weather_presets(self):
        return self.__weather_list
    
    def get_active_weather(self):
        return self.__active_weather
    
    def print_all_weather_presets(self):    
        for idx, weather in enumerate(self.__weather_list):
            print(f'{idx}: {weather[1]}')

    def __activate_weather_preset(self, idx):
        self.__world.set_weather(self.__weather_list[idx][0])

    def set_active_weather_preset(self, weather):
        if weather == "No Shadows":
            # Custom settings to eliminate harsh shadows
            w_params = carla.WeatherParameters(
                cloudiness=100.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=0.0,
                sun_azimuth_angle=0.0,
                sun_altitude_angle=90.0, # Directly overhead
                fog_density=0.0,
                fog_distance=0.0,
                fog_falloff=0.0,
                wetness=0.0,
                scattering_intensity=1.0,
                mie_scattering_scale=0.0,
                rayleigh_scattering_scale=0.0331,
                dust_storm=0.0
            )
            # Adjust intensities to diffuse light and remove contrast
            w_params.sun_intensity = 0.0
            w_params.sky_light_intensity = 50.0 
            
            self.__world.set_weather(w_params)
            self.__active_weather = "No Shadows"
            return

        for idx, w in enumerate(self.__weather_list):
            if w[1] == weather:
                self.__active_weather = w[1]
                self.__activate_weather_preset(idx)
                return
    
    def set_random_weather_preset(self):
        idx = random.randint(0, len(self.__weather_list) - 1)
        self.__active_weather = self.__weather_list[idx][1]
        self.__activate_weather_preset(idx)

    # This method let's the user choose with numbers the active preset. It serves as more of a debug.
    def choose_weather(self):
        print('Choose a weather preset:')
        for idx, weather in enumerate(self.__weather_list):
            print(f'{idx}: {weather[1]}')
        
        idx = int(input())

        try:
            self.__active_weather = self.__weather_list[idx][1]
            self.__activate_weather_preset()
        except IndexError:
            print('Invalid index')
            return
        
        print(f'Weather preset {self.__active_weather} activated')
