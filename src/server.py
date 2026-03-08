import os
import subprocess
import time

'''
Server Module

This module contains the CarlaServer class that is responsible for starting and stopping the Carla server.

Requirements:
    - Environment variable CARLA_SERVER that contains the path to the Carla server directory
'''

class CarlaServer:
    @staticmethod
    def initialize_server(port=2000, low_quality = False, offscreen_rendering = False, silent = False, sleep_time = 10):
        # Get environment variable CARLA_SERVER that contains the path to the Carla server directory
        carla_server = os.getenv('CARLA_SERVER')

        # Fallback to local path if not set (assuming standard CARLA structure)
        if carla_server is None:
            # We are in .../PythonAPI/examples/CARLA-RL-Agents/
            # Go up 3 levels to get to WindowsNoEditor/
            candidate = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
            if os.path.exists(os.path.join(candidate, "CarlaUE4.exe")) or os.path.exists(os.path.join(candidate, "CarlaUE4.sh")):
                carla_server = candidate
                if not silent:
                    print(f"CARLA_SERVER env var not found. Using detected path: {carla_server}")

        if carla_server is None:
            raise RuntimeError("Environment variable 'CARLA_SERVER' is not set and could not be detected. Please set it to the CARLA root directory.")

        # If it is Unix add the CarlaUE4.sh to the path else add CarlaUE4.exe
        if os.name == 'posix':
            carla_exe = os.path.join(carla_server, 'CarlaUE4.sh')
            command = f'bash "{carla_exe}" -carla-rpc-port={port} {"--quality-level=Low" if low_quality else ""} {"--RenderOffScreen" if offscreen_rendering else ""}'
        else:
            carla_exe = os.path.join(carla_server, 'CarlaUE4.exe')
            command = f'"{carla_exe}" -carla-rpc-port={port} {"--quality-level=Low" if low_quality else ""} {"--RenderOffScreen" if offscreen_rendering else ""}'

        # Run the command
        if not silent:
            print('Starting Carla server, please wait...')
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
        # Wait for the server to start
        time.sleep(sleep_time)
        if not silent:
            print('Carla server started')

        return process
    
    @staticmethod
    def close_server(process, silent = False):
        if os.name == 'posix':
            os.killpg(os.getpgid(process.pid), 15)
            if not silent:
                print('Carla server closed')
        else:
            # On Windows, use taskkill to terminate the process and all its children
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not silent:
                print('Carla server closed')
    
    @staticmethod
    def restart_server(process, low_quality = False, offscreen_rendering = False, silent = False, sleep_time = 10):
        CarlaServer.close_server(process, silent)
        return CarlaServer.initialize_server(low_quality, offscreen_rendering, silent, sleep_time)
    
    @staticmethod
    def kill_carla_linux():
        if os.name == 'posix':
            os.system('pkill -9 -f CarlaUE4')
            print('Carla server closed')
        else:
            print('This method is only for Unix systems! Please close the Carla server manually.')
