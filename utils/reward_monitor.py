"""
reward_monitor.py
-----------------
Live terminal dashboard that displays a per-episode reward breakdown for two parallel
training workers (Port 2000 and Port 3000) side by side in real time.

It reads structured reward log files written by the training environment
(`logs_cone/reward_port_<port>.log`) and parses each episode's reward breakdown
sections (e.g. Distance Progress, Cone Proximity, Steering Jerk) then renders them
in a colour-coded two-column terminal view, refreshing every 2 seconds.

Useful for:
  - Monitoring which reward components are dominating during training
  - Quickly spotting reward hacking or collapse across parallel workers
  - Comparing training dynamics between two CARLA server instances

Usage:
    python utils/reward_monitor.py

Run from the project root directory while a training session is active.
Requires log files at `logs_cone/reward_port_2000.log` and `logs_cone/reward_port_3000.log`.
"""

import time
import os

LOG_DIR = "logs_cone"
PORTS = [2000, 3000]

class RewardMonitor:
    def __init__(self):
        self.log_dir = LOG_DIR
        self.ports = PORTS

    def parse_reward_log(self, lines):
        """Parses the detailed reward breakdown from log lines."""
        episodes = []
        current_ep = None
        in_breakdown = False
        
        for line in lines:
            if "REWARD BREAKDOWN" in line:
                current_ep = {"sections": {}, "total": 0.0, "reason": "Unknown", "steps": 0}
                in_breakdown = True
                continue
            
            if in_breakdown:
                if "------" in line:
                    continue
                if "TOTAL SCORE" in line:
                    try:
                        current_ep["total"] = float(line.split(":")[1].strip())
                    except: pass
                    continue
                if "====" in line:
                    in_breakdown = False
                    if current_ep: episodes.append(current_ep)
                    continue
                
                if ":" in line:
                    parts = line.split(":")
                    name = parts[0].strip()
                    try:
                        val = float(parts[1].strip())
                        current_ep["sections"][name] = val
                    except: pass
            
            if "Episode" in line and "End" in line:
                try:
                    reason_part = line.split("Reason:")[1].split("|")[0].strip()
                    if current_ep: 
                        current_ep["reason"] = reason_part
                except: pass

        return episodes

    def draw_dashboard(self, port_logs):
        """Draws a side-by-side dashboard for multiple ports."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{' CARLA RL TRAINING MONITOR ':=^100}")
        print(f"{'Port 2000':^48} | {'Port 3000':^48}")
        print("-" * 100)
        
        ep2000 = port_logs.get(2000, [])
        ep3000 = port_logs.get(3000, [])
        
        latest_2000 = ep2000[-1] if ep2000 else None
        latest_3000 = ep3000[-1] if ep3000 else None
        
        categories = [
            "Distance Progress", "Waypoint Progress", "Lane Centering",
            "Speed Policy", "Goal reach Bns", "Cone Proximity",
            "Steering Jerk", "Throttle Jerk", "Stand-Still",
            "Collision Pnlty", "Solid Line Pnlty", "Cone Hit Pnlty", 
            "Suicide Pnlty", "Survival Steps"
        ]
        
        for cat in categories:
            val1 = latest_2000["sections"].get(cat, 0.0) if latest_2000 else 0.0
            val2 = latest_3000["sections"].get(cat, 0.0) if latest_3000 else 0.0
            
            c1 = "\033[92m" if val1 > 0 else ("\033[91m" if val1 < 0 else "")
            c2 = "\033[92m" if val2 > 0 else ("\033[91m" if val2 < 0 else "")
            reset = "\033[0m"
            
            print(f"{cat:<20}: {c1}{val1:>10.2f}{reset}    |    {cat:<20}: {c2}{val2:>10.2f}{reset}")

        print("-" * 100)
        total1 = latest_2000["total"] if latest_2000 else 0.0
        total2 = latest_3000["total"] if latest_3000 else 0.0
        print(f"{'TOTAL SCORE':<20}: {total1:>10.2f}    |    {'TOTAL SCORE':<20}: {total2:>10.2f}")
        
        reason1 = (latest_2000["reason"][:45] if latest_2000 else "N/A")
        reason2 = (latest_3000["reason"][:45] if latest_3000 else "N/A")
        print(f"Status: {reason1:<40} | Status: {reason2:<40}")
        print("=" * 100)
        print(f"Update Time: {time.strftime('%H:%M:%S')} | Eps: {len(ep2000)} (P2) / {len(ep3000)} (P3)")

    def run(self):
        print("Starting Monitor... (Searching for logs in logs_cone/)")
        while True:
            port_logs = {}
            for port in [2000, 3000]:
                log_file = os.path.join("logs_cone", f"reward_port_{port}.log")
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        port_logs[port] = self.parse_reward_log(lines)
            
            if port_logs:
                self.draw_dashboard(port_logs)
            
            time.sleep(2)

if __name__ == "__main__":
    monitor = RewardMonitor()
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
