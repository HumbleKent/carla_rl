# Quick Start Reference
Common commands for developing and running CARLA RL Agents.
---
## Start CARLA Servers
## Automation (Recommended)
Use `run.ps1` to handle server launch, scenario selection, and training in one step:
```powershell
.\run.ps1
```
The script will:
1. Prompt for a **run name**
2. Show a numbered **scenario menu** (read automatically from `env/cone_spawn.json`)
3. Show a **task menu** — single server, parallel servers, evaluate, or exit
---
## Start CARLA Servers (Manual)
```bash
# Training server — Port 2000 
.\CarlaUE4.exe -carla-rpc-port=2000
# Second training server — Port 3000 
.\CarlaUE4.exe -carla-rpc-port=3000
```
---
## Training
## Training (Manual)
```bash
# Single server training
python train_cone_avoidance.py --ports 2000 --run-name <run_name>
# Parallel training across two servers
python train_cone_avoidance.py --ports 2000 3000 --run-name <run_name>
# Train on a specific scenario only
python train_cone_avoidance.py --ports 2000 --run-name <run_name> --scenario "Lane Guidance"
# Resume from a specific checkpoint
python train_cone_avoidance.py \
    --ports 2000 3000 \
    --run-name <run_name> \
    --load-path <load_path>
```
---
## Evaluation
```bash
# Evaluate best model from a run (10 episodes)
python evaluate_best_model.py --name <run_name> --episodes 20
```
---
## Debug Session
```bash
# Visual debug session (Pygame overlays: waypoints, target, termination)
python debug_session.py --name <run_name> --episodes 5
python utils/debug_session.py --name <run_name> --episodes 5
```
---
## Monitor training with TensorBoard
```bash
tensorboard --logdir ./tensorboard_cone/
```
---
## Tips
- The `--ports` flag accepts multiple ports: `-p 2000 3000`
- Use `--scenario <scenario_name>` to restrict training to one scenario
- Use `--camera bev` to switch to Bird's Eye View camera
- Checkpoints are saved every `--save-step-freq` steps (default: 20,000)
- The evaluation callback runs on a dedicated `--eval-port` (default: 4000)
