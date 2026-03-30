# Risk-Bounded Multi-Agent Visual Navigation via Iterative Risk Allocation

[Viraj Parimi](https://people.csail.mit.edu/vparimi/), [Brian Williams](https://www.csail.mit.edu/person/brian-williams)  
Massachusetts Institute of Technology  
**[ICAPS 2026](https://icaps26.icaps-conference.org/)**

**Project:** [rb-safe-visual-mapf-mers.csail.mit.edu](https://rb-safe-visual-mapf-mers.csail.mit.edu/) • **Paper:** [arXiv:2509.08157](https://arxiv.org/abs/2509.08157)

This repository is for the risk-bounded multi-agent visual navigation problem, where, given a set of agents with start locations, goal locations, a map, and a user-specified global risk bound, we compute coordinated trajectories to their respective destinations as quickly as possible using only visual observations such as camera inputs, while dynamically allocating local risk budgets across agents to satisfy the overall safety constraint.  

## Requirements

```sh
conda env create -f environment.yml
```

## Training and Setup

For Habitat-Sim installation, ReplicaCAD setup, environment preparation, training, and visualization workflows, follow the instructions at [safe-visual-mapf-mers.csail.mit.edu](https://safe-visual-mapf-mers.csail.mit.edu/).

This repository reuses the same training setup and launch flow unless noted otherwise below.

## Experimental Reproduction

To reproduce the results as described in the paper, please follow these instructions

1. Download the codebase
2. Export the python path and point it to the root directory of this codebase
```sh
export PYTHONPATH=/path/to/codebase
```
3. Download the models inside the base directory by clicking the following [link](https://nas.mers.csail.mit.edu/sharing/dzVKC2O83)
    - Unzip the models
        ```sh
        unzip models.zip
        ```
4. Run the python illustration notebooks provided in `pud/plots/` to re-generate the plots in the paper
    - Use `plot_safe_pointenv_illustration.ipynb` for 2D Navigation related experiments
    - Use `plot_safe_habitat_illustration.ipynb` for Visual Navigation related experiments
5. Benchmark the approach on different problems
    - Ensure that the script is executable
        ```sh
            chmod u+x pud/plots/collect_all_trajs.sh
        ```
    - Generate your own problems
        ```sh
        pud/plots/collect_all_trajs.sh <env_name> <config_path> <unconstrained_ckpt_path> <constrained_ckpt_path> true 
        ```
    - Collect the new trajectories corresponding to the new problems
        ```sh
            pud/plots/collect_all_trajs.sh <env_name> <config_path> <unconstrained_ckpt_path> <constrained_ckpt_path>
        ```
6. Run the python metrics notebooks provided in `pud/plots/` to re-generate the data used for tables in the paper. Note that you will need to 
    - Use `plot_safe_pointenv_metrics.ipynb` for 2D Navigation related experiments
    - Use `plot_safe_habitat_metrics.ipynb` for Visual Navigation related experiments
7. To use the same data that was used to generate the results in the paper, including the precomputed benchmark and ROS/GZSim problem-set artifacts under `pud/plots/data`, simply download that data using the following [link](https://nas.mers.csail.mit.edu/sharing/ZP7GYgB4s)
    - Unzip the data inside `pud/plots/` directory
        ```sh
        tar -xzvf data.tar.gz -C /path/to/pud/plots/
        ```
    Rerun Step 6.

## ROS GZSim Setup

For ROS/Gazebo simulation experiments, use Ubuntu 22.04 with ROS 2 Humble and Gazebo Harmonic. The commands below assume the simulation path rather than PX4 or real Crazyflies, and that launch commands are run from `pud/gzsim`.

Install the required ROS and Gazebo packages:
```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-bridge \
  ros-humble-rviz2 \
  ros-humble-cv-bridge \
  ros-humble-gazebo-ros-pkgs
```

Before building, ensure that pretrained checkpoints are available under `models/` at the repository root, export the repository root on `PYTHONPATH`, and place ReplicaCAD under `external_data/replica_cad` for Habitat-based runs.

Build the ROS workspace:
```bash
export PYTHONPATH=/path/to/codebase
cd pud/gzsim
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select rbmapf_interfaces rbmapf_gzsim
```

Launch the default simulation:
```bash
cd pud/gzsim
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
ros2 daemon stop
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rbmapf_gzsim multi_vehicle_spawn.launch.py
```

For Habitat experiments, launch the same script with `habitat:=True` and override `num_drones`, `config_file`, `problem_set_file`, `constrained_ckpt_file`, and `unconstrained_ckpt_file` as needed.

The ROS/GZSim launch path also supports `problem_index` for selecting a grouped problem from a precomputed problem set. Some benchmark-generated problems were created for offline evaluation rather than ROS simulation, so a given grouped instance may place starts and goals too close together, causing drones to block each other and make little or no progress. In those cases, keep the same `problem_set_file` and rerun with a different `problem_index`, for example:
```bash
ros2 launch rbmapf_gzsim multi_vehicle_spawn.launch.py \
  habitat:=True \
  problem_index:=7
```

The precomputed problem-set artifacts used by ROS/GZSim live under `pud/plots/data` and are the same benchmark data linked in Experimental Reproduction Step 7 above.

## Extra Notes:
1. The launch scripts can be found in the `launch_jobs/` directory
2. You may visualize individual steps of the approach using the scripts inside `pud/visualizers/` directory
    - Use `gen_graph.py` for 2D Navigation problems
    - Use `gen_visual_nav_graph.py` for Visual Navigation problems


## Results
### 2D Navigation
<img src="pud/plots/figures/paper/safe_single_agent_pointenv_comparison.svg" alt="Visualized 2D Navigation results">

### Visual Navigation
<img src="pud/plots/figures/paper/safe_multi_agent_habitatenv_comparison.svg" alt="Visualized 2D Navigation results">
