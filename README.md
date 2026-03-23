# Mini Pi Plus based on BeyondMimic 

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab/v2.2.0/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)
![gif](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/6363667a0f27da450e1059a30c2b274b.gif)

## Introduction

This project is modified and built based on [Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking). The original project, proposed by author qiayuanl, is a versatile humanoid robot control framework. It can provide highly dynamic motion tracking and state-of-the-art motion quality in real-world deployment, along with controllable test-time control through a guidance diffusion-based controller.

This repo includes motion tracking training, provides assets for HighTorque Robotics robots, and optimizes configuration parameters specifically for HighTorque Robotics robots. **You can train sim-to-real motion using the datasets in the provided motion folder without adjusting any parameters**.

## Installation

Install Isaac Sim v5.0.0, Isaac Lab v2.2.0, and rsl_rl 2.3.1 following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using Conda for installation as it simplifies calling Python scripts from the terminal (Note: Isaac Sim v4.5 and Isaac Lab v2.1.0 environments are also compatible).

- Clone this repository separately from the Isaac Lab installation directory (i.e., outside the `IsaacLab` directory):

```bash
git clone https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic
```

- Install the project using the Conda virtual environment where Isaac Lab is installed. Navigate to the project directory and execute:

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Obtain Data & GMR Retargeting Data
Use the GMR project for dataset retargeting. Original project address: https://github.com/YanjieZe/GMR

```bash
# Create a Conda environment. For avoiding environment conflicts, retargeting is performed in an independent virtual environment
conda deactivate
conda create -n gmr python=3.10 -y
conda activate gmr

pip install -e GMR
conda install -c conda-forge libstdcxx-ng -y
```

**Note: If you are not using the GMR provided in this project but downloaded the content from the original link, modify `numpy==1.24.4` in setup.py before installation.**

**Note: To facilitate your use of HighTorque Robotics robots, we provide CSV-format retargeted data files and converted NPZ-format file templates in `source/motion`. If you need to retarget other files, you can follow the steps below.**

### Motion Preprocessing & Registry Setup

For easy data inspection:

```bash
# Retargeting
python scripts/bvh_to_robot.py --bvh_file MotionData/lafan1/{xxx}.bvh --robot pi_football --save_path RetargetData/lafan1/csv/pi_plus/{xxx}.csv --rate_limit

# Trimming
python scripts/csv_cut.py --input_csv RetargetData/lafan1/csv/pi_plus/pi_plus_dance1_subject2.csv --output_csv RetargetData/lafan1/csv/pi_plus/{xxx}.csv --start_frame {number} --end_frame {number} --remove_frame_column --z_offset 0.00 --decimal_places 6

# NPZ format conversion
conda activate {your_env_isaaclab}

python scripts/csv_to_npz.py --robot pi_plus --input_file source/motion/hightorque/pi_plus/csv/xxx.csv --input_fps 30 --output_name source/motion/hightorque/pi_plus/npz/{motion_name}
# Add --headless if you don't want to load the graphical interface

# Data playback
python scripts/replay_npz.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz 
```
### conda env error

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/isaacsim_fix.sh" <<'SH'
# Prefer conda libstdc++ to satisfy CXXABI_1.3.15 for Isaac Sim / IsaacLab
export _OLD_LD_PRELOAD="$LD_PRELOAD"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
SH

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/isaacsim_fix.sh" <<'SH'
export LD_PRELOAD="$_OLD_LD_PRELOAD"
unset _OLD_LD_PRELOAD
SH
```

### Model Training

Train the policy with the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --headless --log_project_name pi_plus_beyondmimic
# Remove --logger wandb if you don't want to use wandb
# Resume training with --resume {load_run_name}
```

### Model Export

Play the trained policy with the following command:
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-PI-Plus-Wo-v0 --checkpoint {logs_path_to}/model_xxx.pt --num_envs=1 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```
![if](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/e7faf89fbdbf87cf909bbf81ceeb1a7f.gif)

### FB-Zero Off-Policy Workflow (Standalone)

Use this section if you want to train and evaluate the FB-Zero pipeline (`train_fb_zero.py` / `infer_fb_zero.py`) instead of PPO.

#### X2 Task Presets

The X2 flat-ground config currently provides these task presets:

- `Tracking-Flat-X2-v0`: base training task for standard flat-ground training.
- `Tracking-Flat-X2-Robust-v0`: robust training task with reset randomization, COM/material randomization, push disturbances, and joint default offset randomization.
- `Tracking-Flat-X2-Simple-Robust-v0`: conservative sim-to-real training task built on the base setup with light observation noise, mild COM / joint default perturbations, moderate contact friction randomization, and low-frequency push disturbances.
- `Tracking-Flat-X2-Play-v0`: base play task with deterministic phase reset, no early-failure terminations, and no disturbances.
- `Tracking-Flat-X2-Robust-Play-v0`: robust play task that keeps robust randomization/disturbance settings while also using deterministic phase reset and no early-failure terminations.

Typical usage:

- Train a normal policy with `Tracking-Flat-X2-v0`.
- Train a conservative sim-to-real policy with `Tracking-Flat-X2-Simple-Robust-v0`.
- Train a fully robust policy with `Tracking-Flat-X2-Robust-v0`.
- Play or record a normal policy with `Tracking-Flat-X2-Play-v0`.
- Play or stress-test a robust policy with `Tracking-Flat-X2-Robust-Play-v0`.

1. Train FB-Zero from scratch:

```bash
# Paper-style defaults are already set in train_fb_zero.py.
PYTHONUNBUFFERED=1 python scripts/offpolicy/train_fb_zero.py \
  --task=Tracking-Flat-X2-v0 \
  --motion_file=source/motion/x2/npz/step_low_far.npz \
  --headless
```

Optional override examples:

```bash
# Disable compiled update path for debugging
python scripts/offpolicy/train_fb_zero.py --task=Tracking-Flat-X2-v0 --motion_file=source/motion/x2/npz/step_low_far.npz --headless --no-compile-update

# Smaller-GPU profile
python scripts/offpolicy/train_fb_zero.py --task=Tracking-Flat-X2-v0 --motion_file=source/motion/x2/npz/step_low_far.npz --headless --num_envs=256 --batch_size=512 --num_updates=8

# Conservative sim-to-real profile
python scripts/offpolicy/train_fb_zero.py --task=Tracking-Flat-X2-Simple-Robust-v0 --motion_file=source/motion/x2/npz/step_low_far.npz --headless

# Robust profile
python scripts/offpolicy/train_fb_zero.py --task=Tracking-Flat-X2-Robust-v0 --motion_file=source/motion/x2/npz/step_low_far.npz --headless
```

Expected outputs:
- Run directory line, e.g. `logs/offpolicy_fb/{timestamp}`.
- Periodic training logs as metric dictionaries (BFM-zero style) plus heartbeat lines.
- Saved checkpoints under `logs/offpolicy_fb/{run_name}/checkpoints/`.
- Exported `logs/offpolicy_fb/{run_name}/bfm_zero_full_defaults.json` (ported full arch/train preset).
- Default profile is paper-style (`num_envs=1024`, high UTD schedule, `grad_penalty=10`, CUDA replay, compiled updates).
- `--num_iterations` is the primary stop condition when provided.
- `--random_steps` and `--update_every` are measured in env-steps (not collector iterations).
- For smaller GPUs, reduce `--num_envs` first (e.g. 128/256), then tune `--num_updates` and `--batch_size`.
- Expert source knob:
  - `--expert_source=reward|motion|reward_or_motion` (`reward_or_motion` default).
- In-training tracking eval (BFM-style cadence):
  - `--eval_every_steps` and `--eval_steps`
  - runtime prints `[EVAL] ...` and writes `logs/offpolicy_fb/{run_name}/tracking_eval_log.csv`

2. Resume FB-Zero training from a checkpoint:

```bash
python scripts/offpolicy/train_fb_zero.py \
  --task=Tracking-Flat-X2-v0 \
  --motion_file=source/motion/x2/npz/step_low_far.npz \
  --resume logs/offpolicy_fb/{run_name}/checkpoints/final.pt \
  --num_iterations=40000 \
  --total_env_steps=400000
```

3. Infer latent `z` and rollout check (Phase 4 final inference):

```bash
python scripts/offpolicy/infer_fb_zero.py \
  --checkpoint logs/offpolicy_fb/{run_name}/checkpoints/final.pt \
  --task=Tracking-Flat-X2-Play-v0 \
  --motion_file=source/motion/x2/npz/step_low_far.npz \
  --num_envs=1 \
  --mode=tracking \
  --steps=2000
```

Optional latent modes:

```bash
python scripts/offpolicy/infer_fb_zero.py --checkpoint {ckpt} --task=Tracking-Flat-X2-Play-v0 --mode=goal --goal_index=0
python scripts/offpolicy/infer_fb_zero.py --checkpoint {ckpt} --task=Tracking-Flat-X2-Play-v0 --mode=reward --reward_samples=4096 --reward_temperature=10.0
python scripts/offpolicy/infer_fb_zero.py --checkpoint {ckpt} --task=Tracking-Flat-X2-Robust-Play-v0 --motion_file=source/motion/x2/npz/step_low_far.npz --num_envs=1 --mode=tracking
```

Expected outputs:
- Console log with selected mode and latent norm, e.g. `[INFO] mode=tracking, z_norm=1.0000`.
- Console log with rollout reward summary, e.g. `[INFO] Avg reward over 2000 steps: ...`.
- If `--save_z` is set, a JSON file containing the inferred latent vector `z`.

4. Evaluate latent tracking quality:

```bash
python scripts/offpolicy/eval_fb_zero.py \
  --checkpoint logs/offpolicy_fb/{run_name}/checkpoints/final.pt \
  --task=Tracking-Flat-X2-Play-v0 \
  --motion_file=source/motion/x2/npz/step_low_far.npz \
  --mode=tracking \
  --steps=3000 \
  --save_report logs/offpolicy_fb/{run_name}/eval_tracking.json
```

Expected outputs:
- `avg_step_reward`, `done_count`, `episodes_finished`, `avg_episode_return`.
- Mean scalar metrics collected from environment `info` (if available).
- Optional JSON report via `--save_report`.

5. Deploy-like tracking evaluation (sequence-z + gamma/window smoothing):

```bash
python scripts/offpolicy/eval_fb_zero_tracking.py \
  --checkpoint logs/offpolicy_fb/{run_name}/checkpoints/final.pt \
  --task=Tracking-Flat-X2-Robust-Play-v0 \
  --motion_file=source/motion/x2/npz/step_low_far.npz \
  --num_envs=1 \
  --steps=3000 \
  --window_size=3 \
  --gamma=0.8 \
  --save_z_seq logs/offpolicy_fb/{run_name}/z_seq_tracking.json \
  --save_report logs/offpolicy_fb/{run_name}/eval_tracking_seq.json
```

Expected outputs:
- Runtime tracking evaluation with time-varying latent `z_t`.
- `z_t` uses discounted smoothing over recent latent history (`window_size`, `gamma`).
- Optional saved `z` sequence (`--save_z_seq`) and metrics report (`--save_report`).

### Model Evaluation

Validate the policy in Mujoco with the following command:

```bash
python scripts/sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --xml_path source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus/mjcf/pi_20dof.xml --policy_path {logs_path_to}/exported/{model_xxx}.onnx --save_json --loop
# Use --loop to play the policy repeatedly
```
![f](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/de78f3ab232911f9a93e936cb5463164.gif)

## Code Structure

The following is an overview of the project's code structure:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains atomic functions for defining BeyondMimic's MDP. Below is a breakdown of these functions:
    - **`commands.py`**
      Command library for calculating relevant variables based on reference motion, current robot state, and errors.
      Calculations include pose and velocity error computation, initial state randomization, and adaptive sampling.

    - **`rewards.py`**
      Implements DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early termination and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains environment (MDP) hyperparameter configurations for tracking tasks.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains PPO hyperparameters for tracking tasks.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including skeleton parameters, joint stiffness/damping calculations, and action scaling computations.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers extending the project.

#### Original Project Repository
[whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)




# Mini Pi Plus based on BeyondMimic （中文翻译）

## 介绍

此项目是基于 [Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking) 进行修改建立的。原项目是由作者qiayuanl提出的一个多功能的人形机器人控制框架，它能够在实际部署中提供高度动态的运动跟踪和最先进的运动质量，并通过基于引导扩散的控制器提供可控的测试时间控制。

此 repo 涵盖了运动追踪训练，提供了高擎机电机器人的asset，并针对高擎机电机器人进行了配置参数的调优。**您能够
在所提供的motion文件夹下的数据集中训练可模拟到现实的运动，而无需调整任何参数**。

## 安装

按照[安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Sim v5.0.0 、 Isaac Lab v2.2.0 和 rsl_rl 2.3.1。我们建议使用 conda 安装，因为它可以简化从终端调用 Python 脚本的操作（备注:Isaac Sim v4.5,Isaac Lab v2.1.0环境也可以）。

- 将此代码库与 Isaac Lab 安装目录分开克隆（即，克隆到`IsaacLab`目录之外）：

```bash
git clone https://github.com/HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic
```

- 使用安装了 Isaac Lab 的 Conda 虚拟环境安装本项目，进入项目目录后，执行

```bash
python -m pip install -e source/whole_body_tracking
```

## 动作跟踪

### 获取数据，GMR重定向数据
使用GMR项目进行数据集的重定向，原项目地址: https://github.com/YanjieZe/GMR
```
# create conda env 为避免环境冲突等问题，重定向在一个独立的虚拟环境中进行
conda deactivate
conda create -n gmr python=3.10 -y
conda activate gmr

pip install -e GMR
conda install -c conda-forge libstdcxx-ng -y
```


**注：如果没有使用本项目提供的GMR而是是下载的原链接内容，注意安装前修改setup.py 中 numpy==1.24.4。**

**注：为了便于您使用高擎机电的机器人，我们在`source/motion`中提供了csv版本的重定向数据文件和转换好的npz格式文件模板供你使用，若需要重定向其他文件可以自行按照下方步骤执行**

### Motion Preprocessing & Registry Setup

为了便于检查数据内容，

```bash
# 重定向
python scripts/bvh_to_robot.py --bvh_file MotionData/lafan1/{xxx}.bvh --robot pi_football --save_path RetargetData/lafan1/csv/pi_plus/{xxx}.csv --rate_limit

# 裁剪
python scripts/csv_cut.py --input_csv RetargetData/lafan1/csv/pi_plus/pi_plus_dance1_subject2.csv --output_csv RetargetData/lafan1/csv/pi_plus/{xxx}.csv --start_frame {number} --end_frame {number} --remove_frame_column --z_offset 0.00 --decimal_places 6

# npz格式转换
conda activate {your_env_isaaclab}

python scripts/csv_to_npz.py --robot pi_plus --input_file source/motion/hightorque/pi_plus/csv/xxx.csv --input_fps 30 --output_name source/motion/hightorque/pi_plus/npz/{motion_name}
#若不想加载图形化界面则添加参数 --headless

# 数据播放
python scripts/replay_npz.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz 
```

### X2 任务用法

当前 X2 平地配置提供以下 5 个 task：

- `Tracking-Flat-X2-v0`
  - 基础训练任务，适合常规 flat-ground 训练。
- `Tracking-Flat-X2-Robust-v0`
  - 鲁棒训练任务，包含 reset 随机化、质心随机化、接触材质随机化、push 扰动和关节默认偏置随机化。
- `Tracking-Flat-X2-Simple-Robust-v0`
  - 保守版 sim2real 训练任务，基于 base 配置，只加入轻量观测噪声、轻微质心与关节零位扰动、适度接触摩擦随机化和低频 push 扰动。
- `Tracking-Flat-X2-Play-v0`
  - 基础回放任务，使用固定 phase reset，关闭提前终止和扰动，适合看动作效果、录视频和调试。
- `Tracking-Flat-X2-Robust-Play-v0`
  - 鲁棒回放任务，保留 robust 的随机化和扰动，同时关闭提前终止并固定 phase reset，适合观察 robust 策略在扰动下的表现。

推荐使用方式：

- 训练普通策略：`Tracking-Flat-X2-v0`
- 训练保守版 sim2real 策略：`Tracking-Flat-X2-Simple-Robust-v0`
- 训练鲁棒策略：`Tracking-Flat-X2-Robust-v0`
- 回放普通策略：`Tracking-Flat-X2-Play-v0`
- 回放或压测鲁棒策略：`Tracking-Flat-X2-Robust-Play-v0`

### 模型训练

通过以下命令训练策略：

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-PI-Plus-Wo-v0 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --headless --log_project_name pi_plus_beyondmimic
#若不想使用wandb，可以去掉 --logger wandb
#继续训练 --resume {load_run_name}

# 其他可用参数：
--save_interval=10 
--experiment_name={experiment_name}
--log_dir_path={log_dir_path}
如果提供了--experiment_name和--run_name, log保存路径会是：logs/rsl_rl/{agent_cfg.experiment_name}/{"%Y-%m-%d_%H-%M-%S"}_{run_name}
如果提供了--log_dir_path就会保存在log_dir_path下
```
### 模型导出

通过以下命令play训练好的策略：
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-PI-Plus-Wo-v0 --checkpoint {logs_path_to}/model_xxx.pt --num_envs=1 --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz
```
![if](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/e7faf89fbdbf87cf909bbf81ceeb1a7f.gif)

### 模型评估

通过以下命令在mujoco中验证策略：

```bash
python scripts/sim2sim.py --robot pi_plus --motion_file source/motion/hightorque/pi_plus/npz/{motion_name}.npz --xml_path source/whole_body_tracking/whole_body_tracking/assets/hightorque/pi_plus/mjcf/pi_20dof.xml --policy_path {logs_path_to}/exported/{model_xxx}.onnx --save_json --loop
#使用--loop可以循环播放策略
```
![f](https://github.com/Daily-study-HT/bydmimic_publish/blob/main/gif/de78f3ab232911f9a93e936cb5463164.gif)
## 代码结构

以下是此项目的代码结构概述：

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  此目录包含用于定义 BeyondMimic 的 MDP 的原子函数。以下是这些函数的细分：
    - **`commands.py`**
      命令库用于根据参考运动、当前机器人状态和误差计算相关变量。
      计算包括位姿和速度误差计算、初始状态随机化和自适应采样。

    - **`rewards.py`**
      实现 DeepMimic 奖励函数和平滑项。

    - **`events.py`**
      实现域随机化项。

    - **`observations.py`**
      实现运动跟踪和数据收集的观测项。

    - **`terminations.py`**
      实现提前终止和超时。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  包含跟踪任务的环境（MDP）超参数配置。

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  包含跟踪任务的 PPO 超参数。

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  包含机器人特定设置，包括骨架参数、关节刚度/阻尼计算和动作比例计算。

- **`scripts`**
  包括用于预处理运动数据、训练策略和评估训练策略的实用脚本。

该结构旨在确保开发人员扩展项目的模块化和易于寻找。


## 原项目地址
[whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
