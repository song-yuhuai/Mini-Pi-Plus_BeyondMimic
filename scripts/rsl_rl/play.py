"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Ensure local package import works when running script from repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WHOLE_BODY_TRACKING_PATH = REPO_ROOT / "source" / "whole_body_tracking"
if WHOLE_BODY_TRACKING_PATH.exists():
    sys.path.insert(0, str(WHOLE_BODY_TRACKING_PATH))

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--debug_resets",
    action="store_true",
    default=False,
    help="Print reset diagnostics (reasons and key states) whenever an environment resets.",
)
parser.add_argument(
    "--deterministic_start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="For playback only: force motion command resets to start at frame/phase index 0.",
)
# parser.add_argument("--motion_file", type=str, required=True, help="Path to the motion file.")
# parser.add_argument("--resume_path", type=str, required=True, help="Path to the trained model checkpoint.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _tensor_to_bool_tensor(value, device: torch.device) -> torch.Tensor | None:
    """Convert an unknown reason mask object into a 1-D bool tensor when possible."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.bool().unsqueeze(0)
        return value.bool().reshape(-1)
    if isinstance(value, (list, tuple)) and len(value) > 0:
        try:
            return torch.tensor(value, device=device, dtype=torch.bool).reshape(-1)
        except Exception:
            return None
    return None


def _safe_quat_to_rpy(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to roll/pitch/yaw in radians."""
    w, x, y, z = quat_wxyz.unbind(-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)


def _collect_reset_reasons(unwrapped_env, done_mask: torch.Tensor) -> tuple[dict[int, list[str]], dict[str, int]]:
    """Collect per-environment reset reasons from termination manager internals when available."""
    device = done_mask.device
    done_ids = torch.where(done_mask)[0]
    reasons_by_env = {int(i): [] for i in done_ids.tolist()}
    reason_counts: dict[str, int] = {}

    termination_manager = getattr(unwrapped_env, "termination_manager", None)
    if termination_manager is None:
        return reasons_by_env, reason_counts

    reason_map = getattr(termination_manager, "_term_dones", None)
    if not isinstance(reason_map, dict):
        return reasons_by_env, reason_counts

    for reason_name, reason_mask in reason_map.items():
        mask = _tensor_to_bool_tensor(reason_mask, device)
        if mask is None or mask.numel() == 0:
            continue
        for env_id in done_ids.tolist():
            if env_id < mask.numel() and bool(mask[env_id].item()):
                reasons_by_env[int(env_id)].append(str(reason_name))
                reason_counts[str(reason_name)] = reason_counts.get(str(reason_name), 0) + 1

    return reasons_by_env, reason_counts


def _print_reset_debug(unwrapped_env, done_mask: torch.Tensor, infos: dict | None, global_step: int):
    """Print one debug line per reset with reason and key state snapshots."""
    if done_mask.ndim > 1:
        done_mask = done_mask.reshape(-1)
    done_mask = done_mask.bool()
    done_ids = torch.where(done_mask)[0]
    if len(done_ids) == 0:
        return

    reasons_by_env, reason_counts = _collect_reset_reasons(unwrapped_env, done_mask)
    if not reason_counts:
        reason_counts = {"done": int(len(done_ids))}

    robot = None
    try:
        robot = unwrapped_env.scene["robot"]
    except Exception:
        robot = None

    contact_sensor = None
    try:
        contact_sensor = unwrapped_env.scene["contact_forces"]
    except Exception:
        contact_sensor = None

    motion_term = None
    command_manager = getattr(unwrapped_env, "command_manager", None)
    if command_manager is not None and hasattr(command_manager, "get_term"):
        try:
            motion_term = command_manager.get_term("motion")
        except Exception:
            motion_term = None

    for env_id_tensor in done_ids:
        env_id = int(env_id_tensor.item())
        episode_step = None
        if hasattr(unwrapped_env, "episode_length_buf"):
            try:
                episode_step = int(unwrapped_env.episode_length_buf[env_id].item())
            except Exception:
                episode_step = None

        reason_list = reasons_by_env.get(env_id, [])
        if not reason_list:
            reason_list = ["done"]

        base_pos = "n/a"
        base_rpy = "n/a"
        base_vel = "n/a"
        if robot is not None:
            try:
                pos = robot.data.root_pos_w[env_id]
                quat = robot.data.root_quat_w[env_id]
                lin_vel = robot.data.root_lin_vel_w[env_id]
                ang_vel = robot.data.root_ang_vel_w[env_id]
                rpy = _safe_quat_to_rpy(quat)
                base_pos = f"({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                base_rpy = f"({rpy[0]:+.3f},{rpy[1]:+.3f},{rpy[2]:+.3f})"
                base_vel = (
                    f"lin=({lin_vel[0]:+.3f},{lin_vel[1]:+.3f},{lin_vel[2]:+.3f})/"
                    f"ang=({ang_vel[0]:+.3f},{ang_vel[1]:+.3f},{ang_vel[2]:+.3f})"
                )
            except Exception:
                pass

        contacts = "n/a"
        if contact_sensor is not None:
            try:
                net_forces = contact_sensor.data.net_forces_w[env_id]
                norms = torch.norm(net_forces, dim=-1)
                contacts = f"active>1N={int((norms > 1.0).sum().item())},max={float(norms.max().item()):.2f}N"
            except Exception:
                pass

        motion_idx = "n/a"
        start_idx = "n/a"
        if motion_term is not None and hasattr(motion_term, "time_steps"):
            try:
                cur_idx = int(motion_term.time_steps[env_id].item())
                start_idx = int(getattr(motion_term, "phase_start_count", 0))
                end_idx = int(getattr(motion_term, "phase_end_count", -1))
                motion_idx = f"{cur_idx}/{end_idx}"
            except Exception:
                pass

        print(
            f"[RESET] env={env_id} step={global_step} episode_step={episode_step} "
            f"reasons={reason_list} reason_counts={reason_counts} "
            f"base_pos={base_pos} base_rpy={base_rpy} base_vel={base_vel} "
            f"contacts={contacts} motion_idx={motion_idx} phase_start={start_idx}"
        )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # if args_cli.wandb_path:
    #     import wandb
    #
    #     run_path = args_cli.wandb_path
    #
    #     api = wandb.Api()
    #     if "model" in args_cli.wandb_path:
    #         run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
    #     wandb_run = api.run(run_path)
    #     # loop over files in the run
    #     files = [file.name for file in wandb_run.files() if "model" in file.name]
    #     # files are all model_xxx.pt find the largest filename
    #     if "model" in args_cli.wandb_path:
    #         file = args_cli.wandb_path.split("/")[-1]
    #     else:
    #         file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    #
    #     wandb_file = wandb_run.file(str(file))
    #     wandb_file.download("./logs/rsl_rl/temp", replace=True)
    #
    #     print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
    #     resume_path = f"./logs/rsl_rl/temp/{file}"
    #
    #     if args_cli.motion_file is not None:
    #         print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
    #         env_cfg.commands.motion.motion_file = args_cli.motion_file
    #
    #     art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
    #     if art is None:
    #         print("[WARN] No model artifact found in the run.")
    #     else:
    #         env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    # env_cfg.commands.motion.motion_file = "/home/wenconggan/whole_body_tracking/motion/chars.npz"
    # resume_path = "/home/wenconggan/whole_body_tracking/logs/rsl_rl/x2_flat/2025-09-01_11-17-33/model_4000.pt"

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Playback-only deterministic motion start override.
    env_cfg.commands.motion.deterministic_start = args_cli.deterministic_start
    if args_cli.deterministic_start:
        env_cfg.commands.motion.phase_start_count = 0

    print(
        "[INFO] Playback motion start config: "
        f"deterministic_start={env_cfg.commands.motion.deterministic_start}, "
        f"phase_start_count={env_cfg.commands.motion.phase_start_count}, "
        f"phase_end_count={env_cfg.commands.motion.phase_end_count}"
    )

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = None
    if agent_cfg.load_checkpoint is not None:
        potential_path = agent_cfg.load_checkpoint
        if os.path.isabs(potential_path) or os.sep in potential_path:
            if os.path.isfile(potential_path):
                resume_path = os.path.abspath(potential_path)
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {potential_path}")

    if resume_path is None:
        run_dir_expr = agent_cfg.load_run if isinstance(agent_cfg.load_run, str) and len(agent_cfg.load_run) > 0 else ".*"
        checkpoint_expr = (
            agent_cfg.load_checkpoint if (isinstance(agent_cfg.load_checkpoint, str) and os.sep not in agent_cfg.load_checkpoint)
            else ".*"
        )
        resume_path = get_checkpoint_path(log_root_path, run_dir_expr, checkpoint_expr)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    exported_policy_name = resume_path.split('/')[-1]
    exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename=exported_onnx_name,
    )
    attach_onnx_metadata(
        env.unwrapped,
        args_cli.wandb_path if args_cli.wandb_path else "none",
        export_model_dir,
        exported_onnx_name,
        yaml_path=os.path.join(log_dir, "params", "deploy.yaml"),
    )
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    global_step = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, infos = env.step(actions)
            global_step += 1
            if args_cli.debug_resets:
                _print_reset_debug(env.unwrapped, dones, infos, global_step)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
