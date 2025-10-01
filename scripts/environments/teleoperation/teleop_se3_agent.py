# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", choices=['keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'], help="Device for interacting with environment")
parser.add_argument("--port", type=str, default='/dev/ttyACM0', help="Port for the teleop device:so101leader, default is /dev/ttyACM0")
parser.add_argument("--left_arm_port", type=str, default='/dev/ttyACM0', help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0")
parser.add_argument("--right_arm_port", type=str, default='/dev/ttyACM1', help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")

# huggingface upload params
parser.add_argument("--hf_repo_id", type=str, default=None, help="Hugging Face dataset repo id (e.g. username/repo)")
parser.add_argument("--hf_private", action="store_true", help="Create or use a private repo on Hugging Face")
parser.add_argument("--hf_branch", type=str, default="main", help="Branch to push to on Hugging Face")
parser.add_argument("--hf_upload_on_success", action="store_true", help="Automatically upload after each successful demo reset")
parser.add_argument("--hf_cli_path", type=str, default="huggingface-cli", help="Path to huggingface-cli executable")
parser.add_argument("--hf_convert_to_lerobot", action="store_true", help="Convert HDF5 to LeRobot format before upload")
parser.add_argument("--hf_lerobot_fps", type=int, default=30, help="FPS for LeRobot dataset conversion")
parser.add_argument("--hf_lerobot_task", type=str, default="Bimanual manipulation task", help="Task description for LeRobot dataset")

# runtime-configurable env geometry
parser.add_argument("--arm_gap", type=float, default=None, help="Gap between left and right arm bases (meters)")
parser.add_argument("--center_pos", type=float, nargs=3, default=None, help="Center position x y z for arm midpoint")
parser.add_argument("--top_cam_pos", type=float, nargs=3, default=None, help="Top camera world position x y z")
parser.add_argument("--top_cam_quat_ros", type=float, nargs=4, default=None, help="Top camera ROS quaternion w x y z")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg, DatasetExportMode

from leisaac.devices import Se3Keyboard, SO101Leader, BiSO101Leader
from leisaac.devices.keyboard.bi_se3_keyboard import BiSe3Keyboard
from leisaac.enhance.managers import StreamingRecorderManager, EnhanceDatasetExportMode
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
import subprocess
import datetime
import tempfile
import shutil
import h5py
import numpy as np


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def convert_hdf5_to_lerobot(hdf5_path, output_name, fps=30, task_description="Bimanual manipulation task"):
    """Convert HDF5 dataset to LeRobot format."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("LeRobot not installed. Install with: pip install lerobot")
        return hdf5_path
    
    # Create temp directory for conversion
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{output_name}.hdf5")
    
    try:
        # Create LeRobot dataset
        dataset = LeRobotDataset.create(
            repo_id="temp_conversion",  # Will be overwritten
            fps=fps,
            robot_type="bi_so101_follower",
            features={
                "left": "image",
                "right": "image", 
                "top": "image",
                "left_arm_action": "tensor",
                "left_gripper_action": "tensor",
                "right_arm_action": "tensor", 
                "right_gripper_action": "tensor",
            }
        )
        
        # Process HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            demo_names = list(f['data'].keys())
            print(f"Converting {len(demo_names)} demos to LeRobot format...")
            
            for demo_name in demo_names:
                demo_group = f['data'][demo_name]
                
                # Skip failed demos
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    continue
                
                # Extract data
                left_images = demo_group['left'][:]
                right_images = demo_group['right'][:]
                top_images = demo_group['top'][:]
                left_arm_actions = demo_group['left_arm_action'][:]
                left_gripper_actions = demo_group['left_gripper_action'][:]
                right_arm_actions = demo_group['right_arm_action'][:]
                right_gripper_actions = demo_group['right_gripper_action'][:]
                
                # Create frames
                num_steps = len(left_images)
                for i in range(num_steps):
                    frame = {
                        "left": left_images[i],
                        "right": right_images[i],
                        "top": top_images[i],
                        "left_arm_action": left_arm_actions[i],
                        "left_gripper_action": left_gripper_actions[i],
                        "right_arm_action": right_arm_actions[i],
                        "right_gripper_action": right_gripper_actions[i],
                    }
                    dataset.add_frame(frame=frame, task=task_description)
                
                dataset.save_episode()
        
        # Save to temp file
        dataset.save(output_path)
        print(f"LeRobot conversion complete: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"LeRobot conversion failed: {e}")
        return hdf5_path
    finally:
        # Clean up temp dataset
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = 'FXAA'
        env_cfg.sim.render.rendering_mode = 'quality'

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert args_cli.teleop_device in ["bi-so101leader", "bi-keyboard"], "only support bi-so101leader or bi-keyboard for bi-arm task"

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if args_cli.record:
        if args_cli.resume:
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            assert os.path.exists(args_cli.dataset_file), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(args_cli.dataset_file), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    else:
        env_cfg.recorders = None

    # apply runtime geometry overrides if using custom bi-arm env
    try:
        if args_cli.task and "CustomBiArm" in args_cli.task:
            if args_cli.arm_gap is not None:
                env_cfg.arm_gap = float(args_cli.arm_gap)
            if args_cli.center_pos is not None:
                env_cfg.center_pos = tuple(args_cli.center_pos)
            if args_cli.top_cam_pos is not None:
                env_cfg.top_cam_pos = tuple(args_cli.top_cam_pos)
            if args_cli.top_cam_quat_ros is not None:
                env_cfg.top_cam_quat_ros = tuple(args_cli.top_cam_quat_ros)
    except Exception as e:
        print(f"Warning: failed to apply geometry overrides: {e}")

    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = 'lzf'

    # create controller
    if args_cli.teleop_device == "keyboard":
        teleop_interface = Se3Keyboard(env, sensitivity=0.25 * args_cli.sensitivity)
    elif args_cli.teleop_device == "so101leader":
        teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "bi-so101leader":
        teleop_interface = BiSO101Leader(env, left_port=args_cli.left_arm_port, right_port=args_cli.right_arm_port, recalibrate=args_cli.recalibrate)
    elif args_cli.teleop_device == "bi-keyboard":
        teleop_interface = BiSe3Keyboard(env, sensitivity=0.25 * args_cli.sensitivity)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)

    def upload_to_hub():
        if not args_cli.hf_repo_id:
            print("Hugging Face repo id not provided; skip upload.")
            return
        
        # Create timestamped filename to avoid overwrites
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
        timestamped_name = f"{base_name}_{timestamp}"
        
        repo = args_cli.hf_repo_id
        branch = args_cli.hf_branch
        private_flag = "--private" if args_cli.hf_private else ""
        cli = args_cli.hf_cli_path
        
        try:
            # create repo if not exists
            subprocess.run(f"{cli} repo create {repo} {private_flag} --type dataset --non-interactive", shell=True, check=False)
            
            if args_cli.hf_convert_to_lerobot:
                # Convert to LeRobot format first
                print("Converting HDF5 to LeRobot format...")
                lerobot_file = convert_hdf5_to_lerobot(args_cli.dataset_file, timestamped_name, args_cli.hf_lerobot_fps, args_cli.hf_lerobot_task)
                upload_file = lerobot_file
                print(f"Converted to LeRobot format: {lerobot_file}")
            else:
                upload_file = args_cli.dataset_file
            
            # Upload the file
            subprocess.run(f"{cli} upload {repo} {upload_file} --repo-type dataset --revision {branch}", shell=True, check=True)
            print(f"Uploaded dataset to Hugging Face: {os.path.basename(upload_file)}")
            
        except Exception as e:
            print(f"Upload failed: {e}")

    teleop_interface.add_callback("U", upload_to_hub)
    print(teleop_interface)

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    env.reset()
    teleop_interface.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
            actions = teleop_interface.advance()
            if should_reset_task_success:
                print("Task Success!!!")
                should_reset_task_success = False
                if args_cli.record:
                    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)))
                    env.termination_manager.compute()
            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                if start_record_state:
                    if args_cli.record:
                        print("Stop Recording!!!")
                    start_record_state = False
                if args_cli.record:
                    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
                    env.termination_manager.compute()
                # print out the current demo count if it has changed
                if args_cli.record and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if args_cli.record and args_cli.hf_upload_on_success:
                    upload_to_hub()
                if args_cli.record and args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break

            elif actions is None:
                env.render()
            # apply actions
            else:
                if not start_record_state:
                    if args_cli.record:
                        print("Start Recording!!!")
                    start_record_state = True
                env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
