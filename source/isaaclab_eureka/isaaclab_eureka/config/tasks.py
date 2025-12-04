# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for the tasks supported by Isaac Lab Eureka.

`TASKS_CFG` is a dictionary that maps task names to their configuration. Each task configuration
is a dictionary that contains the following keys:

- `description`: A description of the task.
- `success_metric`: A Python expression that computes the success metric for the task.
- `success_metric_to_win`: The threshold for the success metric to win the task and stop.
- `success_metric_tolerance`: The tolerance for the success metric to consider the task successful.
"""

TASKS_CFG = {
    "Isaac-Cartpole-Direct-v0": {
        "description": "balance a pole on a cart so that the pole stays upright",
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.01,
    },
    "Isaac-Velocity-Flat-Unitree-Go1-v0": {
        "description": (
            "Make the Unitree Go1 quadruped track a commanded forward velocity of 2.0 m/s on flat ground. "
            "Prioritize steady locomotion with minimal action rate, keep the torso near 0.34 m, maintain an "
            "upright orientation, and avoid hitting joint limits."
        ),
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.02,
    },
    "Isaac-Velocity-Rough-Unitree-Go1-v0": {
        "description": (
            "Make the Unitree Go1 traverse rough terrain while tracking a forward velocity command of 2.0 m/s. "
            "Aim for smooth, transferable gaits that keep the torso near 0.34 m, aligned with gravity, and avoid "
            "banging into joint limits."
        ),
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.02,
    },
    "Isaac-Velocity-Flat-OpenMutt-v0": {
        "description": (
            "Make the OpenMutt quadruped track a commanded forward velocity between 0.6 and 1.0 m/s on flat terrain. "
            "Aim for smooth, transferable gaits that keep the torso aligned with gravity and limit excessive joint "
            "motion."
        ),
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.02,
    },
    "Isaac-Velocity-Rough-OpenMutt-v0": {
        "description": (
            "Make the OpenMutt quadruped track a commanded forward velocity between 0.6 and 1.0 m/s across rough "
            "terrain. Aim for smooth, transferable gaits that keep the torso aligned with gravity and limit excessive "
            "joint motion."
        ),
        "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.02,
    },
    "Isaac-Quadcopter-Direct-v0": {
        "description": (
            "Bring the quadcopter to the target position self._desired_pos_w while keeping the flight smooth."
        ),
        "success_metric": (
            "torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        "success_metric_tolerance": 0.2,
    },
}
