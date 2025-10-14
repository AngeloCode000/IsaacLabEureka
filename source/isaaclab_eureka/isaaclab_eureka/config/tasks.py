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
            "To make the go1 quadruped run forward with a velocity of exactly 2.0 m/s in the positive x "
            "direction of the global coordinate frame. The policy will be trained in simulation and deployed "
            "in the real world, so the policy should be as steady and stable as possible with minimal action "
            "rate. Specifically, as it's running, the torso should remain near a z position of 0.34, and the "
            "orientation should be perpendicular to gravity. Also, the legs should move smoothly and avoid the "
            "DOF limits"
        ),
        "success_metric": (
            "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length"
        ),
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.02,
    },
    "Isaac-Velocity-Rough-Unitree-Go1-v0": {
        "description": (
           "Make the Unitree Go1 track a forward velocity of 2.0 m/s over rough terrain. "
           "Prioritize stable, low-jerk locomotion suitable for transfer. Keep torso height "
         "near ~0.34 m and orientation aligned with gravity; avoid joint limit banging."
     ),
    # Use a robust, generic success metric (episode length fraction) to avoid missing attributes.
    "success_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
    "success_metric_to_win": 1.0,
    "success_metric_tolerance": 0.02,
},

    "Isaac-Quadcopter-Direct-v0": {
        "description": (
            "bring the quadcopter to the target position: self._desired_pos_w, while making sure it flies smoothly"
        ),
        "success_metric": (
            "torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean()"
        ),
        "success_metric_to_win": 0.0,
        "success_metric_tolerance": 0.2,
    },
}

