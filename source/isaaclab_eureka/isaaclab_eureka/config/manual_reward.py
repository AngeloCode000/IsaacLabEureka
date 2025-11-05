def _get_rewards_eureka(self):
    """Manual reward shaping that prioritises forward velocity tracking in the robot body frame."""
    import torch

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

    device = self.device
    observations = self._get_observations_synth()

    robot = self.scene["robot"]
    root_lin_vel_w = robot.data.root_lin_vel_w
    root_quat_w = robot.data.root_quat_w
    yaw_only_quat = yaw_quat(root_quat_w)

    joint_velocities = observations["joint_velocities"]
    joint_positions = observations["joint_positions"]
    joint_limit_lower = observations["joint_limit_lower"]
    joint_limit_upper = observations["joint_limit_upper"]
    chassis_orientation = observations["chassis_orientation"]

    # Pull the commanded velocity, falling back to the command manager if needed.
    desired_velocity = observations.get("desired_velocity")
    if desired_velocity is None and getattr(self, "command_manager", None) is not None:
        desired_velocity = self.command_manager.get_command("base_velocity")
    if desired_velocity is None:
        desired_velocity = torch.zeros_like(root_lin_vel_w[:, :3])

    desired_velocity = desired_velocity.to(device)
    if desired_velocity.shape[-1] < 3:
        zeros = torch.zeros(desired_velocity.shape[:-1] + (3,), device=device, dtype=desired_velocity.dtype)
        zeros[..., : desired_velocity.shape[-1]] = desired_velocity
        desired_velocity = zeros
    else:
        desired_velocity = desired_velocity[..., :3]

    # Rotate velocities and commands into the gravity-aligned body frame (yaw frame).
    vel_body = quat_apply_inverse(yaw_only_quat, root_lin_vel_w[:, :3])

    current_forward = vel_body[..., 0]
    current_lateral = vel_body[..., 1]
    desired_forward = torch.linalg.norm(desired_velocity[..., :2], dim=-1)

    speed_tolerance = 0.5  # m/s tolerance before velocity reward decays quickly
    forward_tracking_error = torch.abs(current_forward - desired_forward)
    forward_reward = 2.0 * (1.0 - torch.tanh(forward_tracking_error / speed_tolerance))

    # Penalise sideways and backward motion in the body frame.
    lateral_penalty = 1.0 * torch.abs(current_lateral)
    backward_penalty = 1.5 * torch.relu(-current_forward)

    # Encourage aligning the robot's forward axis with the commanded heading when the command is meaningful.
    forward_axis_body = torch.zeros_like(root_lin_vel_w)
    forward_axis_body[..., 0] = 1.0
    forward_axis_world = quat_apply(yaw_only_quat, forward_axis_body)[..., :2]
    command_xy = desired_velocity[..., :2]
    command_speed = torch.linalg.norm(command_xy, dim=-1, keepdim=True)
    command_dir_world = torch.where(
        command_speed > 1e-3, command_xy / torch.clamp(command_speed, min=1e-3), torch.zeros_like(command_xy)
    )
    heading_alignment = torch.sum(forward_axis_world * command_dir_world, dim=-1)
    heading_threshold = 0.15
    heading_reward = torch.where(
        command_speed.squeeze(-1) > heading_threshold, torch.clamp(heading_alignment, min=0.0), torch.zeros_like(heading_alignment)
    )

    upright_error = torch.linalg.norm(chassis_orientation[..., :2], dim=-1)
    upright_tolerance = 0.3
    balance_reward = torch.clamp(1.0 - upright_error / upright_tolerance, min=0.0)

    joint_motion_penalty = 0.05 * torch.mean(torch.abs(joint_velocities), dim=-1)

    excess_upper = torch.relu(joint_positions - joint_limit_upper)
    excess_lower = torch.relu(joint_limit_lower - joint_positions)
    joint_limit_penalty = 0.2 * (excess_upper + excess_lower).sum(dim=-1)

    total_reward = (
        forward_reward
        + heading_reward
        + balance_reward
        - joint_motion_penalty
        - joint_limit_penalty
        - lateral_penalty
        - backward_penalty
    )
    rewards_dict = {
        "forward_reward": forward_reward,
        "heading_reward": heading_reward,
        "balance_reward": balance_reward,
        "lateral_penalty": lateral_penalty,
        "backward_penalty": backward_penalty,
        "joint_motion_penalty": joint_motion_penalty,
        "joint_limit_penalty": joint_limit_penalty,
    }
    return total_reward.to(device), {k: v.to(device) for k, v in rewards_dict.items()}
