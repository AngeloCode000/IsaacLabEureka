def _get_rewards_eureka(self):
    """Simplified reward: strongly incentivize commanded forward velocity.

    Rationale:
    - Policy tends to stay in place; we boost forward-velocity tracking and add a
      direct linear velocity bonus in commanded direction.
    - Simplify shaping: keep small upright term; soften lateral/backward and
      joint smoothness penalties; drop stance/contact shaping to avoid distracting
      gradients.
    """
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
    step_dt = float(getattr(self, "step_dt", 0.02)) or 0.02

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
    command_body = quat_apply_inverse(yaw_only_quat, desired_velocity)

    current_forward = vel_body[..., 0]
    current_lateral = vel_body[..., 1]
    desired_forward = command_body[..., 0]
    command_speed_scalar = torch.abs(desired_forward)

    # Track the commanded forward velocity with a sharper peak and add a
    # direct linear bonus along commanded direction.
    speed_tolerance = torch.clamp(0.3 + 0.3 * command_speed_scalar, min=0.3)
    speed_error = current_forward - desired_forward
    forward_tracking_reward = torch.exp(-0.5 * torch.square(speed_error / speed_tolerance))
    # Strong weight on tracking; strictly zero if command is near-zero to avoid drift bias
    forward_gain = torch.where(
        command_speed_scalar > 0.2,
        torch.full_like(command_speed_scalar, 8.0),
        torch.zeros_like(command_speed_scalar),
    )
    # Linear bonus in the commanded direction (sign-aware)
    commanded_dir = torch.sign(desired_forward)
    aligned_speed = commanded_dir * current_forward
    speed_bonus = torch.where(
        command_speed_scalar > 0.2,
        1.5 * torch.relu(aligned_speed),
        torch.zeros_like(aligned_speed),
    )
    forward_reward = forward_gain * forward_tracking_reward + speed_bonus

    # Penalise sideways and backward motion in the body frame.
    lateral_weight = torch.where(
        command_speed_scalar > 0.2,
        torch.full_like(command_speed_scalar, 0.8),
        torch.full_like(command_speed_scalar, 0.4),
    )
    lateral_penalty = lateral_weight * torch.abs(current_lateral)
    # Small penalty for motion opposite to command (zero when no command)
    backward_penalty = torch.where(
        command_speed_scalar > 0.2,
        0.6 * torch.relu(-aligned_speed),
        torch.zeros_like(aligned_speed),
    )

    # Encourage aligning the robot's forward axis with the commanded heading when the command is meaningful.
    forward_axis_body = torch.zeros_like(root_lin_vel_w)
    forward_axis_body[..., 0] = 1.0
    forward_axis_world = quat_apply(yaw_only_quat, forward_axis_body)[..., :2]
    command_xy = desired_velocity[..., :2]
    command_speed_heading = torch.linalg.norm(command_xy, dim=-1, keepdim=True)
    command_dir_world = torch.where(
        command_speed_heading > 1e-3,
        command_xy / torch.clamp(command_speed_heading, min=1e-3),
        torch.zeros_like(command_xy),
    )
    heading_alignment = torch.sum(forward_axis_world * command_dir_world, dim=-1)
    heading_threshold = 0.15
    heading_reward = torch.where(
        command_speed_heading.squeeze(-1) > heading_threshold,
        0.5 * torch.clamp(heading_alignment, min=0.0),
        torch.zeros_like(heading_alignment),
    )

    upright_error = torch.linalg.norm(chassis_orientation[..., :2], dim=-1)
    upright_tolerance = 0.3
    balance_reward = 0.5 * torch.clamp(1.0 - upright_error / upright_tolerance, min=0.0)

    # Gentle encouragement: keep base Z-axis parallel with gravity (body parallel to ground).
    # Compute base Z-axis in world frame and reward its alignment with world Z (sign-agnostic).
    # This is intentionally small so it cannot outweigh velocity/stability terms.
    z_axis_body = torch.zeros_like(root_lin_vel_w)
    z_axis_body[..., 2] = 1.0
    base_z_world = quat_apply(root_quat_w, z_axis_body)[..., :3]
    z_alignment = torch.abs(base_z_world[..., 2])  # 1.0 when parallel to +/- world Z
    z_parallel_reward = 0.15 * z_alignment

    # Softer joint smoothness penalties so they don't dominate
    smooth_scale = torch.where(
        command_speed_scalar < 0.3,
        torch.full_like(command_speed_scalar, 1.0),
        torch.ones_like(command_speed_scalar),
    )
    joint_motion_penalty = 0.005 * smooth_scale * torch.mean(torch.abs(joint_velocities), dim=-1)

    if not hasattr(self, "_prev_joint_velocities"):
        self._prev_joint_velocities = torch.zeros_like(joint_velocities)
    joint_acc = torch.abs(joint_velocities - self._prev_joint_velocities) / max(step_dt, 1e-3)
    joint_acc_penalty = 0.001 * smooth_scale * torch.mean(joint_acc, dim=-1)
    self._prev_joint_velocities = joint_velocities.detach().clone()

    excess_upper = torch.relu(joint_positions - joint_limit_upper)
    excess_lower = torch.relu(joint_limit_lower - joint_positions)
    joint_limit_penalty = 0.2 * (excess_upper + excess_lower).sum(dim=-1)

    # Drop contact/stance shaping to simplify signal
    contact_penalty = torch.zeros(self.num_envs, device=device)
    stance_balance_reward = torch.zeros(self.num_envs, device=device)

    total_reward = (
        forward_reward
        + heading_reward
        + balance_reward
        + z_parallel_reward
        - joint_motion_penalty
        - joint_acc_penalty
        - joint_limit_penalty
        - lateral_penalty
        - backward_penalty
    )
    rewards_dict = {
        "forward_reward": forward_reward,
        "heading_reward": heading_reward,
        "balance_reward": balance_reward,
        "z_parallel_reward": z_parallel_reward,
        "stance_balance_reward": stance_balance_reward,
        "lateral_penalty": lateral_penalty,
        "backward_penalty": backward_penalty,
        "joint_motion_penalty": joint_motion_penalty,
        "joint_acc_penalty": joint_acc_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "non_foot_contact_penalty": contact_penalty,
    }
    return total_reward.to(device), {k: v.to(device) for k, v in rewards_dict.items()}
