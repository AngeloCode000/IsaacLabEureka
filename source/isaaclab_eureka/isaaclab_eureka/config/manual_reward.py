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

    # Track the commanded forward velocity with a sharper peak and encourage any forward drive when commanded.
    speed_tolerance = torch.clamp(0.35 + 0.35 * command_speed_scalar, min=0.35)
    speed_error = current_forward - desired_forward
    forward_tracking_reward = torch.exp(-0.5 * torch.square(speed_error / speed_tolerance))
    forward_gain = torch.where(command_speed_scalar > 0.2, torch.full_like(command_speed_scalar, 6.0), torch.full_like(command_speed_scalar, 3.0))
    forward_motion_bonus = torch.where(
        command_speed_scalar > 0.2,
        1.0 * torch.relu(current_forward),
        0.5 * torch.relu(current_forward),
    )
    forward_reward = forward_gain * forward_tracking_reward + forward_motion_bonus

    # Penalise sideways and backward motion in the body frame.
    lateral_weight = torch.where(
        command_speed_scalar > 0.2,
        torch.full_like(command_speed_scalar, 1.5),
        torch.full_like(command_speed_scalar, 0.75),
    )
    lateral_penalty = lateral_weight * torch.abs(current_lateral)
    backward_penalty = 1.5 * torch.relu(-current_forward)

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
        torch.clamp(heading_alignment, min=0.0),
        torch.zeros_like(heading_alignment),
    )

    upright_error = torch.linalg.norm(chassis_orientation[..., :2], dim=-1)
    upright_tolerance = 0.3
    balance_reward = torch.clamp(1.0 - upright_error / upright_tolerance, min=0.0)

    smooth_scale = torch.where(
        command_speed_scalar < 0.3,
        torch.full_like(command_speed_scalar, 1.2),
        torch.ones_like(command_speed_scalar),
    )
    joint_motion_penalty = 0.01 * smooth_scale * torch.mean(torch.abs(joint_velocities), dim=-1)

    if not hasattr(self, "_prev_joint_velocities"):
        self._prev_joint_velocities = torch.zeros_like(joint_velocities)
    joint_acc = torch.abs(joint_velocities - self._prev_joint_velocities) / max(step_dt, 1e-3)
    joint_acc_penalty = 0.005 * smooth_scale * torch.mean(joint_acc, dim=-1)
    self._prev_joint_velocities = joint_velocities.detach().clone()

    excess_upper = torch.relu(joint_positions - joint_limit_upper)
    excess_lower = torch.relu(joint_limit_lower - joint_positions)
    joint_limit_penalty = 0.2 * (excess_upper + excess_lower).sum(dim=-1)

    # Contact-based shaping: penalize non-foot contacts and reward balanced stance counts.
    contact_penalty = torch.zeros(self.num_envs, device=device)
    stance_balance_reward = torch.zeros(self.num_envs, device=device)
    sensors = getattr(self.scene, "sensors", None)
    contact_sensor = None
    if sensors is not None:
        if isinstance(sensors, dict):
            contact_sensor = sensors.get("contact_forces")
        else:
            getter = getattr(sensors, "get", None)
            if callable(getter):
                contact_sensor = getter("contact_forces")
    if contact_sensor is not None:
        net_forces = getattr(contact_sensor.data, "net_forces_w", None)
        body_names = getattr(contact_sensor, "body_names", [])
        if net_forces is not None and len(body_names) == net_forces.shape[1]:
            if not hasattr(self, "_foot_contact_mask_cpu"):
                foot_keywords = ("Leg_Knee_Cartilage_Outer",)
                foot_mask_list = [
                    any(keyword in body_name for keyword in foot_keywords) for body_name in body_names
                ]
                self._foot_contact_mask_cpu = torch.tensor(foot_mask_list, dtype=torch.bool)
            foot_mask = self._foot_contact_mask_cpu.to(net_forces.device)
            if torch.any(foot_mask):
                non_foot_mask = ~foot_mask
                contact_magnitudes = torch.linalg.norm(net_forces, dim=-1)
                contact_active = contact_magnitudes > 1.0
                non_foot_contact_count = (contact_active & non_foot_mask.unsqueeze(0)).sum(dim=1).float()
                contact_penalty = 0.1 * non_foot_contact_count.to(device)

                foot_contacts = (contact_active & foot_mask.unsqueeze(0)).float()
                foot_contact_count = foot_contacts.sum(dim=1)
                stance_target = 2.0
                stance_balance_reward = 0.1 * torch.exp(
                    -0.5 * torch.square((foot_contact_count - stance_target) / max(stance_target, 1.0))
                ).to(device)

    total_reward = (
        forward_reward
        + heading_reward
        + balance_reward
        + stance_balance_reward
        - joint_motion_penalty
        - joint_acc_penalty
        - joint_limit_penalty
        - lateral_penalty
        - backward_penalty
        - contact_penalty
    )
    rewards_dict = {
        "forward_reward": forward_reward,
        "heading_reward": heading_reward,
        "balance_reward": balance_reward,
        "stance_balance_reward": stance_balance_reward,
        "lateral_penalty": lateral_penalty,
        "backward_penalty": backward_penalty,
        "joint_motion_penalty": joint_motion_penalty,
        "joint_acc_penalty": joint_acc_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "non_foot_contact_penalty": contact_penalty,
    }
    return total_reward.to(device), {k: v.to(device) for k, v in rewards_dict.items()}
