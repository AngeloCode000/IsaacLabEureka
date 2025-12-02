def _get_rewards_eureka(self):
    """Reward with strong commanded forward velocity tracking plus
    stability, pose, and smoothness shaping in the spirit of Sarrocco/Bertelli.
    """
    import torch
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

    device = self.device
    observations = self._get_observations_synth()

    robot = self.scene["robot"]
    root_lin_vel_w = robot.data.root_lin_vel_w
    root_ang_vel_w = robot.data.root_ang_vel_w  # NEW: for yaw-rate tracking
    root_pos_w = robot.data.root_pos_w          # NEW: for height tracking
    root_quat_w = robot.data.root_quat_w
    yaw_only_quat = yaw_quat(root_quat_w)

    joint_velocities = observations["joint_velocities"]
    joint_positions = observations["joint_positions"]
    joint_limit_lower = observations["joint_limit_lower"]
    joint_limit_upper = observations["joint_limit_upper"]
    chassis_orientation = observations["chassis_orientation"]
    step_dt = float(getattr(self, "step_dt", 0.02)) or 0.02

    # ---------------------------------------------------------------------
    # Command handling (linear velocity command)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 1) Linear velocity tracking (blog-style exponential + your bonus)
    # ---------------------------------------------------------------------
    speed_tolerance = torch.clamp(0.2 + 0.2 * command_speed_scalar, min=0.2)
    speed_error = current_forward - desired_forward
    forward_tracking_reward = torch.exp(-0.5 * torch.square(speed_error / speed_tolerance))

    forward_gain = torch.where(
        command_speed_scalar > 0.2,
        torch.full_like(command_speed_scalar, 6.0),
        torch.full_like(command_speed_scalar, 1.5),
    )

    commanded_dir = torch.sign(desired_forward)
    aligned_speed = commanded_dir * current_forward
    speed_bonus = torch.where(
        command_speed_scalar > 0.2,
        1.0 * torch.relu(aligned_speed),
        0.25 * torch.relu(aligned_speed),
    )
    forward_reward = forward_gain * forward_tracking_reward + speed_bonus

    # Additional drive term to favor actually moving when commanded.
    normalized_speed = torch.where(
        command_speed_scalar > 0.1,
        torch.relu(aligned_speed) / torch.clamp(command_speed_scalar, min=0.1),
        torch.zeros_like(aligned_speed),
    )
    progress_reward = torch.where(
        command_speed_scalar > 0.1,
        4.0 * torch.clamp(normalized_speed, max=2.0),
        torch.zeros_like(normalized_speed),
    )
    stalled_penalty = torch.where(
        command_speed_scalar > 0.4,
        2.5 * torch.relu(0.6 - normalized_speed),
        torch.zeros_like(normalized_speed),
    )

    # ---------------------------------------------------------------------
    # 2) Angular velocity tracking (blog: exp(-(w_ref - w)^2))
    # ---------------------------------------------------------------------
    current_yaw_rate = root_ang_vel_w[..., 2]

    desired_ang_vel = observations.get("desired_ang_vel", None)
    if desired_ang_vel is None:
        cmd_mgr = getattr(self, "command_manager", None)
        if cmd_mgr is not None:
            get_command = getattr(cmd_mgr, "get_command", None)
            if get_command is not None:
                try:
                    desired_ang_vel = get_command("base_ang_velocity")
                except KeyError:
                    try:
                        desired_ang_vel = get_command("base_velocity")
                    except KeyError:
                        desired_ang_vel = None

    if desired_ang_vel is None:
        desired_yaw_rate = torch.zeros_like(current_yaw_rate)
    else:
        desired_ang_vel = desired_ang_vel.to(device)
        if desired_ang_vel.shape[-1] >= 3:
            desired_yaw_rate = desired_ang_vel[..., 2]
        else:
            # Fallback: treat scalar/1D as yaw rate
            desired_yaw_rate = desired_ang_vel[..., 0]

    yaw_error = current_yaw_rate - desired_yaw_rate
    yaw_tolerance = 0.5
    ang_vel_tracking_reward = torch.exp(-0.5 * torch.square(yaw_error / yaw_tolerance))
    ang_vel_reward = 1.5 * ang_vel_tracking_reward  # weight; tune as needed

    # ---------------------------------------------------------------------
    # 3) Penalise sideways and backward motion in the body frame.
    # ---------------------------------------------------------------------
    lateral_weight = torch.where(
        command_speed_scalar > 0.2,
        torch.full_like(command_speed_scalar, 1.0),
        torch.full_like(command_speed_scalar, 0.6),
    )
    lateral_penalty = lateral_weight * torch.abs(current_lateral)

    backward_penalty = torch.where(
        command_speed_scalar > 0.2,
        0.8 * torch.relu(-aligned_speed),
        0.2 * torch.relu(-aligned_speed),
    )

    # ---------------------------------------------------------------------
    # 4) Heading alignment (directional tracking in world XY)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 5) Upright stability & roll/pitch-style shaping
    # ---------------------------------------------------------------------
    upright_error = torch.linalg.norm(chassis_orientation[..., :2], dim=-1)
    balance_reward = 1.0 * torch.exp(-0.5 * torch.square(upright_error / 0.25))

    # Gentle encouragement: base Z-axis parallel to world Z.
    z_axis_body = torch.zeros_like(root_lin_vel_w)
    z_axis_body[..., 2] = 1.0
    base_z_world = quat_apply(root_quat_w, z_axis_body)[..., :3]
    z_alignment = torch.abs(base_z_world[..., 2])  # 1.0 when parallel to +/- world Z
    z_parallel_reward = 0.25 * z_alignment

    # ---------------------------------------------------------------------
    # 6) Height tracking (blog: (z - z_ref)^2 penalty)
    # ---------------------------------------------------------------------
    base_height = root_pos_w[..., 2]

    desired_height = observations.get("desired_height", None)
    if desired_height is None:
        # Fallback: use a default height if defined, else the current mean.
        default_h = getattr(self, "default_base_height", None)
        if default_h is None:
            default_h = float(base_height.mean().detach())
        desired_height = torch.full_like(base_height, default_h, device=device)
    else:
        desired_height = desired_height.to(device).squeeze(-1)

    # Height tracking and strong low-height penalty
    height_error = base_height - desired_height

    # Keep this relatively mild so it doesn't dominate stepping behavior
    height_penalty = 1.0 * torch.square(height_error)

    # Strong punishment for being significantly below desired trot height.
    # margin: how far below desired_height we start calling it "collapsing"
    collapse_margin = 0.04  # ~4 cm below desired height; tune to the robot

    collapse_target = torch.clamp(desired_height - collapse_margin, min=0.20)
    collapse_depth = torch.relu(collapse_target - base_height)

    # Make this big enough to matter relative to ~100+ forward reward.
    collapse_penalty = 200.0 * collapse_depth

    # Mask out locomotion rewards when the base is clearly below trot height.
    upright_mask = torch.where(base_height > collapse_target, torch.ones_like(base_height), torch.zeros_like(base_height))
    forward_reward = forward_reward * upright_mask
    progress_reward = progress_reward * upright_mask

    # ---------------------------------------------------------------------
    # 7) Vertical velocity penalty (blog: v_z^2)
    # ---------------------------------------------------------------------
    vertical_vel = root_lin_vel_w[..., 2]
    vertical_vel_penalty = 0.25 * torch.square(vertical_vel)  # tune weight

    # Penalise rapid roll/pitch angular rates that make the robot shake.
    roll_pitch_rate_penalty = 0.2 * torch.linalg.norm(root_ang_vel_w[..., :2], dim=-1)

    # Encourage smooth base motion by penalising rapid changes in lin/ang velocity.
    if not hasattr(self, "_prev_root_lin_vel"):
        self._prev_root_lin_vel = torch.zeros_like(root_lin_vel_w)
        self._prev_root_ang_vel = torch.zeros_like(root_ang_vel_w)
    root_lin_acc = (root_lin_vel_w - self._prev_root_lin_vel) / max(step_dt, 1e-3)
    root_ang_acc = (root_ang_vel_w - self._prev_root_ang_vel) / max(step_dt, 1e-3)
    base_acc_penalty = 0.02 * torch.linalg.norm(root_lin_acc[..., :2], dim=-1)
    base_ang_acc_penalty = 0.01 * torch.abs(root_ang_acc[..., 2])
    self._prev_root_lin_vel = root_lin_vel_w.detach().clone()
    self._prev_root_ang_vel = root_ang_vel_w.detach().clone()

    # ---------------------------------------------------------------------
    # 8) Pose similarity penalty (blog: ||q - q_default||^2)
    # ---------------------------------------------------------------------
    default_joint_positions = observations.get("default_joint_positions", None)
    if default_joint_positions is None:
        default_joint_positions = getattr(self, "default_joint_positions", None)

    pose_penalty = torch.zeros(self.num_envs, device=device)
    if default_joint_positions is not None:
        default_joint_positions = default_joint_positions.to(device)
        pose_penalty = 0.05 * torch.mean(
            torch.square(joint_positions - default_joint_positions), dim=-1
        )  # tune weight

    # ---------------------------------------------------------------------
    # 9) Joint smoothness (velocity + acc penalties)
    # ---------------------------------------------------------------------
    smooth_scale = torch.where(
        command_speed_scalar < 0.3,
        torch.full_like(command_speed_scalar, 1.2),
        torch.ones_like(command_speed_scalar),
    )
    joint_motion_penalty = 0.01 * smooth_scale * torch.mean(torch.abs(joint_velocities), dim=-1)

    if not hasattr(self, "_prev_joint_velocities"):
        self._prev_joint_velocities = torch.zeros_like(joint_velocities)
    joint_acc = torch.abs(joint_velocities - self._prev_joint_velocities) / max(step_dt, 1e-3)
    joint_acc_clipped = torch.clamp(joint_acc, max=200.0)
    joint_acc_penalty = 0.003 * smooth_scale * torch.mean(joint_acc_clipped, dim=-1)
    self._prev_joint_velocities = joint_velocities.detach().clone()

    excess_upper = torch.relu(joint_positions - joint_limit_upper)
    excess_lower = torch.relu(joint_limit_lower - joint_positions)
    joint_limit_penalty = 0.3 * (excess_upper + excess_lower).sum(dim=-1)

    # ---------------------------------------------------------------------
    # 9b) Knee pose penalty to discourage deep crouch / knee-crawling
    # ---------------------------------------------------------------------
    # TODO: replace these indices with the actual knee joint indices for this robot
    K0, K1, K2, K3 = 0, 1, 2, 3
    knee_joint_indices = torch.tensor([K0, K1, K2, K3], device=device, dtype=torch.long)

    # Select knee angles: shape [num_envs, num_knees]
    knee_angles = joint_positions[:, knee_joint_indices]

    # Choose a "too bent" threshold in radians.
    # Example: if standing knees are ~0.6 rad and crawling is > 1.2 rad,
    # set threshold somewhere around 0.9–1.0.
    kneel_thresh = 1.0

    # Penalize only the excess beyond this threshold
    knee_flex_excess = torch.relu(torch.abs(knee_angles) - kneel_thresh)

    # Weight so that staying in a deep crouch is clearly worse than standing.
    knee_pose_penalty = 6.0 * knee_flex_excess.mean(dim=-1)

    # ---------------------------------------------------------------------
    # 10) Explicit action-rate penalty (blog: ||a_t - a_{t-1}||^2)
    # ---------------------------------------------------------------------
    action_rate_penalty = torch.zeros(self.num_envs, device=device)
    action_jerk_penalty = torch.zeros(self.num_envs, device=device)
    actions = getattr(self, "actions", None)
    if actions is not None:
        actions = actions.to(device)
        if not hasattr(self, "_prev_actions"):
            self._prev_actions = torch.zeros_like(actions)
        if not hasattr(self, "_prev_action_delta"):
            self._prev_action_delta = torch.zeros_like(actions)
        action_delta = actions - self._prev_actions
        action_delta_clipped = torch.clamp(action_delta, min=-2.0, max=2.0)
        action_rate_penalty = 0.01 * torch.mean(torch.square(action_delta_clipped), dim=-1)
        action_jerk = (action_delta_clipped - self._prev_action_delta) / max(step_dt, 1e-3)
        action_jerk_penalty = 0.002 * torch.mean(torch.abs(action_jerk), dim=-1)
        self._prev_actions = actions.detach().clone()
        self._prev_action_delta = action_delta_clipped.detach().clone()

    # ---------------------------------------------------------------------
    # 11) Non-foot / knee contact logic (temporarily disabled: no sensors wired)
    # ---------------------------------------------------------------------
    contact_penalty = torch.zeros(self.num_envs, device=device)
    knee_contact_penalty = torch.zeros(self.num_envs, device=device)
    stance_balance_reward = torch.zeros(self.num_envs, device=device)
    # ---------------------------------------------------------------------
    # Total reward aggregation
    # ---------------------------------------------------------------------
    total_reward = (
        forward_reward
        + ang_vel_reward
        + heading_reward
        + balance_reward
        + z_parallel_reward
        # stance_balance_reward removed (currently always zero / disabled)
        # contact_penalty and knee_contact_penalty removed (no sensors wired)
        - roll_pitch_rate_penalty
        - base_acc_penalty
        - base_ang_acc_penalty
        - joint_motion_penalty
        - joint_acc_penalty
        - joint_limit_penalty
        - lateral_penalty
        - backward_penalty
        - height_penalty
        - collapse_penalty
        - vertical_vel_penalty
        - pose_penalty
        - action_rate_penalty
        - action_jerk_penalty
        - stalled_penalty
        - knee_pose_penalty
        + progress_reward
    )

    rewards_dict = {
        "forward_reward": forward_reward,
        "ang_vel_reward": ang_vel_reward,
        "progress_reward": progress_reward,
        "stalled_penalty": stalled_penalty,
        "heading_reward": heading_reward,
        "balance_reward": balance_reward,
        "z_parallel_reward": z_parallel_reward,
        "lateral_penalty": lateral_penalty,
        "backward_penalty": backward_penalty,
        "roll_pitch_rate_penalty": roll_pitch_rate_penalty,
        "base_acc_penalty": base_acc_penalty,
        "base_ang_acc_penalty": base_ang_acc_penalty,
        "joint_motion_penalty": joint_motion_penalty,
        "joint_acc_penalty": joint_acc_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "height_penalty": height_penalty,
        "collapse_penalty": collapse_penalty,
        "vertical_vel_penalty": vertical_vel_penalty,
        "pose_penalty": pose_penalty,
        "action_rate_penalty": action_rate_penalty,
        "action_jerk_penalty": action_jerk_penalty,
        "knee_pose_penalty": knee_pose_penalty,
    }
    return total_reward.to(device), {k: v.to(device) for k, v in rewards_dict.items()}
