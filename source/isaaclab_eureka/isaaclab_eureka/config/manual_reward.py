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
    balance_reward = 0.8 * torch.exp(-0.5 * torch.square(upright_error / 0.25))

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

    height_error = base_height - desired_height
    height_penalty = 2.0 * torch.square(height_error)  # tune weight
    collapse_target = 0.85 * torch.clamp(desired_height, min=0.25)
    collapse_penalty = 4.0 * torch.relu(collapse_target - base_height)

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
    # 11) Non-foot contact penalty (anti knee-crawling)
    # ---------------------------------------------------------------------
    if not hasattr(self, "_debug_contact_once"):
        obs_contact_keys = [k for k in observations.keys() if "contact" in k or "force" in k]
        data_contact_keys = [k for k in dir(robot.data) if "contact" in k or "force" in k]
        print("obs contact-ish keys:", obs_contact_keys)
        print("robot.data contact-ish keys:", data_contact_keys)
        self._debug_contact_once = True

    contact_penalty = torch.zeros(self.num_envs, device=device)
    knee_contact_penalty = torch.zeros(self.num_envs, device=device)
    stance_balance_reward = torch.zeros(self.num_envs, device=device)
    contact_forces = None
    force_mag = None
    valid_feet = []
    contact_sensor = getattr(self.scene, "contact_forces", None)
    if contact_sensor is None and hasattr(self.scene, "sensors"):
        contact_sensor = self.scene.sensors.get("contact_forces", None)
    if contact_sensor is not None:
        try:
            contact_forces = contact_sensor.data.net_forces_w
            if not hasattr(self, "_foot_body_ids"):
                body_names = contact_sensor.body_names
                if not hasattr(self, "_logged_contact_names"):
                    print("contact sensor body names:", body_names)
                    self._logged_contact_names = True
                foot_ids, _ = contact_sensor.find_bodies(
                    [
                        ".*_foot.*",
                        ".*foot.*",
                        ".*toe.*",
                        ".*FOOT.*",
                        ".*Toe.*",
                    ]
                )
                self._foot_body_ids = [int(i) for i in foot_ids]
                knee_ids, _ = contact_sensor.find_bodies(
                    [
                        ".*knee.*",
                        ".*thigh.*",
                        ".*shin.*",
                        ".*shank.*",
                        ".*calf.*",
                    ]
                )
                self._knee_body_ids = [int(i) for i in knee_ids if int(i) not in self._foot_body_ids]
                if not hasattr(self, "_logged_resolved_feet"):
                    print("resolved foot ids:", self._foot_body_ids)
                    self._logged_resolved_feet = True
                if not hasattr(self, "_logged_resolved_knees"):
                    print("resolved knee ids:", getattr(self, "_knee_body_ids", []))
                    self._logged_resolved_knees = True
        except Exception:
            contact_forces = None

    if contact_forces is None:
        contact_force_keys = (
            "contact_forces_w",
            "net_contact_forces_w",
            "contact_forces",
            "net_contact_forces",
            "contact_force_w",
            "contact_force",
        )
        for key in contact_force_keys:
            if key in observations and observations[key] is not None:
                contact_forces = observations[key]
                break
            value = getattr(robot.data, key, None)
            if value is not None:
                contact_forces = value
                break

    if contact_forces is not None and contact_forces.dim() >= 3 and contact_forces.numel() > 0:
        force_mag = torch.linalg.norm(contact_forces[..., :3], dim=-1)
        num_bodies = force_mag.shape[1] if force_mag.dim() > 1 else 0
        foot_indices = None
        if hasattr(self, "_foot_body_ids"):
            foot_indices = self._foot_body_ids
        else:
            for attr in ("foot_indices", "feet_indices", "feet_ids", "foot_ids", "foot_links"):
                candidate = getattr(self, attr, None)
                if candidate is not None:
                    try:
                        foot_indices = [int(i) for i in candidate]
                    except TypeError:
                        pass
                    break
        if num_bodies > 0:
            if foot_indices is not None:
                valid_feet = [i for i in foot_indices if 0 <= i < num_bodies]
            if valid_feet:
                body_mask = torch.ones(num_bodies, dtype=torch.bool, device=device)
                body_mask[torch.tensor(valid_feet, device=device)] = False
                non_foot_forces = force_mag[:, body_mask]
            else:
                non_foot_forces = force_mag
            contact_penalty = 0.03 * non_foot_forces.sum(dim=-1)

            if valid_feet:
                foot_ids_tensor = torch.as_tensor(valid_feet, device=device, dtype=torch.long)
                foot_forces = force_mag[:, foot_ids_tensor]
                stance_contacts = (foot_forces > 5.0).float()
                stance_count = stance_contacts.sum(dim=-1)
                stance_balance_reward = 0.1 * torch.exp(
                    -0.5 * torch.square((stance_count - 2.5) / 1.0)
                )
            knee_indices = None
            if hasattr(self, "_knee_body_ids"):
                knee_indices = [i for i in self._knee_body_ids if 0 <= i < num_bodies]
            if knee_indices:
                knee_ids_tensor = torch.as_tensor(knee_indices, device=device, dtype=torch.long)
                knee_forces = force_mag[:, knee_ids_tensor]
                knee_force_excess = torch.relu(knee_forces - 5.0)
                knee_contact_penalty = 0.06 * knee_force_excess.sum(dim=-1)

    # ---------------------------------------------------------------------
    # Total reward aggregation
    # ---------------------------------------------------------------------
    total_reward = (
        forward_reward
        + ang_vel_reward
        + heading_reward
        + balance_reward
        + z_parallel_reward
        + stance_balance_reward
        - contact_penalty
        - knee_contact_penalty
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
        "stance_balance_reward": stance_balance_reward,
        "lateral_penalty": lateral_penalty,
        "backward_penalty": backward_penalty,
        "roll_pitch_rate_penalty": roll_pitch_rate_penalty,
        "base_acc_penalty": base_acc_penalty,
        "base_ang_acc_penalty": base_ang_acc_penalty,
        "joint_motion_penalty": joint_motion_penalty,
        "joint_acc_penalty": joint_acc_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "non_foot_contact_penalty": contact_penalty,
        "knee_contact_penalty": knee_contact_penalty,
        "height_penalty": height_penalty,
        "collapse_penalty": collapse_penalty,
        "vertical_vel_penalty": vertical_vel_penalty,
        "pose_penalty": pose_penalty,
        "action_rate_penalty": action_rate_penalty,
        "action_jerk_penalty": action_jerk_penalty,
    }
    return total_reward.to(device), {k: v.to(device) for k, v in rewards_dict.items()}
