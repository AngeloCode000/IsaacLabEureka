def _get_rewards_eureka(self):
    """Manual reward shaping that prioritises commanded velocity tracking without exponential squashing."""
    observations = self._get_observations_synth()

    current_velocity = observations["current_velocity"]
    desired_velocity = observations["desired_velocity"]
    joint_velocities = observations["joint_velocities"]
    joint_positions = observations["joint_positions"]
    joint_limit_lower = observations["joint_limit_lower"]
    joint_limit_upper = observations["joint_limit_upper"]
    chassis_orientation = observations["chassis_orientation"]

    desired_speed = desired_velocity[..., 0]
    current_speed = current_velocity[..., 0]
    speed_error = torch.abs(current_speed - desired_speed)
    speed_tolerance = 0.5  # m/s tolerance before velocity reward decays quickly
    velocity_reward = 2.0 * (1.0 - torch.tanh(speed_error / speed_tolerance))

    upright_error = torch.linalg.norm(chassis_orientation[..., :2], dim=-1)
    upright_tolerance = 0.3
    balance_reward = torch.clamp(1.0 - upright_error / upright_tolerance, min=0.0)

    joint_motion_penalty = 0.05 * torch.mean(torch.abs(joint_velocities), dim=-1)

    excess_upper = torch.relu(joint_positions - joint_limit_upper)
    excess_lower = torch.relu(joint_limit_lower - joint_positions)
    joint_limit_penalty = 0.2 * (excess_upper + excess_lower).sum(dim=-1)

    total_reward = velocity_reward + balance_reward - joint_motion_penalty - joint_limit_penalty
    rewards_dict = {
        "velocity_reward": velocity_reward,
        "balance_reward": balance_reward,
        "joint_motion_penalty": joint_motion_penalty,
        "joint_limit_penalty": joint_limit_penalty,
    }
    return total_reward.to(self.device), {k: v.to(self.device) for k, v in rewards_dict.items()}
