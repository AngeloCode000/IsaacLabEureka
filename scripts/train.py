# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train an RL agent with Isaac Lab Eureka."""

import argparse
import os

from isaaclab_eureka.eureka import Eureka


def main(args_cli):
    prompt_addendum = None
    if args_cli.prompt_addendum_file is not None:
        with open(args_cli.prompt_addendum_file, "r", encoding="utf-8") as f:
            prompt_addendum = f.read()

    manual_reward = None
    if args_cli.manual_reward_file is not None:
        with open(args_cli.manual_reward_file, "r", encoding="utf-8") as f:
            manual_reward = f.read()

    eureka = Eureka(
        task=args_cli.task,
        rl_library=args_cli.rl_library,
        num_parallel_runs=args_cli.num_parallel_runs,
        device=args_cli.device,
        env_seed=args_cli.env_seed,
        max_training_iterations=args_cli.max_training_iterations,
        feedback_subsampling=args_cli.feedback_subsampling,
        temperature=args_cli.temperature,
        gpt_model=args_cli.gpt_model,
        prompt_addendum=prompt_addendum,
        manual_reward=manual_reward,
    )

    eureka.run(max_eureka_iterations=args_cli.max_eureka_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent with Eureka.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0", help="Name of the task.")
    parser.add_argument(
        "--num_parallel_runs", type=int, default=1, help="Number of Eureka runs to execute in parallel."
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument("--env_seed", type=int, default=42, help="The random seed to use for the environment.")
    parser.add_argument("--max_eureka_iterations", type=int, default=5, help="The number of Eureka iterations to run.")
    parser.add_argument(
        "--max_training_iterations",
        type=int,
        default=100,
        help="The number of RL training iterations to run for each Eureka iteration.",
    )
    parser.add_argument(
        "--feedback_subsampling",
        type=int,
        default=10,
        help="The subsampling of the metrics given as feedack to the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controls the randomness of the GPT output (0 is deterministic, 1 is highly diverse).",
    )
    parser.add_argument("--gpt_model", type=str, default="gpt-4", help="The GPT model to use.")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games"],
        help="The RL training library to use.",
    )
    parser.add_argument(
        "--prompt_addendum_file",
        type=str,
        default=None,
        help="Optional path to a text file whose contents are appended to every Eureka user prompt.",
    )
    parser.add_argument(
        "--manual_reward_file",
        type=str,
        default=None,
        help=(
            "Optional path to a file containing a `_get_rewards_eureka` function. When provided, the first Eureka "
            "iteration runs strictly with that reward before falling back to GPT suggestions."
        ),
    )
    args_cli = parser.parse_args()

    # Check parameter validity
    if os.name == "nt" and args_cli.num_parallel_runs > 1:
        print(
            "[WARNING]: Running with num_parallel_runs > 1 is not supported on Windows. Setting num_parallel_runs = 1."
        )
        args_cli.num_parallel_runs = 1

    # Run the main function
    main(args_cli)
