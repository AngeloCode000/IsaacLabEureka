# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from collections import defaultdict
from typing import Any

import GPUtil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(path: str):
    """Load tensorboard logs from a given path.

    Args:
        path: The path to the tensorboard logs.

    Returns:
        A dictionary with the tags and their respective values.
    """
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)

    return data


def get_freest_gpu():
    """Get the GPU with the most free memory."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    # Sort GPUs by memory usage
    gpus.sort(key=lambda gpu: gpu.memoryUsed)
    return gpus[0].id


def shrink_physx_gpu_buffers(env_cfg: Any, *, max_envs: int = 128):
    """Downscale PhysX GPU buffer caps for lightweight interactive runs.

    Isaac Sim's GPU physics backend pre-allocates buffers based on several ``gpu_max_*`` hints on the
    ``PhysxCfg`` object. The defaults are chosen for large batch RL training (thousands of envs), which
    can lead to oversized allocations and CUDA OOM errors when only a handful of environments are run.

    This helper reduces the biggest contributors (rigid contact and patch buffers) whenever the scene
    count is modest. The values are rounded up to the nearest power-of-two to align with PhysX
    expectations and clamped so we never increase the user's original settings.
    """

    scene_cfg = getattr(env_cfg, "scene", None)
    physx_cfg = getattr(getattr(env_cfg, "sim", None), "physx", None)
    if scene_cfg is None or physx_cfg is None:
        return

    num_envs = getattr(scene_cfg, "num_envs", None)
    if num_envs is None or num_envs > max_envs:
        return

    def _next_power_of_two(value: int) -> int:
        value = max(1, value)
        return 1 << (value - 1).bit_length()

    contacts_per_env = 16384
    patches_per_env = 1024
    min_contacts_cap = 1 << 16  # ~65k pairs still plenty for a single robot on rough terrain.
    min_patches_cap = 1 << 13

    target_contact_cap = max(min_contacts_cap, _next_power_of_two(num_envs * contacts_per_env))
    if hasattr(physx_cfg, "gpu_max_rigid_contact_count"):
        original = physx_cfg.gpu_max_rigid_contact_count
        physx_cfg.gpu_max_rigid_contact_count = min(original, target_contact_cap)

    target_patch_cap = max(min_patches_cap, _next_power_of_two(num_envs * patches_per_env))
    if hasattr(physx_cfg, "gpu_max_rigid_patch_count"):
        original = physx_cfg.gpu_max_rigid_patch_count
        physx_cfg.gpu_max_rigid_patch_count = min(original, target_patch_cap)


class MuteOutput:
    """Context manager to mute stdout and stderr."""

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
