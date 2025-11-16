#!/usr/bin/env python3
"""
Build a thin USD wrapper that rotates the OpenMutt asset so its +X axis lines up with IsaacLab's convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd, UsdGeom, Gf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a corrective yaw transform above the OpenMutt articulation."
    )
    parser.add_argument(
        "--usd-in",
        default="/home/eppl/Downloads/openMutt_IsaacLab/configuration/OpenMuttMasterStage.usd",
        help="Existing OpenMutt USD to reference.",
    )
    parser.add_argument(
        "--usd-out",
        default="/home/eppl/Downloads/openMutt_IsaacLab/configuration/OpenMuttMasterStage_xfwd.usd",
        help="Destination USD that contains the wrapper transform.",
    )
    parser.add_argument(
        "--prim-path",
        default="/World/MASTER",
        help="Prim path of the articulated robot within the source USD.",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=-90.0,
        help="Yaw rotation to apply (degrees). Use +90 if the model faces -Y.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.usd_in).expanduser().resolve()
    dst = Path(args.usd_out).expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source USD not found: {src}")

    stage = Usd.Stage.CreateNew(str(dst))

    # Pull the source USD in as a sublayer so we inherit every prim definition verbatim.
    # To undo this wrapper, delete the generated *_xfwd.usd file and point configs back at `--usd-in`.
    stage.GetRootLayer().subLayerPaths.append(str(src))

    # Keep the same default prim (/World) so downstream configs remain valid.
    world_prim = stage.OverridePrim("/World")
    stage.SetDefaultPrim(world_prim)

    wrapper = stage.OverridePrim(args.prim_path)
    xform = UsdGeom.Xformable(wrapper)
    existing_ops = list(xform.GetOrderedXformOps())

    yaw_op = xform.AddRotateZOp(opSuffix="yaw_fix")
    yaw_op.Set(args.yaw_deg)

    # Override any baked-in +90 deg orient about X so the asset stands upright without extra config.
    # Remove this block (or set the quaternion back to the original value) to restore the authored roll.
    for op in existing_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    # Ensure the yaw happens before the baked-in +90° roll (xformOp:orient). We place all existing
    # translate ops first, then the yaw, followed by the rest so the up axis stays Z. Remove this block
    # (or set a neutral op order) to revert to the original transform stack.
    translate_ops = [op for op in existing_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    remaining_ops = [op for op in existing_ops if op.GetOpType() != UsdGeom.XformOp.TypeTranslate]
    new_order = translate_ops + [yaw_op] + remaining_ops
    xform.SetXformOpOrder(new_order)

    stage.GetRootLayer().Save()
    print(f"Created wrapper stage: {dst}")


if __name__ == "__main__":
    main()
