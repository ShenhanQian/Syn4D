# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2025 Max Planck Society
#
# Convert SMPL-X (locked head) animation in .npz format to Alembic .abc geometry cache with 2023 UV map
#
# Requirements:
#   + Blender 4.0.2+
#   + SMPL-X Blender add-on 20240206+ (UV_2023)
#
#
# Notes:
# + Running via command-line: blender --background --python smplx_anim_to_alembic.py -- --input /path/to/npz --output /path/to/abc
#

import argparse
import bpy
from pathlib import Path
import sys
import time


import addon_utils

# Try enabling the SMPL-X addon (module name must match the addon's module)
# Common approach: search for likely modules and enable the one found.
candidates = ["smplx_tool", "smplx", "smplx_blender", "smplx_addon", "smplx_blender_addon"]
# candidates = ["smplx_blender_addon"]


##################################################
# Globals
##################################################
smplx_animation_path = Path("smplx_animation.npz")
output_path = Path("smplx_animation.abc")

def convert_to_abc(smplx_animation_path, output_path):
    if (not smplx_animation_path.exists()) or (smplx_animation_path.suffix != ".npz"):
        print(f"ERROR: Invalid input path: {smplx_animation_path}")
        return False

    if output_path.suffix != ".abc":
        print(f"ERROR: Invalid output path: {output_path}")
        return False

    # Import animation
    bpy.data.window_managers["WinMan"].smplx_tool.smplx_version = 'locked_head' # Use SMPL-X locked head (no head bun)
    bpy.data.window_managers["WinMan"].smplx_tool.smplx_uv = "UV_2023" # Use 2023 UV map

    anim_format="SMPL-X"
    bpy.ops.object.smplx_add_animation(filepath=str(smplx_animation_path), anim_format=anim_format, keyframe_corrective_pose_weights=True, target_framerate=30)

    # Export Alembic
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.smplx_export_alembic(filepath=str(output_path))
    return True

##############################################################################
# Main
##############################################################################
if __name__== "__main__":
    # Parse command-line arguments when invoked via `blender --background smplx_anim_to_alembic.py -- --input in.npz --output out.abc`
    if bpy.app.background:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1:]  # get all args after "--"

            parser = argparse.ArgumentParser(description="Convert SMPL-X animation in .npz format to Alembic .abc geometry cache")

            parser.add_argument("--input", required=True, type=str, help="Path to .npz input file")
            parser.add_argument("--output", required=True, type=str, help="Path to .abc output file")
            args = parser.parse_args(argv)
            smplx_animation_path = Path(args.input)
            output_path = Path(args.output)
    

    # addon_utils.enable("smplx_addon", default_set=True, persistent=True)
    
    # enabled = False
    # for mod in candidates:
    #     try:
    #         addon_utils.enable(mod, default_set=True, persistent=True)
    #         enabled = True
    #         print(f"Enabled addon module: {mod}")
    #         break
    #     except Exception:
    #         pass

    # if not enabled:
    #     raise RuntimeError(
    #         "Could not enable SMPL-X addon. Check the add-on module name in Preferences > Add-ons."
    #     )

    # Force addon registration if needed
    for mod in addon_utils.modules():
        if mod.__name__ in candidates:
            try:
                mod.register()
            except Exception:
                pass

    wm = bpy.context.window_manager
    if not hasattr(wm, "smplx_tool"):
        raise RuntimeError("SMPL-X addon enabled, but WindowManager.smplx_tool is still missing.")

    print(f"Converting: {smplx_animation_path} => {output_path}")
    start_time = time.perf_counter()
    convert_to_abc(smplx_animation_path, output_path)
    print(f"  Finished. Time: {(time.perf_counter() - start_time):.1f}s")