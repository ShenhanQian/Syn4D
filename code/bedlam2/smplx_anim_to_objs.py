# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2025 Max Planck Society
#
# Convert SMPL-X animation:
#   + .npz -> Alembic .abc geometry cache (via SMPL-X addon)
#   + .abc -> OBJ sequence (one OBJ per frame)
#
# Requirements:
#   + Blender 4.0.2+
#   + SMPL-X Blender add-on 20240206+ (UV_2023) for .npz -> .abc
#
#
# Notes:
# + Running via command-line:
#     blender --background --python smplx_anim_to_objs.py -- --input /path/to/anim.npz --output /path/to/anim.abc
#     blender --background --python smplx_anim_to_objs.py -- --input /path/to/anim.abc --output /path/to/obj_dir --obj_prefix body
#

import argparse
import bpy
from pathlib import Path
import sys
import time
from math import ceil
import numpy as np


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


def _enable_smplx_addon_if_needed():
    # Force addon registration if needed.
    for mod in addon_utils.modules():
        if mod.__name__ in candidates:
            try:
                mod.register()
            except Exception:
                pass

    wm = bpy.context.window_manager
    if not hasattr(wm, "smplx_tool"):
        raise RuntimeError("SMPL-X addon enabled, but WindowManager.smplx_tool is still missing.")


def _export_obj(filepath):
    # Blender 4.x operator
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=str(filepath), export_selected_objects=True)
        return

    # Fallback for old Blender versions/addons
    if hasattr(bpy.ops.export_scene, "obj"):
        bpy.ops.export_scene.obj(filepath=str(filepath), use_selection=True)
        return

    raise RuntimeError("No OBJ export operator found (expected bpy.ops.wm.obj_export or bpy.ops.export_scene.obj).")


def _extract_vertices_and_faces(mesh_object, depsgraph):
    obj_eval = mesh_object.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    try:
        mesh_eval.calc_loop_triangles()

        verts = np.empty((len(mesh_eval.vertices), 3), dtype=np.float32)
        mesh_eval.vertices.foreach_get("co", verts.reshape(-1))

        faces = np.empty((len(mesh_eval.loop_triangles), 3), dtype=np.int32)
        mesh_eval.loop_triangles.foreach_get("vertices", faces.reshape(-1))
        return verts, faces
    finally:
        obj_eval.to_mesh_clear()


def convert_abc_to_obj_sequence(
    abc_path,
    output_dir,
    frame_start=None,
    frame_end=None,
    obj_prefix="frame",
    target_fps=30.0,
    source_fps=None,
    garment_mesh_name=None,
    npz_output_path=None,
):
    if (not abc_path.exists()) or (abc_path.suffix.lower() != ".abc"):
        print(f"ERROR: Invalid Alembic input path: {abc_path}")
        return False

    # output_dir.mkdir(parents=True, exist_ok=True)
    if npz_output_path is not None:
        npz_output_path.parent.mkdir(parents=True, exist_ok=True)
    

    # Track objects before import to identify newly imported Alembic objects.
    existing_names = {obj.name for obj in bpy.data.objects}
    scene = bpy.context.scene
    scene.render.fps = int(target_fps)
    scene.render.fps_base = 1.0
    bpy.ops.wm.alembic_import(filepath=str(abc_path), as_background_job=False)
    imported_objects = [obj for obj in bpy.data.objects if obj.name not in existing_names]
    mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    if not mesh_objects:
        print("ERROR: Alembic import finished but no mesh objects were found.")
        return False

    garment_mesh = mesh_objects[0]
    if garment_mesh_name is not None:
        matches = [obj for obj in mesh_objects if obj.name == garment_mesh_name]
        if not matches:
            print(f"ERROR: Could not find garment mesh '{garment_mesh_name}' among imported meshes.")
            return False
        garment_mesh = matches[0]

    if npz_output_path is None:
        npz_output_path = output_dir / f"{obj_prefix}_garment_seq.npz"
    else:
        npz_output_path = Path(npz_output_path)
    npz_output_path.parent.mkdir(parents=True, exist_ok=True)
    start = scene.frame_start if frame_start is None else int(frame_start)
    end = scene.frame_end if frame_end is None else int(frame_end)
    print(f"start: {start}, end: {end}")
    if end < start:
        print(f"ERROR: Invalid frame range [{start}, {end}]")
        return False

    detected_source_fps = scene.render.fps / scene.render.fps_base
    for obj in mesh_objects:
        for modifier in obj.modifiers:
            if modifier.type == "MESH_SEQUENCE_CACHE" and getattr(modifier, "cache_file", None):
                cache_file = modifier.cache_file
                if hasattr(cache_file, "fps") and float(cache_file.fps) > 0.0:
                    detected_source_fps = float(cache_file.fps)
                break

    source_fps = detected_source_fps if source_fps is None else float(source_fps)
    target_fps = float(target_fps)
    if source_fps <= 0.0 or target_fps <= 0.0:
        print(f"ERROR: FPS must be positive (source_fps={source_fps}, target_fps={target_fps}).")
        return False

    duration_seconds = (end - start) / source_fps
    num_output_frames = int(ceil(duration_seconds * target_fps)) + 1

    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices_seq = []
    faces = None

    for export_idx in range(start, end + 1):
        # src_frame_float = start + (export_idx / target_fps) * source_fps
        # if src_frame_float > end + 1e-6:
        #     break

        frame_int = export_idx
        # subframe = src_frame_float - frame_int
        # scene.frame_set(frame_int, subframe=subframe)
        scene.frame_set(frame_int)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            if obj.name in bpy.data.objects:
                obj.select_set(True)

        bpy.context.view_layer.objects.active = mesh_objects[0]
        obj_path = output_dir / f"{obj_prefix}_{export_idx:06d}.obj"
        # _export_obj(obj_path)

        frame_vertices, frame_faces = _extract_vertices_and_faces(garment_mesh, depsgraph)
        if faces is None:
            faces = frame_faces
        elif faces.shape != frame_faces.shape or not np.array_equal(faces, frame_faces):
            print("ERROR: Garment mesh topology changes across frames; cannot save constant 'faces' array.")
            return False
        vertices_seq.append(frame_vertices)

    if not vertices_seq:
        print("ERROR: No frames were exported.")
        return False

    np.savez(
        str(npz_output_path),
        vertices_seq=np.stack(vertices_seq, axis=0),
        faces=faces,
        start_frame=start,
        end_frame=end,
        target_fps=target_fps,
        source_fps=source_fps,
        # garment_mesh_name=garment_mesh_name,
    )

    print(f"Saved NPZ sequence: {npz_output_path}")

    return True

##############################################################################
# Main
##############################################################################
if __name__== "__main__":
    frame_start = None
    frame_end = None
    obj_prefix = "frame"
    target_fps = 30.0
    source_fps = None
    garment_mesh_name = None
    npz_output_path = None

    # Parse command-line arguments when invoked via:
    # blender --background --python smplx_anim_to_objs.py -- --input in.npz --output out.abc
    # blender --background --python smplx_anim_to_objs.py -- --input in.abc --output out_dir --obj_prefix body
    if bpy.app.background:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1:]  # get all args after "--"

            parser = argparse.ArgumentParser(description="Convert .npz to .abc, or .abc to OBJ sequence")

            parser.add_argument("--input", required=True, type=str, help="Path to .npz or .abc input file")
            parser.add_argument("--output", required=True, type=str, help="Output .abc path (.npz input) or OBJ output directory (.abc input)")
            parser.add_argument("--frame_start", required=False, type=int, default=None, help="Optional first frame to export for .abc input")
            parser.add_argument("--frame_end", required=False, type=int, default=None, help="Optional last frame to export for .abc input")
            parser.add_argument("--obj_prefix", required=False, type=str, default="frame", help="OBJ filename prefix for .abc input")
            parser.add_argument("--target_fps", required=False, type=float, default=30.0, help="Output OBJ sequence framerate for .abc input")
            parser.add_argument("--source_fps", required=False, type=float, default=None, help="Optional source Alembic framerate override")
            parser.add_argument("--garment_mesh_name", required=False, type=str, default=None, help="Mesh object name to save in NPZ (default: first imported mesh)")
            parser.add_argument("--npz_output", required=False, type=str, default=None, help="Optional NPZ output path (default: <output_dir>/<obj_prefix>_garment_seq.npz)")
            args = parser.parse_args(argv)
            smplx_animation_path = Path(args.input)
            output_path = Path(args.output)
            frame_start = args.frame_start
            frame_end = args.frame_end
            obj_prefix = args.obj_prefix
            target_fps = args.target_fps
            source_fps = args.source_fps
            garment_mesh_name = args.garment_mesh_name
            npz_output_path = args.npz_output

    start_time = time.perf_counter()
    input_suffix = smplx_animation_path.suffix.lower()

    if input_suffix == ".npz":
        _enable_smplx_addon_if_needed()
        print(f"Converting NPZ to Alembic: {smplx_animation_path} => {output_path}")
        convert_to_abc(smplx_animation_path, output_path)
    elif input_suffix == ".abc":
        print(f"Converting Alembic to OBJ sequence: {smplx_animation_path} => {output_path}")
        convert_abc_to_obj_sequence(
            abc_path=smplx_animation_path,
            output_dir=output_path,
            frame_start=frame_start,
            frame_end=frame_end,
            obj_prefix=obj_prefix,
            target_fps=target_fps,
            source_fps=source_fps,
            garment_mesh_name=garment_mesh_name,
            npz_output_path=output_path,
        )
    else:
        raise ValueError(f"Unsupported input format '{input_suffix}'. Use .npz or .abc input.")

    print(f"Finished. Time: {(time.perf_counter() - start_time):.1f}s")