# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import datetime
import threading
import numpy as np
from tqdm.auto import tqdm
import imageio.v3 as iio
from matplotlib import cm
import cv2
from scipy import ndimage
import torch

import viser
import viser.transforms as tf

from utils import todevice


# ----------------- Helper Functions -----------------
def to_numpy(x):
    return todevice(x, "numpy")

def detect_sky_mask(img_rgb):
    """
    Detect sky pixels using HSV color space and morphological operations.
    Args:
        img_rgb: RGB image normalized to [-1, 1]
    Returns:
        Boolean mask (as int8) where True indicates non-sky pixels.
    """
    img = ((img_rgb + 1) * 127.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([105, 50, 140])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_light_blue = np.array([95, 5, 150])
    upper_light_blue = np.array([145, 100, 255])
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    lower_white = np.array([0, 0, 235])
    upper_white = np.array([180, 10, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = mask_blue | mask_light_blue | mask_white

    height = mask.shape[0]
    upper_third = int(height * 0.4)
    upper_region = hsv[:upper_third, :, :]
    mask[:upper_third, :] |= ((upper_region[:, :, 1] < 50) & (upper_region[:, :, 2] > 150))

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = mask.astype(bool)
    labels, num_labels = ndimage.label(mask)
    if num_labels > 0:
        top_row_labels = set(labels[0, :])
        top_row_labels.discard(0)
        if top_row_labels:
            mask = np.isin(labels, list(top_row_labels))
            labels, num_labels = ndimage.label(mask)
            if num_labels > 0:
                sizes = ndimage.sum(mask, labels, range(1, num_labels + 1))
                mask_size = mask.size
                big_enough = sizes > mask_size * 0.01
                mask = np.isin(labels, np.where(big_enough)[0] + 1)
    return (~mask).astype(np.int8)

def is_outdoor_scene(frame_data_list):
    sky_ratios = []
    for fd in frame_data_list:
        mask = fd.get('sorted_not_sky_global', np.ones(1))
        sky_ratio = 1.0 - np.mean(mask)
        sky_ratios.append(float(sky_ratio))
    significant = sum(1 for ratio in sky_ratios if ratio > 0.2)
    return significant >= len(sky_ratios) / 4

# ----------------- Update Handlers -----------------
# These functions are lightweight and respond only to their respective events.
def update_frame_visibility(server, frame_data_list, gui_timestep, num_frames, gui_show_global, gui_show_local, gui_show_track, gui_show_traj, gui_show_high_conf, gui_show_low_conf):
    current = int(gui_timestep.value)
    with server.atomic():
        for i in range(num_frames):
            fd = frame_data_list[i]
            # For simplicity, we show the frame if i = current.
            show_frame = (i == current)
            fd['frame_node'].visible = show_frame and show_frame
            fd['frustum_node'].visible = show_frame and show_frame
            if fd.get('frustum_node_secondary', None) is not None:
                fd['frustum_node_secondary'].visible = show_frame and show_frame
            fd['point_node_global'].visible = show_frame and gui_show_global.value
            fd['point_node_local'].visible = show_frame and gui_show_local.value
            fd['point_node_track'].visible = show_frame and gui_show_track.value
            fd['point_node_traj'].visible = show_frame and gui_show_traj.value
        server.flush()

def update_point_cloud_colors(server, frame_data_list, gui_timestep, gui_show_confidence_color, gui_rainbow_color_option, gui_show_global, gui_show_local, gui_show_track):
    with server.atomic():
        for i in range(len(frame_data_list)):
            fd = frame_data_list[i]
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global']
                colors_local = fd['colors_confidence_local']
                colors_track = fd['colors_confidence_track']
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global']
                colors_local = fd['colors_rainbow_local']
                colors_track = fd['colors_rainbow_track']
            else:
                colors_global = fd['colors_rgb_global']
                colors_local = fd['colors_rgb_local']
                colors_track = fd['colors_rgb_track']
            fd['point_node_global'].colors = colors_global if gui_show_global.value else []
            fd['point_node_local'].colors = colors_local if gui_show_local.value else []
            fd['point_node_track'].colors = colors_track if gui_show_track.value else []
            server.flush()

def update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option):
    for i in range(len(frame_data_list)):
        fd = frame_data_list[i]
        total_global = len(fd['sorted_pts3d_global'])
        total_local = len(fd['sorted_pts3d_local'])
        total_track = len(fd['sorted_pts3d_track'])

        # Calculate number of points to show based on the percentile
        num_global = max(1, int(total_global * (100 - gui_min_conf_percentile.value) / 100))
        num_local = max(1, int(total_local * (100 - gui_min_conf_percentile.value) / 100))
        num_track = max(1, int(total_track * (100 - gui_min_conf_percentile.value) / 100))

        # Apply sky masking if enabled
        if gui_mask_sky.value:
            mask_global = fd['sorted_not_sky_global'][:num_global]
            mask_local = fd['sorted_not_sky_local'][:num_local]
            mask_track = fd['sorted_not_sky_track'][:num_track]
            # Filter points based on the mask
            pts3d_global = fd['sorted_pts3d_global'][:num_global][mask_global > 0]
            pts3d_local = fd['sorted_pts3d_local'][:num_local][mask_local > 0]
            pts3d_track = fd['sorted_pts3d_track'][:num_track][mask_track > 0]
            
            # Select the appropriate colors based on the active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_confidence_local'][:num_local][mask_local > 0]
                colors_track = fd['colors_confidence_track'][:num_track][mask_track > 0]
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_rainbow_local'][:num_local][mask_local > 0]
                colors_track = fd['colors_rainbow_track'][:num_track][mask_track > 0]
            else:
                colors_global = fd['colors_rgb_global'][:num_global][mask_global > 0]
                colors_local = fd['colors_rgb_local'][:num_local][mask_local > 0]
                colors_track = fd['colors_rgb_track'][:num_track][mask_track > 0]
        else:
            pts3d_global = fd['sorted_pts3d_global'][:num_global]
            pts3d_local = fd['sorted_pts3d_local'][:num_local]
            pts3d_track = fd['sorted_pts3d_track'][:num_track]

            # Select the appropriate colors based on the active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global'][:num_global]
                colors_local = fd['colors_confidence_local'][:num_local]
                colors_track = fd['colors_confidence_track'][:num_track]
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global'][:num_global]
                colors_local = fd['colors_rainbow_local'][:num_local]
                colors_track = fd['colors_rainbow_track'][:num_track]
            else:
                colors_global = fd['colors_rgb_global'][:num_global]
                colors_local = fd['colors_rgb_local'][:num_local]
                colors_track = fd['colors_rgb_track'][:num_track]

        # Update point clouds
        fd['point_node_global'].points = pts3d_global
        fd['point_node_local'].points = pts3d_local
        fd['point_node_track'].points = pts3d_track
        # update colors
        fd['point_node_global'].colors = colors_global
        fd['point_node_local'].colors = colors_local
        fd['point_node_track'].colors = colors_track
                
        server.flush()

# ----------------- Playback Loop -----------------
def playback_loop(gui_playing, gui_timestep, num_frames, gui_framerate):
    while True:
        if gui_playing.value:
            gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
        time.sleep(1.0 / float(gui_framerate.value))

def bind_update(widget, update_func):
    widget.on_update(lambda _: update_func())

# ----------------- Main Visualization Function -----------------
def start_visualization(
    output,
    min_conf_thr_percentile=10,
    global_conf_thr_value_to_drop_view=1.5,
    host="127.0.0.1",
    port=8020,
    point_size=0.0004,
):
    server = viser.ViserServer(host=host, port=port)
    server.gui.set_panel_label("Show Controls")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)

    @server.on_client_connect
    def on_client_connect(client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = (-0.00141163, -0.01910395, -0.06794288)
            client.camera.look_at = (-0.00352821, -0.01143425, 0.0154939)
        client.flush()

    poses_c2w = [
        to_numpy(pred["extrinsic"].cpu() if isinstance(pred["extrinsic"], torch.Tensor) else pred["extrinsic"])
        for pred in output["preds"]
    ]
    poses_c2w_secondary = [
        to_numpy(
            pred["extrinsic_secondary"].cpu()
            if isinstance(pred["extrinsic_secondary"], torch.Tensor)
            else pred["extrinsic_secondary"]
        ) if "extrinsic_secondary" in pred else None
        for pred in output["preds"]
    ]
    gt_focals = [
        float(
            to_numpy(pred["intrinsic"].cpu() if isinstance(pred["intrinsic"], torch.Tensor) else pred["intrinsic"])[0, 0]
        )
        for pred in output["preds"]
    ]
    gt_focals_secondary = [
        float(
            to_numpy(
                pred["intrinsic_secondary"].cpu()
                if isinstance(pred["intrinsic_secondary"], torch.Tensor)
                else pred["intrinsic_secondary"]
            )[0, 0]
        ) if "intrinsic_secondary" in pred else None
        for pred in output["preds"]
    ]

    server.scene.set_up_direction((0.0, -1.0, 0.0))
    server.scene.world_axes.visible = False

    num_frames = len(output['preds'])
    frame_data_list = []
    cumulative_pts = []

    # ----------------- Grouped GUI Controls -----------------
    with server.gui.add_folder("Point and Camera Options", expand_by_default=False):
        gui_point_size = server.gui.add_slider("Point Size", min=1e-6, max=0.005, step=1e-5, initial_value=point_size)
        gui_traj_width = server.gui.add_slider("Traj Width", min=0.01, max=2.0, step=0.01, initial_value=1.5)
        gui_traj_percent = server.gui.add_slider("Traj Percent", min=1, max=100, step=1, initial_value=50)
        gui_traj_len = server.gui.add_slider("Traj Length (Frames)", min=1, max=num_frames, step=1, initial_value=min(10, num_frames))
        gui_traj_motion_percent = server.gui.add_slider("Traj Motion Percent", min=1, max=100, step=1, initial_value=100)
        gui_frustum_size_percent = server.gui.add_slider("Camera Size (%)", min=0.1, max=10.0, step=0.1, initial_value=2.0)
        gui_mask_sky = server.gui.add_checkbox("Mask Sky", False)
        gui_show_confidence_color = server.gui.add_checkbox("Show Confidence", False)
        gui_rainbow_color_option = server.gui.add_checkbox("Color by View", False)
        gui_keep_points_percent = server.gui.add_slider(
            "Visualization Point Keep (%)", min=1, max=100, step=1, initial_value=100
        )

    with server.gui.add_folder("Playback Options", expand_by_default=False):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0)
        gui_next_frame = server.gui.add_button("Next Frame")
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_displayall = server.gui.add_checkbox("Display All", False)
        gui_framerate = server.gui.add_slider("FPS", min=0.25, max=60, step=0.25, initial_value=10)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("0.5", "1", "10", "20", "30", "60"))

    with server.gui.add_folder("Pointmap Head Options", expand_by_default=False):
        gui_show_global = server.gui.add_checkbox("Global", False)
        gui_show_local = server.gui.add_checkbox("Local", True)
        gui_show_track = server.gui.add_checkbox("Track", False)
        gui_show_traj = server.gui.add_checkbox("Traj", False)

    with server.gui.add_folder("Confidence Options", expand_by_default=False):
        gui_show_high_conf = server.gui.add_checkbox("Show High-Conf Views", True)
        gui_show_low_conf = server.gui.add_checkbox("Show Low-Conf Views", False)
        gui_global_conf_threshold = server.gui.add_slider("High/Low Conf Threshold", min=1.0, max=12.0, step=0.1, initial_value=global_conf_thr_value_to_drop_view)
        gui_min_conf_percentile = server.gui.add_slider("Per-View Conf Percentile", min=0, max=100, step=1, initial_value=min_conf_thr_percentile)

    button_render_gif = server.gui.add_button("Render a GIF")
    button_render_dynamic = server.gui.add_button("Render Dynamic")
    button_render_static = server.gui.add_button("Render Static")
    button_save_png = server.gui.add_button("Save Current View as PNG")

    @gui_next_frame.on_click
    def next_frame(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def prev_frame(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def playing_update(_):
        state = gui_playing.value
        gui_timestep.disabled = state
        gui_next_frame.disabled = state
        gui_prev_frame.disabled = state

    @gui_displayall.on_update
    def playall_update(_):
        state = gui_displayall.value

        gui_playing.value = False
        gui_playing.disabled = state
        gui_timestep.disabled = state
        gui_next_frame.disabled = state
        gui_prev_frame.disabled = state

        current = int(gui_timestep.value)

        with server.atomic():
            for i in range(num_frames):
                cur_shortcut = (state or i==current)

                fd = frame_data_list[i]
                fd['frame_node'].visible = True and cur_shortcut
                # Set frustum visibility based on confidence settings
                if fd['is_high_confidence']:
                    fd['frustum_node'].visible = gui_show_high_conf.value and cur_shortcut
                else:
                    fd['frustum_node'].visible = gui_show_low_conf.value and cur_shortcut
                if fd.get('frustum_node_secondary', None) is not None:
                    if fd['is_high_confidence']:
                        fd['frustum_node_secondary'].visible = gui_show_high_conf.value and cur_shortcut
                    else:
                        fd['frustum_node_secondary'].visible = gui_show_low_conf.value and cur_shortcut
                fd['point_node_global'].visible = gui_show_global.value and cur_shortcut
                fd['point_node_local'].visible = gui_show_local.value and cur_shortcut
                fd['point_node_track'].visible = gui_show_track.value and cur_shortcut
        server.flush()

    @gui_framerate_options.on_click
    def fps_options(_):
        gui_framerate.value = float(gui_framerate_options.value)

    server.scene.add_frame("/cams", show_axes=False)

    # ----------------- Frame Processing -----------------
    def _match_array_length(arr, target_len):
        if arr.shape[0] == target_len:
            return arr
        if arr.shape[0] > target_len:
            return arr[:target_len]
        reps = int(np.ceil(target_len / max(1, arr.shape[0])))
        return np.tile(arr, (reps, 1))[:target_len] if arr.ndim == 2 else np.tile(arr, reps)[:target_len]

    def _normalize_track_query_idx(track_query_idx, num_frames):
        if isinstance(track_query_idx, torch.Tensor):
            track_query_idx = track_query_idx.detach().cpu().flatten().tolist()
        elif isinstance(track_query_idx, (list, tuple)):
            track_query_idx = list(track_query_idx)
        else:
            track_query_idx = [int(track_query_idx)]
        track_query_idx = [int(idx) for idx in track_query_idx if 0 <= int(idx) < num_frames]
        if not track_query_idx:
            track_query_idx = [0]
        return track_query_idx

    first_pred = output['preds'][0]
    track_query_idx_list = _normalize_track_query_idx(
        first_pred.get('track_query_idx', 0), num_frames
    )
    if isinstance(first_pred, dict) and "track_multi" in first_pred:
        track_query_idx_list = track_query_idx_list[: first_pred["track_multi"].shape[1]]
    primary_query_idx = track_query_idx_list[0]
    print(f"track_query_idx: {track_query_idx_list}")

    view_query_primary = output['views'][primary_query_idx]
    img_rgb_orig_track_0 = to_numpy(view_query_primary['img'].cpu().squeeze().permute(1,2,0))
    not_sky_mask_track_0 = detect_sky_mask(img_rgb_orig_track_0).flatten().astype(np.int8)
    img_rgb_flat_track_0 = img_rgb_orig_track_0.reshape(-1, 3)
    if 'img_secondary' in view_query_primary:
        img_rgb_orig_track_1 = to_numpy(view_query_primary['img_secondary'].cpu().squeeze().permute(1,2,0))
        not_sky_mask_track_1 = detect_sky_mask(img_rgb_orig_track_1).flatten().astype(np.int8)
        img_rgb_flat_track_1 = img_rgb_orig_track_1.reshape(-1, 3)
        img_rgb_flat_track = np.concatenate([img_rgb_flat_track_0, img_rgb_flat_track_1], axis=0)
        not_sky_mask_track = np.concatenate([not_sky_mask_track_0, not_sky_mask_track_1], axis=0)
    else:
        img_rgb_flat_track = img_rgb_flat_track_0
        not_sky_mask_track = not_sky_mask_track_0

    for i in tqdm(range(num_frames)):
        pred = output['preds'][i]
        view = output['views'][i]

        img_rgb_orig_0 = to_numpy(view['img'].cpu().squeeze().permute(1,2,0))
        not_sky_mask_0 = detect_sky_mask(img_rgb_orig_0).flatten().astype(np.int8)

        pts3d_global = to_numpy(pred['pts3d_in_other_view'].cpu().squeeze()).reshape(-1, 3)
        conf_global = to_numpy(pred['conf'].cpu().squeeze()).flatten()
        pts3d_local = to_numpy(pred['pts3d_local_aligned_to_global'].cpu().squeeze()).reshape(-1, 3)
        conf_local = to_numpy(pred['conf_local'].cpu().squeeze()).flatten()
        pts3d_track = to_numpy(pred['track_aligned_to_global'].cpu().squeeze()).reshape(-1, 3)
        conf_track = to_numpy(pred['conf_track'].cpu().squeeze()).flatten()

        img_rgb = img_rgb_orig_0
        img_rgb_flat_0 = img_rgb.reshape(-1, 3)
        if 'img_secondary' in view:
            img_rgb_orig_1 = to_numpy(view['img_secondary'].cpu().squeeze().permute(1,2,0))
            not_sky_mask_1 = detect_sky_mask(img_rgb_orig_1).flatten().astype(np.int8)
            img_rgb_flat_1 = img_rgb_orig_1.reshape(-1, 3)
            img_rgb_flat = np.concatenate([img_rgb_flat_0, img_rgb_flat_1], axis=0)
            not_sky_mask = np.concatenate([not_sky_mask_0, not_sky_mask_1], axis=0)
        else:
            img_rgb_flat = img_rgb_flat_0
            not_sky_mask = not_sky_mask_0

        # Ensure color/mask buffers match the current merged point count.
        img_rgb_flat = _match_array_length(img_rgb_flat, pts3d_global.shape[0])
        not_sky_mask = _match_array_length(not_sky_mask, pts3d_global.shape[0])
        img_rgb_flat_track_matched = _match_array_length(img_rgb_flat_track, pts3d_track.shape[0])
        not_sky_mask_track_matched = _match_array_length(not_sky_mask_track, pts3d_track.shape[0])

        cumulative_pts.append(pts3d_global)

        sort_idx_global = np.argsort(-conf_global)
        sorted_conf_global = conf_global[sort_idx_global]
        sorted_pts3d_global = pts3d_global[sort_idx_global]
        sorted_img_rgb_global = img_rgb_flat[sort_idx_global]
        sorted_not_sky_global = not_sky_mask[sort_idx_global]

        sort_idx_local = np.argsort(-conf_local)
        sorted_conf_local = conf_local[sort_idx_local]
        sorted_pts3d_local = pts3d_local[sort_idx_local]
        sorted_img_rgb_local = img_rgb_flat[sort_idx_local]
        sorted_not_sky_local = not_sky_mask[sort_idx_local]

        # For track we do not sort by confidence (we need temporal ordering),
        # but we still want aligned lengths.
        sort_idx_track = np.arange(sort_idx_local.shape[0])
        sorted_conf_track = conf_track[sort_idx_track]
        sorted_pts3d_track = pts3d_track[sort_idx_track]
        sorted_img_rgb_track = img_rgb_flat_track_matched[sort_idx_track]
        sorted_not_sky_track = not_sky_mask_track_matched[sort_idx_track]

        track_multi = pred.get("track_multi")
        if track_multi is None:
            track_multi = pred["track_aligned_to_global"].unsqueeze(1)
            conf_track_multi = pred["conf_track"].unsqueeze(1)

        sorted_pts3d_track_multi = []
        for q_i in range(track_multi.shape[1]):
            pts3d_track_q = to_numpy(track_multi[:, q_i].cpu().squeeze()).reshape(-1, 3)
            sorted_pts3d_track_q = pts3d_track_q[sort_idx_track]
            sorted_pts3d_track_multi.append(sorted_pts3d_track_q)

        colors_rgb_global = ((sorted_img_rgb_global + 1) * 127.5).astype(np.uint8) / 255.0
        colors_rgb_local = ((sorted_img_rgb_local + 1) * 127.5).astype(np.uint8) / 255.0
        colors_rgb_track = ((sorted_img_rgb_track + 1) * 127.5).astype(np.uint8) / 255.0

        conf_norm_global = (sorted_conf_global - sorted_conf_global.min()) / (sorted_conf_global.max() - sorted_conf_global.min() + 1e-8)
        conf_norm_local = (sorted_conf_local - sorted_conf_local.min()) / (sorted_conf_local.max() - sorted_conf_local.min() + 1e-8)
        conf_norm_track = (sorted_conf_track - sorted_conf_track.min()) / (sorted_conf_track.max() - sorted_conf_track.min() + 1e-8)
        colormap = cm.turbo
        colors_confidence_global = colormap(conf_norm_global)[:, :3]
        colors_confidence_local = colormap(conf_norm_local)[:, :3]
        colors_confidence_track = colormap(conf_norm_track)[:, :3]

        def rainbow_color(n, total):
            import colorsys
            hue = n / total
            return colorsys.hsv_to_rgb(hue, 1.0, 1.0)

        rainbow_color_for_frame = rainbow_color(i, num_frames)
        colors_rainbow_global = np.tile(rainbow_color_for_frame, (sorted_pts3d_global.shape[0], 1))
        colors_rainbow_local = np.tile(rainbow_color_for_frame, (sorted_pts3d_local.shape[0], 1))
        colors_rainbow_track = np.tile(rainbow_color_for_frame, (sorted_pts3d_track.shape[0], 1))

        max_conf_global = conf_global.max()
        is_high_confidence = max_conf_global >= gui_global_conf_threshold.value

        c2w = poses_c2w[i]
        c2w_secondary = poses_c2w_secondary[i]
        height, width = view['img'].shape[2], view['img'].shape[3]
        focal_length = gt_focals[i]
        focal_length_secondary = gt_focals_secondary[i]
        img_rgb_reshaped = img_rgb.reshape(height, width, 3)
        img_rgb_normalized = ((img_rgb_reshaped + 1) * 127.5).astype(np.uint8)
        img_downsampled = img_rgb_normalized[::4, ::4]

        frame_data = {
            'sorted_pts3d_global': sorted_pts3d_global,
            'colors_rgb_global': colors_rgb_global,
            'colors_confidence_global': colors_confidence_global,
            'colors_rainbow_global': colors_rainbow_global,
            'sorted_pts3d_local': sorted_pts3d_local,
            'colors_rgb_local': colors_rgb_local,
            'colors_confidence_local': colors_confidence_local,
            'colors_rainbow_local': colors_rainbow_local,
            'sorted_not_sky_global': sorted_not_sky_global,
            'sorted_not_sky_local': sorted_not_sky_local,
            'max_conf_global': float(max_conf_global),
            'sorted_pts3d_track': sorted_pts3d_track,
            'sorted_pts3d_track_multi': sorted_pts3d_track_multi,
            'colors_rgb_track': colors_rgb_track,
            'colors_confidence_track': colors_confidence_track,
            'colors_rainbow_track': colors_rainbow_track,
            'sorted_not_sky_track': sorted_not_sky_track,
            'is_high_confidence': is_high_confidence,
            'c2w': c2w,
            'c2w_secondary': c2w_secondary,
            'height': height,
            'width': width,
            'focal_length': focal_length,
            'focal_length_secondary': focal_length_secondary,
            'img_downsampled': img_downsampled,
            'rainbow_color': rainbow_color_for_frame,
        }

        frame_data_list.append(frame_data)

    # Percentile for scene extent calculation (10th to 90th percentile by default)
    extent_percentile = 80
    cumulative_pts_combined = np.concatenate(cumulative_pts, axis=0)
    # Calculate percentiles for each coordinate
    min_coords = np.percentile(cumulative_pts_combined, 100 - extent_percentile, axis=0)
    max_coords = np.percentile(cumulative_pts_combined, extent_percentile, axis=0)
    scene_extent = max_coords - min_coords
    max_extent = np.max(scene_extent)

    # ----------------- Create Visualization Nodes -----------------
    query_count = len(track_query_idx_list)
    sorted_query_pairs = sorted(enumerate(track_query_idx_list), key=lambda x: x[1])
    prev_query_idx_by_qi = {q_i: None for q_i in range(query_count)}
    for pos, (q_i, q_idx) in enumerate(sorted_query_pairs):
        if pos == 0:
            prev_query_idx_by_qi[q_i] = None
        else:
            prev_query_idx_by_qi[q_i] = sorted_query_pairs[pos - 1][1]

    traj_frame_ranges = []
    for q_i, q_idx in enumerate(track_query_idx_list):
        if query_count == 1:
            lower = 0
            upper = num_frames - 1
        else:
            lower = max(q_idx - 2, 0)
            upper = min(q_idx + 2, num_frames - 1)
        traj_frame_ranges.append((lower, upper))

    track_valid_mask_list = []
    moving_mask_list = []
    norms_list = []
    for q_i, q_idx in enumerate(track_query_idx_list):
        base_frame_idx = int(q_idx)
        track_points_first = frame_data_list[base_frame_idx]['sorted_pts3d_track_multi'][q_i]
        # Filter invalid track points in training data; fall back if all are filtered.
        track_valid_mask = ~(np.all(np.abs(track_points_first) < 0.001, axis=1))
        if track_valid_mask.sum() == 0:
            track_valid_mask = np.ones(track_points_first.shape[0], dtype=bool)
        track_points_first = track_points_first[track_valid_mask]

        delta = np.zeros_like(track_points_first)
        for i in tqdm(range(num_frames - 1, num_frames)):
            track_points_prev = frame_data_list[i]['sorted_pts3d_track_multi'][q_i][track_valid_mask] # cur
            delta += abs(track_points_first - track_points_prev)
        norms = np.linalg.norm(delta, axis=1)
        threshold = np.percentile(norms, 0) if norms.size > 0 else 0.0
        moving_mask = norms >= threshold

        track_valid_mask_list.append(track_valid_mask)
        moving_mask_list.append(moving_mask)
        norms_list.append(norms)

    def _build_traj_lines_for_frame(frame_idx):
        lines_flat_list = []
        colors_flat_list = []

        for q_i, q_idx in enumerate(track_query_idx_list):
            lower, upper = traj_frame_ranges[q_i]
            norms = norms_list[q_i]
            total_tracks = norms.shape[0]
            if total_tracks == 0:
                continue
            keep_count = max(1, int(round(total_tracks * (gui_traj_motion_percent.value / 100.0))))
            top_idx = np.argsort(norms)[-keep_count:]
            motion_top_mask = np.zeros(total_tracks, dtype=bool)
            motion_top_mask[top_idx] = True
            motion_mask = moving_mask_list[q_i] & motion_top_mask

            lines = []
            traj_len = int(gui_traj_len.value)
            start_idx = max(frame_idx - traj_len + 1, 0)
            for j in range(start_idx, frame_idx + 1):
                if query_count > 1 and not (lower <= j <= upper):
                    continue
                track_points_prev = frame_data_list[max(j - 1, 0)]['sorted_pts3d_track_multi'][q_i][track_valid_mask_list[q_i]][motion_mask]  # start
                track_points_next = frame_data_list[j]['sorted_pts3d_track_multi'][q_i][track_valid_mask_list[q_i]][motion_mask]  # end
                line = np.stack((track_points_prev, track_points_next), axis=1)  # N, 2, 3
                lines.append(line)
            if not lines:
                continue
            lines = np.stack(lines, axis=1)  # N, t, 2, 3
            lines = lines[::10]
            num_lines, len_t = lines.shape[:2]

            # Color trajectory by temporal order so it changes gradually.
            # Older segments use cooler colors, newer segments warmer colors.
            temporal_colors = cm.turbo(np.linspace(0.05, 0.95, len_t))[:, :3]  # t, 3
            traj_colors = np.tile(temporal_colors[np.newaxis, :, np.newaxis, :], (num_lines, 1, 2, 1))

            lines_full = lines
            colors_full = traj_colors

            # Apply traj percent slicing here (slider logic)
            if num_lines == 0:
                lines_flat = lines_full.reshape(-1, 2, 3)
                colors_flat = colors_full.reshape(-1, 2, 3)
            else:
                keep_count = max(1, int(round(num_lines * (gui_traj_percent.value / 100.0))))
                keep_idx = np.linspace(0, num_lines - 1, keep_count, dtype=int)
                lines_subset = lines_full[keep_idx]
                colors_subset = colors_full[keep_idx]
                lines_flat = lines_subset.reshape(-1, 2, 3)
                colors_flat = colors_subset.reshape(-1, 2, 3)

            lines_flat_list.append(lines_flat)
            colors_flat_list.append(colors_flat)

        if not lines_flat_list:
            empty = np.zeros((0, 2, 3))
            return empty, empty, empty, empty

        lines_flat = np.concatenate(lines_flat_list, axis=0)
        colors_flat = np.concatenate(colors_flat_list, axis=0)
        lines_full = lines_flat
        colors_full = colors_flat

        return lines_full, colors_full, lines_flat, colors_flat

    for i in tqdm(range(num_frames)):
        fd = frame_data_list[i]
        frame_node = server.scene.add_frame(f"/cams/t{i}", show_axes=False, visible=False)

        lines_full, colors_full, lines, traj_colors = _build_traj_lines_for_frame(i)

        point_node_traj = server.scene.add_line_segments(
            name = f"/traj/t{i}",
            points = lines,
            colors = traj_colors,
            line_width=gui_traj_width.value,
            visible=False,
        )

        point_node_global = server.scene.add_point_cloud(
            name=f"/pts3d_global/t{i}",
            points=fd['sorted_pts3d_global'],
            colors=fd['colors_rgb_global'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )
        point_node_local = server.scene.add_point_cloud(
            name=f"/pts3d_local/t{i}",
            points=fd['sorted_pts3d_local'],
            colors=fd['colors_rgb_local'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )
        point_node_track = server.scene.add_point_cloud(
            name=f"/pts3d_track/t{i}",
            points=fd['sorted_pts3d_track'],
            colors=fd['colors_rgb_track'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )

        rotation_matrix = fd['c2w'][:3, :3]
        position = fd['c2w'][:3, 3]
        rotation_quaternion = tf.SO3.from_matrix(rotation_matrix).wxyz
        try:
            fov = 2 * np.arctan2(fd['height'] / 2, fd['focal_length'])
        except Exception as e:
            print(f"Error calculating FOV: {e}")
            fov = 60
        aspect_ratio = fd['width'] / fd['height']
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)

        frustum_node = server.scene.add_camera_frustum(
            name=f"/cams/t{i}/frustum",
            fov=fov,
            aspect=aspect_ratio,
            scale=frustum_scale,
            color=fd['rainbow_color'],
            image=fd['img_downsampled'],
            wxyz=rotation_quaternion,
            position=position,
            visible=False,
        )
        frustum_node_secondary = None
        if fd.get('c2w_secondary', None) is not None and fd.get('focal_length_secondary', None) is not None:
            rotation_matrix_secondary = fd['c2w_secondary'][:3, :3]
            position_secondary = fd['c2w_secondary'][:3, 3]
            rotation_quaternion_secondary = tf.SO3.from_matrix(rotation_matrix_secondary).wxyz
            try:
                fov_secondary = 2 * np.arctan2(fd['height'] / 2, fd['focal_length_secondary'])
            except Exception:
                fov_secondary = fov
            frustum_node_secondary = server.scene.add_camera_frustum(
                name=f"/cams_secondary/t{i}/frustum",
                fov=fov_secondary,
                aspect=aspect_ratio,
                scale=frustum_scale,
                color=np.clip(np.array(fd['rainbow_color']) * 0.6 + 0.4, 0.0, 1.0),
                image=fd['img_downsampled'],
                wxyz=rotation_quaternion_secondary,
                position=position_secondary,
                visible=False,
            )

        fd['frame_node'] = frame_node
        fd['point_node_global'] = point_node_global
        fd['point_node_local'] = point_node_local
        fd['point_node_track'] = point_node_track
        fd['point_node_traj'] = point_node_traj
        fd['frustum_node'] = frustum_node
        fd['frustum_node_secondary'] = frustum_node_secondary
        fd['traj_lines_full'] = lines_full
        fd['traj_colors_full'] = colors_full

    # Initially set all nodes hidden
    for fd in frame_data_list:
        fd['frame_node'].visible = False
        fd['point_node_global'].visible = False
        fd['point_node_local'].visible = False
        fd['point_node_track'].visible = False
        fd['point_node_traj'].visible = False
        fd['frustum_node'].visible = False
        if fd.get('frustum_node_secondary', None) is not None:
            fd['frustum_node_secondary'].visible = False
    server.flush()

    # Initialize timestep to show first frames and disable playing
    gui_timestep.value = 0
    gui_playing.value = False

    # Scene type detection and sky masking initialization
    is_outdoor = is_outdoor_scene(frame_data_list)
    gui_mask_sky.value = False # we dont mask sky for tracking

    print("\nScene type detection:")
    sky_ratios = [1.0 - np.mean(fd['sorted_not_sky_global']) for fd in frame_data_list]
    significant = sum(1 for r in sky_ratios if r > 0.2)
    print(f"- Found {significant}/{len(sky_ratios)} frames with significant sky presence (>20% sky pixels)")
    print(f"- Scene classified as: {'outdoor' if is_outdoor else 'indoor'}, setting mask_sky to {is_outdoor}")

    # Initial visibility setup
    with server.atomic():
        for i in range(num_frames):
            fd = frame_data_list[i]
            if i == gui_timestep.value:
                fd['frame_node'].visible = True
                fd['frustum_node'].visible = True if fd['is_high_confidence'] else False
                if fd.get('frustum_node_secondary', None) is not None:
                    fd['frustum_node_secondary'].visible = True if fd['is_high_confidence'] else False
            else:
                fd['frame_node'].visible = False
                fd['frustum_node'].visible = False
                if fd.get('frustum_node_secondary', None) is not None:
                    fd['frustum_node_secondary'].visible = False
            
            # Set up initial points with sky masking if needed
            pts3d_global = fd['sorted_pts3d_global']
            pts3d_local = fd['sorted_pts3d_local']
            pts3d_track = fd['sorted_pts3d_track']
            
            # Select appropriate colors based on active color option
            if gui_show_confidence_color.value:
                colors_global = fd['colors_confidence_global']
                colors_local = fd['colors_confidence_local']
                colors_track = fd['colors_confidence_track']
            elif gui_rainbow_color_option.value:
                colors_global = fd['colors_rainbow_global']
                colors_local = fd['colors_rainbow_local']
                colors_track = fd['colors_rainbow_track']
            else:
                colors_global = fd['colors_rgb_global']
                colors_local = fd['colors_rgb_local']
                colors_track = fd['colors_rgb_track']
            
            if is_outdoor and gui_mask_sky.value:  # Apply sky masking if outdoor scene
                mask_global = fd['sorted_not_sky_global']
                mask_local = fd['sorted_not_sky_local']
                mask_track = fd['sorted_not_sky_track']
                pts3d_global = pts3d_global[mask_global > 0]
                pts3d_local = pts3d_local[mask_local > 0]
                pts3d_track = pts3d_track[mask_track > 0]
                colors_global = colors_global[mask_global > 0]
                colors_local = colors_local[mask_local > 0]
                colors_track = colors_track[mask_track > 0]
            
            # Update point clouds
            fd['point_node_global'].points = pts3d_global
            fd['point_node_local'].points = pts3d_local
            fd['point_node_track'].points = pts3d_track
            fd['point_node_global'].colors = colors_global
            fd['point_node_local'].colors = colors_local
            fd['point_node_track'].colors = colors_track
            if i == gui_timestep.value:
                fd['point_node_global'].visible = gui_show_global.value
                fd['point_node_local'].visible = gui_show_local.value
                fd['point_node_track'].visible = gui_show_track.value
                fd['point_node_traj'].visible = gui_show_traj.value
            else:
                fd['point_node_global'].visible = False
                fd['point_node_local'].visible = False
                fd['point_node_track'].visible = False
                fd['point_node_traj'].visible = False

    server.flush()

    # ----------------- GUI Callback Updates -----------------
    @gui_timestep.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                # we only show current frame
                if i == current:
                    fd['frame_node'].visible = True
                    # Set frustum visibility based on confidence settings
                    if fd['is_high_confidence']:
                        fd['frustum_node'].visible = gui_show_high_conf.value
                    else:
                        fd['frustum_node'].visible = gui_show_low_conf.value
                    if fd.get('frustum_node_secondary', None) is not None:
                        if fd['is_high_confidence']:
                            fd['frustum_node_secondary'].visible = gui_show_high_conf.value
                        else:
                            fd['frustum_node_secondary'].visible = gui_show_low_conf.value
                    fd['point_node_global'].visible = gui_show_global.value
                    fd['point_node_local'].visible = gui_show_local.value
                    fd['point_node_track'].visible = gui_show_track.value
                    fd['point_node_traj'].visible = gui_show_traj.value
                else:
                    fd['frame_node'].visible = False
                    fd['frustum_node'].visible = False
                    if fd.get('frustum_node_secondary', None) is not None:
                        fd['frustum_node_secondary'].visible = False
                    fd['point_node_global'].visible = False
                    fd['point_node_local'].visible = False
                    fd['point_node_track'].visible = False
                    fd['point_node_traj'].visible = False
        server.flush()

    @gui_keep_points_percent.on_update
    def _(_):
        """Dynamically subsample per-point data according to the keep% slider."""
        keep_percent = float(gui_keep_points_percent.value)
        keep_percent = max(1.0, min(100.0, keep_percent))

        with server.atomic():
            for i, fd in enumerate(frame_data_list):
                def _subsample_2d(arr):
                    if arr is None:
                        return arr
                    N_local = arr.shape[0]
                    if N_local <= 1:
                        return arr
                    target_local = max(1, int(round(N_local * keep_percent / 100.0)))
                    if target_local >= N_local:
                        return arr
                    idx_local = np.linspace(0, N_local - 1, target_local, dtype=int)
                    return arr[idx_local]

                def _subsample_1d(arr):
                    if arr is None:
                        return arr
                    N_local = arr.shape[0]
                    if N_local <= 1:
                        return arr
                    target_local = max(1, int(round(N_local * keep_percent / 100.0)))
                    if target_local >= N_local:
                        return arr
                    idx_local = np.linspace(0, N_local - 1, target_local, dtype=int)
                    return arr[idx_local]

                # Global
                fd['sorted_pts3d_global'] = _subsample_2d(fd['sorted_pts3d_global'])
                fd['colors_rgb_global'] = _subsample_2d(fd['colors_rgb_global'])
                fd['colors_confidence_global'] = _subsample_2d(fd['colors_confidence_global'])
                fd['colors_rainbow_global'] = _subsample_2d(fd['colors_rainbow_global'])
                fd['sorted_not_sky_global'] = _subsample_1d(fd['sorted_not_sky_global'])

                # Local
                fd['sorted_pts3d_local'] = _subsample_2d(fd['sorted_pts3d_local'])
                fd['colors_rgb_local'] = _subsample_2d(fd['colors_rgb_local'])
                fd['colors_confidence_local'] = _subsample_2d(fd['colors_confidence_local'])
                fd['colors_rainbow_local'] = _subsample_2d(fd['colors_rainbow_local'])
                fd['sorted_not_sky_local'] = _subsample_1d(fd['sorted_not_sky_local'])

                # Track single-query
                fd['sorted_pts3d_track'] = _subsample_2d(fd['sorted_pts3d_track'])
                fd['colors_rgb_track'] = _subsample_2d(fd['colors_rgb_track'])
                fd['colors_confidence_track'] = _subsample_2d(fd['colors_confidence_track'])
                fd['colors_rainbow_track'] = _subsample_2d(fd['colors_rainbow_track'])
                fd['sorted_not_sky_track'] = _subsample_1d(fd['sorted_not_sky_track'])

                fd['point_node_global'].points = fd['sorted_pts3d_global']
                fd['point_node_global'].colors = fd['colors_rgb_global']
                fd['point_node_local'].points = fd['sorted_pts3d_local']
                fd['point_node_local'].colors = fd['colors_rgb_local']
                fd['point_node_track'].points = fd['sorted_pts3d_track']
                fd['point_node_track'].colors = fd['colors_rgb_track']

        server.flush()

    @gui_point_size.on_update
    def _(_):
        with server.atomic():
            for fd in frame_data_list:
                fd['point_node_global'].point_size = gui_point_size.value
                fd['point_node_local'].point_size = gui_point_size.value
                fd['point_node_track'].point_size = gui_point_size.value
        server.flush()

    @gui_traj_width.on_update
    def _(_):
        with server.atomic():
            for fd in frame_data_list:
                fd['point_node_traj'].line_width = gui_traj_width.value
        server.flush()

    @gui_traj_percent.on_update
    def _(_):
        with server.atomic():
            for i, fd in enumerate(frame_data_list):
                lines_full, colors_full, lines, colors = _build_traj_lines_for_frame(i)
                fd['traj_lines_full'] = lines_full
                fd['traj_colors_full'] = colors_full
                fd['point_node_traj'].points = lines
                fd['point_node_traj'].colors = colors
        server.flush()

    @gui_traj_len.on_update
    def _(_):
        with server.atomic():
            for i, fd in enumerate(frame_data_list):
                lines_full, colors_full, lines, colors = _build_traj_lines_for_frame(i)
                fd['traj_lines_full'] = lines_full
                fd['traj_colors_full'] = colors_full
                fd['point_node_traj'].points = lines
                fd['point_node_traj'].colors = colors
        server.flush()

    @gui_traj_motion_percent.on_update
    def _(_):
        with server.atomic():
            for i, fd in enumerate(frame_data_list):
                lines_full, colors_full, lines, colors = _build_traj_lines_for_frame(i)
                fd['traj_lines_full'] = lines_full
                fd['traj_colors_full'] = colors_full
                fd['point_node_traj'].points = lines
                fd['point_node_traj'].colors = colors
        server.flush()

    @gui_frustum_size_percent.on_update
    def _(_):
        frustum_scale = max_extent * (gui_frustum_size_percent.value / 100.0)
        with server.atomic():
            for fd in frame_data_list:
                fd['frustum_node'].scale = frustum_scale
                if fd.get('frustum_node_secondary', None) is not None:
                    fd['frustum_node_secondary'].scale = frustum_scale
        server.flush()

    @gui_show_confidence_color.on_update
    def _(_):
        # Make options mutually exclusive
        if gui_show_confidence_color.value and gui_rainbow_color_option.value:
            gui_rainbow_color_option.value = False
        
        # Update colors for all visible points
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_rainbow_color_option.on_update
    def _(_):
        # Make options mutually exclusive
        if gui_rainbow_color_option.value and gui_show_confidence_color.value:
            gui_show_confidence_color.value = False
            
        # Update colors for all visible points
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_min_conf_percentile.on_update
    def _(_):
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_mask_sky.on_update
    def _(_):
        # For each visible frame, update filtering if mask sky changes.
        update_points_filtering(server, frame_data_list, gui_timestep, gui_min_conf_percentile, 
                               gui_mask_sky, gui_show_confidence_color, gui_rainbow_color_option)

    @gui_show_global.on_update
    def _(_):
        with server.atomic():
            current = int(gui_timestep.value)
            for i in range(int(gui_timestep.value)+1):
                if i == current:
                    frame_data_list[i]['point_node_global'].visible = gui_show_global.value
        server.flush()

    @gui_show_local.on_update
    def _(_):
        with server.atomic():
            current = int(gui_timestep.value)
            for i in range(int(gui_timestep.value)+1):
                if i == current:
                    frame_data_list[i]['point_node_local'].visible = gui_show_local.value
        server.flush()

    @gui_show_track.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(int(gui_timestep.value)+1):
                if i == current:
                    frame_data_list[i]['point_node_track'].visible = gui_show_track.value
        server.flush()
    
    @gui_show_traj.on_update
    def _(_):
        current = int(gui_timestep.value)
        with server.atomic():
            for i in range(int(gui_timestep.value)+1):
                if i == current:
                    frame_data_list[i]['point_node_traj'].visible = gui_show_traj.value
        server.flush()

    @gui_show_high_conf.on_update
    def _(_):
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i == int(gui_timestep.value):
                    # Hide frustum and points if high confidence views are disabled
                    if fd['is_high_confidence'] and gui_show_high_conf.value:
                        fd['frustum_node'].visible = gui_show_high_conf.value
                        if fd.get('frustum_node_secondary', None) is not None:
                            fd['frustum_node_secondary'].visible = gui_show_high_conf.value
                        fd['point_node_global'].visible = gui_show_global.value and gui_show_high_conf.value
                        fd['point_node_local'].visible = gui_show_local.value and gui_show_high_conf.value
                    else:
                        fd['frustum_node'].visible = False  # Hide if not high confidence
                        if fd.get('frustum_node_secondary', None) is not None:
                            fd['frustum_node_secondary'].visible = False
                        fd['point_node_global'].visible = False  # Hide if not high confidence
                        fd['point_node_local'].visible = False  # Hide if not high confidence
        server.flush()

    @gui_show_low_conf.on_update
    def _(_):
        with server.atomic():
            for i in range(num_frames):
                fd = frame_data_list[i]
                if i == int(gui_timestep.value):
                    # Hide frustum and points if low confidence views are disabled
                    if not fd['is_high_confidence'] and gui_show_low_conf.value:
                        fd['frustum_node'].visible = gui_show_low_conf.value
                        if fd.get('frustum_node_secondary', None) is not None:
                            fd['frustum_node_secondary'].visible = gui_show_low_conf.value
                        fd['point_node_global'].visible = gui_show_global.value and gui_show_low_conf.value
                        fd['point_node_local'].visible = gui_show_local.value and gui_show_low_conf.value
                    else:
                        fd['frustum_node'].visible = False  # Hide if high confidence
                        if fd.get('frustum_node_secondary', None) is not None:
                            fd['frustum_node_secondary'].visible = False
                        fd['point_node_global'].visible = False  # Hide if high confidence
                        fd['point_node_local'].visible = False  # Hide if high confidence
        server.flush()

    @gui_global_conf_threshold.on_update
    def _(_):
        for fd in frame_data_list:
            fd['is_high_confidence'] = fd['max_conf_global'] >= gui_global_conf_threshold.value
        server.flush()

    # ----------------- Start Playback Loop -----------------
    def local_playback_loop():
        while True:
            if gui_playing.value:
                gui_timestep.value = (int(gui_timestep.value) + 1) % num_frames
            time.sleep(1.0 / float(gui_framerate.value))
    playback_thread = threading.Thread(target=local_playback_loop)
    playback_thread.start()

    def _normalize_render_frame(frame, target_hw=None):
        img = np.asarray(frame)
        if img.ndim > 3:
            img = np.squeeze(img)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3:
            if img.shape[2] == 4:
                img = img[..., :3]
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
        if target_hw is not None and img.shape[:2] != target_hw:
            img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
        if np.issubdtype(img.dtype, np.floating):
            if img.size > 0 and float(np.nanmax(img)) <= 1.0:
                img = np.clip(img, 0.0, 1.0) * 255.0
            else:
                img = np.clip(img, 0.0, 255.0)
            img = img.astype(np.uint8)
        elif img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _append_render_frame(images, frame, target_hw_box):
        normalized = _normalize_render_frame(frame, target_hw=target_hw_box[0])
        if target_hw_box[0] is None:
            target_hw_box[0] = normalized.shape[:2]
        images.append(normalized)

    @button_render_gif.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        fps = gui_framerate.value
        if client is None:
            print("Error: No client connected.")
            return
        try:
            # 获取当前相机参数
            cam_pos = np.array(client.camera.position)
            look_at = np.array(client.camera.look_at)
            up = np.array(client.camera.up_direction)
            # 计算场景中心
            all_pts = np.concatenate([fd['sorted_pts3d_global'] for fd in frame_data_list], axis=0)
            center = np.mean(all_pts, axis=0)
            # 计算半径和轨迹
            radius = np.linalg.norm(cam_pos - center)
            circle_radius = radius * 0.1  # 小圆，半径为当前视角到中心距离的10%
            n_frames = 100
            images = []
            target_hw_box = [None]
            for i in range(n_frames):
                theta = 2 * np.pi * i / n_frames
                # 在当前视角附近做圆周运动
                # 以当前相机的up方向和cam_pos->center方向确定圆的平面
                forward = (center - cam_pos)
                forward = forward / np.linalg.norm(forward)
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                # 圆周点
                offset = np.cos(theta) * right * circle_radius + np.sin(theta) * up * circle_radius
                new_pos = cam_pos + offset
                client.camera.position = tuple(new_pos)
                client.camera.look_at = tuple(center)
                client.camera.up_direction = tuple(up)
                client.flush()
                time.sleep(0.05)
                img = client.get_render(height=720, width=1280)
                _append_render_frame(images, img, target_hw_box)

            # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # save_dir = f"visualization_circle_{now}"
            # os.makedirs(save_dir, exist_ok=True)
            # for i, img in enumerate(images):
            #     iio.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), img)

            gif_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps, loop=0)
            client.send_file_download("visualization.gif", gif_bytes)
        except Exception as e:
            print(f"Error while rendering trajectory GIF: {e}")

    @button_render_dynamic.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            print("Error: No client connected.")
            return
        try:
            # Create a scene serializer to record the dynamic reconstruction as a .viser file
            serializer = server.get_scene_serializer()

            images = []
            target_hw_box = [None]
            original_timestep = gui_timestep.value
            original_playing = gui_playing.value
            gui_playing.value = False
            fps = gui_framerate.value
            # Time step for both rendering and serialization
            dt = 1.0 / float(fps) if fps > 0 else 1.0 / 30.0
            for i in range(num_frames):
                gui_timestep.value = i
                # Advance "time" in the serialized recording
                serializer.insert_sleep(dt)
                time.sleep(dt)
                # Use higher resolution for video export
                _append_render_frame(images, client.get_render(height=1080, width=1920), target_hw_box)

            # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # save_dir = f"visualization_dynamic_{now}"
            # os.makedirs(save_dir, exist_ok=True)
            # for i, img in enumerate(images):
            #     iio.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), img)

            # Create and save high-resolution MP4 instead of GIF
            video_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps)
            client.send_file_download("visualization_dynamic.gif", video_bytes)
            # video_bytes = iio.imwrite("<bytes>", images, extension=".mp4", fps=fps)
            # client.send_file_download("visualization_dynamic.mp4", video_bytes)

            # Also export the same dynamic reconstruction as a .viser recording
            viser_bytes = serializer.serialize()  # bytes representing the embedded visualization
            # Use a timestamped filename to avoid overwriting
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            viser_filename = f"visualization_dynamic_{now}.viser"
            client.send_file_download(viser_filename, viser_bytes)

            gui_timestep.value = original_timestep
            gui_playing.value = original_playing
        except Exception as e:
            print(f"Error while rendering dynamic video: {e}")

    @button_render_static.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            print("Error: No client connected.")
            return
        try:
            images = []
            target_hw_box = [None]
            original_timestep = gui_timestep.value
            original_playing = gui_playing.value
            gui_playing.value = False
            fps = gui_framerate.value
            for i in range(num_frames):
                gui_timestep.value = i
                # Show all previous pointmaps and frustums up to and including i
                with server.atomic():
                    for j in range(num_frames):
                        fd = frame_data_list[j]
                        if j <= i:
                            fd['point_node_global'].visible = gui_show_global.value
                            fd['point_node_local'].visible = gui_show_local.value
                            fd['point_node_track'].visible = gui_show_track.value
                            fd['point_node_traj'].visible = gui_show_traj.value
                            fd['frame_node'].visible = True
                            fd['frustum_node'].visible = True  # Show all previous frustums
                            if fd.get('frustum_node_secondary', None) is not None:
                                fd['frustum_node_secondary'].visible = True
                        else:
                            fd['point_node_global'].visible = False
                            fd['point_node_local'].visible = False
                            fd['point_node_track'].visible = False
                            fd['point_node_traj'].visible = False
                            fd['frame_node'].visible = False
                            fd['frustum_node'].visible = False
                            if fd.get('frustum_node_secondary', None) is not None:
                                fd['frustum_node_secondary'].visible = False
                server.flush()
                time.sleep(0.1)
                # Use higher resolution for video export
                _append_render_frame(images, client.get_render(height=1080, width=1920), target_hw_box)
            # Restore visibility to only the current timestep
            with server.atomic():
                for j in range(num_frames):
                    fd = frame_data_list[j]
                    if j == original_timestep:
                        fd['point_node_global'].visible = gui_show_global.value
                        fd['point_node_local'].visible = gui_show_local.value
                        fd['point_node_track'].visible = gui_show_track.value
                        fd['point_node_traj'].visible = gui_show_traj.value
                        # Restore frustum visibility based on confidence settings
                        if fd['is_high_confidence']:
                            fd['frustum_node'].visible = gui_show_high_conf.value
                            if fd.get('frustum_node_secondary', None) is not None:
                                fd['frustum_node_secondary'].visible = gui_show_high_conf.value
                        else:
                            fd['frustum_node'].visible = gui_show_low_conf.value
                            if fd.get('frustum_node_secondary', None) is not None:
                                fd['frustum_node_secondary'].visible = gui_show_low_conf.value
                    else:
                        fd['point_node_global'].visible = False
                        fd['point_node_local'].visible = False
                        fd['point_node_track'].visible = False
                        fd['point_node_traj'].visible = False
                        fd['frustum_node'].visible = False
                        if fd.get('frustum_node_secondary', None) is not None:
                            fd['frustum_node_secondary'].visible = False
            server.flush()

            # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # save_dir = f"visualization_static_{now}"
            # os.makedirs(save_dir, exist_ok=True)
            # for i, img in enumerate(images):
            #     iio.imwrite(os.path.join(save_dir, f"frame_{i:04d}.png"), img)

            # Create and save high-resolution MP4 instead of GIF
            try:
                video_bytes = iio.imwrite("<bytes>", images, extension=".mp4", fps=fps)
                client.send_file_download("visualization_static.mp4", video_bytes)
            except Exception as e_video:
                # Fallback to GIF if MP4 export fails for any reason
                print(f"MP4 export failed, falling back to GIF: {e_video}")
                gif_bytes = iio.imwrite("<bytes>", images, extension=".gif", fps=fps, loop=0)
                client.send_file_download("visualization_static.gif", gif_bytes)
            gui_timestep.value = original_timestep
            gui_playing.value = original_playing
        except Exception as e:
            import traceback
            print(f"Error while rendering static video: {e}")
            print("Traceback:")
            print(traceback.format_exc())

    @button_save_png.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            print("Error: No client connected.")
            return
        try:
            # Get current render from client
            image = client.get_render(height=720, width=1280)

            # Generate filename with timestamp
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualization_{now}.png"

            # Convert to bytes
            img_bytes = iio.imwrite("<bytes>", image, extension=".png")

            # Send file to client for download
            client.send_file_download(filename, img_bytes)

        except Exception as e:
            import traceback
            print(f"Error while saving PNG: {e}")
            print("Traceback:")
            print(traceback.format_exc())

    # public_url = server.request_share_url()
    return server

# Example usage:
# server = start_visualization(output=your_output_dict, min_conf_thr_percentile=10, global_conf_thr_value_to_drop_view=1.5, port=8020)
