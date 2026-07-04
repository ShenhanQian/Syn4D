import torch
import pause
import numpy as np
import os
import random
import argparse
from torchvision.transforms import ToPILImage

from utils import ImgNorm, todevice, inv, geotrf
from viser_visualizer_track import start_visualization
from syn4d_track import Syn4D_Track


def normalize_pointcloud_from_views(pts_list, norm_mode="avg_dis", valid_list=None, return_scale=False):
    """Normalize point clouds from multiple views, excluding invalid points from normalization."""
    assert all(pts.ndim >= 3 and pts.shape[-1] == 3 for pts in pts_list)

    norm_mode, dis_mode = norm_mode.split("_")

    # Concatenate all point clouds and valid masks if provided
    all_pts = torch.cat(pts_list, dim=1)
    if valid_list is not None:
        all_valid = torch.cat(valid_list, dim=1)
        valid_pts = all_pts[all_valid]  # Keep only valid points for norm calculation
    else:
        valid_pts = all_pts

    # Compute the distance to the origin for valid points
    dis = valid_pts.norm(dim=-1)

    # Apply distance transformation based on dis_mode
    if dis_mode == "dis":
        pass  # Do nothing
    elif dis_mode == "log1p":
        dis = torch.log1p(dis)
    elif dis_mode == "warp-log1p":
        log_dis = torch.log1p(dis)
        warp_factor = log_dis / dis.clip(min=1e-8)
        all_pts = all_pts * warp_factor.view(-1, 1)  # Warp the points with the warp factor
        dis = log_dis  # The final distance is now the log-transformed distance
    else:
        raise ValueError(f"Unsupported distance mode: {dis_mode}")

    # Apply different normalization modes
    if norm_mode == "avg":
        norm_factor = dis.mean()  # Compute mean distance of valid points
    elif norm_mode == "median":
        norm_factor = dis.median()  # Compute median distance of valid points
    else:
        raise ValueError(f"Unsupported normalization mode: {norm_mode}")

    norm_factor = norm_factor.clip(min=1e-8)  # Prevent division by zero

    # Normalize all point clouds
    if valid_list is not None:
        normalized_pts = [torch.where(valid.unsqueeze(-1), pts / norm_factor, pts)
                        for pts, valid in zip(pts_list, valid_list)]
    else:
        normalized_pts = [pts / norm_factor for pts in pts_list]

    if return_scale:
        return normalized_pts, norm_factor
    else:
        return normalized_pts


def convert_input_to_pred_format(views_seq0, views_seq1=None):
    views_seq0 = todevice(views_seq0, "cpu")
    views_seq1 = None if views_seq1 is None else todevice(views_seq1, "cpu")
    n = len(views_seq0)
    if views_seq1 is not None and n != len(views_seq1):
        raise ValueError(
            f"Two view sequences must have the same length, got {n} and {len(views_seq1)}."
        )
    output = dict()

    output["preds"] = [dict() for _ in range(n)]
    output["views"] = [dict() for _ in range(n)]

    inv_matrix_anchor = inv(views_seq0[0]["camera_pose"].float())
    for i in range(n):
        view0 = views_seq0[i]
        if "track" not in view0:
            print(f"warning: track not in views_seq0[{i}], we use pts3d instead")
            view0["track"] = view0["pts3d"]

        # Align sequence to the same global anchor (first frame of seq0)
        track0 = geotrf(inv_matrix_anchor, view0["track"].unsqueeze(0))
        pts0 = geotrf(inv_matrix_anchor, view0["pts3d"].unsqueeze(0))
        if views_seq1 is None:
            track_cat = track0.reshape(1, -1, 3)
            pts_cat = pts0.reshape(1, -1, 3)
        else:
            view1 = views_seq1[i]
            if "track" not in view1:
                print(f"warning: track not in views_seq1[{i}], we use pts3d instead")
                view1["track"] = view1["pts3d"]
            track1 = geotrf(inv_matrix_anchor, view1["track"].unsqueeze(0))
            pts1 = geotrf(inv_matrix_anchor, view1["pts3d"].unsqueeze(0))
            track_cat = torch.cat([track0.reshape(1, -1, 3), track1.reshape(1, -1, 3)], dim=1)
            pts_cat = torch.cat([pts0.reshape(1, -1, 3), pts1.reshape(1, -1, 3)], dim=1)

        output["preds"][i]["track_aligned_to_global"] = track_cat
        output["preds"][i]["pts3d_in_other_view"] = pts_cat
        output["preds"][i]["pts3d_local_aligned_to_global"] = pts_cat.clone()

        output["preds"][i]["conf"] = 10 * torch.ones(1, pts_cat.shape[1])
        output["preds"][i]["conf_local"] = 10 * torch.ones(1, pts_cat.shape[1])
        output["preds"][i]["conf_track"] = 10 * torch.ones(1, track_cat.shape[1])

        output["preds"][i]["extrinsic"] = inv_matrix_anchor @ view0["camera_pose"]
        output["preds"][i]["intrinsic"] = view0["camera_intrinsics"]
        if views_seq1 is not None:
            output["preds"][i]["extrinsic_secondary"] = inv_matrix_anchor @ view1["camera_pose"]
            output["preds"][i]["intrinsic_secondary"] = view1["camera_intrinsics"]
        output["preds"][i]["track_query_idx"] = torch.tensor(
            [views_seq0[0]["track_query_idx"] if "track_query_idx" in views_seq0[0] else 0]
        )

        output["views"][i]["img"] = view0["img"].unsqueeze(0)
        if views_seq1 is not None:
            output["views"][i]["img_secondary"] = view1["img"].unsqueeze(0)

    # Note that we did not count invalid area
    _, norm_factor = normalize_pointcloud_from_views(
        [output["preds"][i]["pts3d_in_other_view"] for i in range(n)],
        return_scale=True,
    )

    for i in range(n):
        output["preds"][i]["pts3d_in_other_view"] = output["preds"][i]["pts3d_in_other_view"]  / norm_factor
        output["preds"][i]["pts3d_local_aligned_to_global"] = output["preds"][i]["pts3d_local_aligned_to_global"] / norm_factor
        output["preds"][i]["track_aligned_to_global"] = output["preds"][i]["track_aligned_to_global"] / norm_factor
        output["preds"][i]["extrinsic"][:3, 3:] = output["preds"][i]["extrinsic"][:3, 3:] / norm_factor
        if "extrinsic_secondary" in output["preds"][i]:
            output["preds"][i]["extrinsic_secondary"][:3, 3:] = output["preds"][i]["extrinsic_secondary"][:3, 3:] / norm_factor

    return output

def parse_args():
    def _parse_scene_names(scene_args):
        if not scene_args:
            return None
        names = []
        for item in scene_args:
            names.extend([name.strip() for name in item.split(",") if name.strip()])
        return names or None

    def _parse_resolution(value):
        text = str(value).lower().replace(" ", "")
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        if "x" in text:
            width_s, height_s = text.split("x", 1)
            width, height = int(width_s), int(height_s)
        elif "," in text:
            width_s, height_s = text.split(",", 1)
            width, height = int(width_s), int(height_s)
        else:
            width = height = int(text)
        if width <= 0 or height <= 0:
            raise argparse.ArgumentTypeError("--resolution values must be positive")
        if width < height:
            raise argparse.ArgumentTypeError(
                "--resolution expects landscape order WIDTHxHEIGHT, or a single square size"
            )
        return (width, height)

    parser = argparse.ArgumentParser(description="Visualize tracking output with Viser.")
    parser.add_argument(
        "--dataset-root",
        dest="dataset_root",
        type=str,
        default="/mnt/c/bedlam2/images",
        help="Dataset root path on the current machine.",
    )
    parser.add_argument(
        "--metadata-root",
        type=str,
        default="/mnt/e",
        help="Optional metadata root.",
    )
    parser.add_argument(
        "--fallback-metadata-root",
        type=str,
        default="/mnt/d/data_objs/obj_glbs",
        help="Fallback metadata root used when files are missing from --metadata-root. Use an empty string to disable.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for the Viser server (use 0.0.0.0 for remote access).",
    )
    parser.add_argument("--port", type=int, default=8020, help="Viser server port.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Request and print a temporary public share URL.",
    )
    parser.add_argument(
        "--scene-name",
        action="append",
        default=[],
        help="Scene filter. Repeat the flag or pass comma-separated names.",
    )
    parser.add_argument(
        "--select-idx-view0",
        type=int,
        default=0,
        help="Dataset index for the primary sequence.",
    )
    parser.add_argument(
        "--select-idx-view1",
        type=int,
        default=None,
        help="Optional dataset index for the secondary sequence. Omit to visualize one sequence.",
    )
    parser.add_argument(
        "--track-query-idx",
        type=int,
        default=None,
        help="Override the dataset-selected reference/query frame. Omit to use the dataset default.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Frame stride used to sample visualization frames. Use 1 for stride-1 processed tracks.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Number of sampled frames to visualize.",
    )
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(512, 512),
        help="Visualization crop/resize resolution. Use a single size like 512 or WIDTHxHEIGHT like 512x384.",
    )
    parser.add_argument(
        "--rgb-source",
        choices=("auto", "png", "mp4"),
        default="auto",
        help="RGB source. auto uses png when available, otherwise mp4.",
    )
    parser.add_argument(
        "--tracking-format",
        choices=("auto", "dense", "compact_faceid", "safetensor"),
        default="auto",
        help="Assert the tracking schema. auto detects NPZ when present and falls back to safetensor shards.",
    )
    parser.add_argument(
        "--debug-faceid-timing",
        action="store_true",
        help="Print timing and distance stats for compact face-id reconstruction.",
    )
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be a positive integer")
    args.scene_name_list = _parse_scene_names(args.scene_name)
    return args


def main():
    args = parse_args()
    # dataset = Syn4D_Track(
    #     split="train", allow_repeat=False, dataset_location="/mnt/d/partial_syn4d",
    #     aug_crop=0, resolution=[((504, 378), 40)], transform=ImgNorm, min_interval=1, max_interval=1, debug=False, is_filter_dynamic=False, filter_dynamic_threshold=70
    # )

    dataset = Syn4D_Track(
        split="train", allow_repeat=False, dataset_root=args.dataset_root, metadata_root=args.metadata_root,
        fallback_metadata_root=args.fallback_metadata_root,
        scene_name_list=args.scene_name_list,
        track_query_idx=args.track_query_idx,
        strides=[args.stride],
        rgb_source=args.rgb_source,
        tracking_format=args.tracking_format,
        debug_faceid_timing=args.debug_faceid_timing,
        aug_crop=0, resolution=[(args.resolution, args.num_frames)], transform=ImgNorm, min_interval=1, max_interval=1, debug=False, is_filter_dynamic=False, filter_dynamic_threshold=70
    )

    select_idx_view0 = args.select_idx_view0
    select_idx_view1 = args.select_idx_view1
    print(f"len(dataset): {len(dataset)}")
    print(f"Selected indices: seq0={select_idx_view0}, seq1={select_idx_view1}")
    views_0 = dataset[select_idx_view0]
    views_1 = None if select_idx_view1 is None else dataset[select_idx_view1]
    print(f"track_query_idx (seq0): {views_0[0]['track_query_idx']}")
    if views_1 is not None and len(views_0) != len(views_1):
        raise RuntimeError(
            f"Selected sequences have different lengths: {len(views_0)} vs {len(views_1)}"
        )

    # save_dir = "develop/debug_syn4d"
    # os.makedirs(save_dir, exist_ok=True)
    # for v, view in enumerate(views):
    #     img = view["img"]
    #     if isinstance(img, torch.Tensor):
    #         img_denorm = ((img + 1.0) / 2.0).clamp(0, 1)
    #         img_uint8 = (img_denorm * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    #     else:
    #         img_uint8 = np.array(img) # Assume PIL or numpy
            
    #     out_img_path = os.path.join(save_dir, f"view_{v:03d}.png")
    #     cv2.imwrite(out_img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)) 

    #     # Depth visualization
    #     depth = view["depthmap"]
    #     if isinstance(depth, torch.Tensor):
    #         depth = depth.cpu().numpy()
        
    #     # Normalize depth for visualization
    #     depth_valid = depth > 0
    #     if depth_valid.sum() > 0:
    #         depth_min, depth_max = depth[depth_valid].min(), depth[depth_valid].max()
    #         if depth_max - depth_min > 1e-5:
    #             depth_norm = (depth - depth_min) / (depth_max - depth_min)
    #         else:
    #             depth_norm = np.zeros_like(depth)
            
    #         depth_uint8 = (depth_norm * 255).astype(np.uint8)
    #         depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            
    #         # Mask out invalid depth
    #         depth_colormap[~depth_valid] = 0
            
    #         out_depth_path = os.path.join(save_dir, f"view_{v:03d}_depth.png")
    #         cv2.imwrite(out_depth_path, depth_colormap)

    output = convert_input_to_pred_format(views_0, views_1)

    server = start_visualization(
        output=output,
        min_conf_thr_percentile=0,
        global_conf_thr_value_to_drop_view=1,
        host=args.host,
        port=args.port,
        point_size=0.0016,
    )

    print(f"Viser server started on http://{args.host}:{args.port}")
    if args.share:
        share_url = server.request_share_url()
        print(f"Share URL: {share_url}")

    pause.days(1)

if __name__ == '__main__':
    main()
