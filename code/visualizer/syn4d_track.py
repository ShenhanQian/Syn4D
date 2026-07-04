import os
import sys
import time
from typing import Any
import torch
import numpy as np
import glob
import OpenEXR as exr
import Imath
import cv2
from pathlib import Path
import pandas as pd
import csv
from dataclasses import dataclass
import json
import tarfile
import math
from collections import OrderedDict
from tqdm import tqdm

from base_dataset import BaseStereoDynamicViewDataset
from utils import depthmap_to_absolute_camera_coordinates, imread_cv2


def infer_object_mask_offset(mask_types, object_count):
    object_mask_ids = []
    for mask_type in mask_types:
        if not str(mask_type).startswith("obj_"):
            continue
        try:
            object_mask_ids.append(int(str(mask_type).split("_", 1)[1]))
        except Exception:
            continue

    object_mask_ids = sorted(set(object_mask_ids))
    if object_count <= 0 or not object_mask_ids:
        return 0
    if all(idx in object_mask_ids for idx in range(object_count)):
        return 0
    if all((idx + 1) in object_mask_ids for idx in range(object_count)):
        return 1
    return object_mask_ids[0]


def get_stride_distribution(strides, dist_type='uniform'):

    # input strides sorted by descreasing order by default
    
    if dist_type == 'uniform':
        dist = np.ones(len(strides)) / len(strides)
    elif dist_type == 'exponential':
        lambda_param = 1.0
        dist = np.exp(-lambda_param * np.arange(len(strides)))
    elif dist_type.startswith('linear'): # e.g., linear_1_2
        try:
            start, end = map(float, dist_type.split('_')[1:])
            dist = np.linspace(start, end, len(strides))
        except ValueError:
            raise ValueError(f'Invalid linear distribution format: {dist_type}')
    else:
        raise ValueError('Unknown distribution type %s' % dist_type)

    # normalize to sum to 1
    return dist / np.sum(dist)

@dataclass
class SequenceBody:
    subject: str
    body_path: str
    clothing_path: str
    hair_path: str
    animation_path: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    start_frame: int
    texture_body: str
    texture_clothing: str
    texture_clothing_overlay: str
    haircolor_path: str
    shoe: str

@dataclass
class ActorPose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    shift: float = 0.0

def read_seq_csv(csv_path, csv_rows=None):
    group_info_list = []

    def get_float(row, key, default=0.0):
        value = row.get(key, default)
        if value is None or value == "":
            return float(default)
        return float(value)

    def parse_comment(comment):
        config = {}
        if not comment:
            return config
        for value in comment.split(";"):
            if not value or "=" not in value:
                continue
            key, val = value.split("=", 1)
            config[key] = val
        return config

    if csv_rows is None:
        with open(csv_path, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            csv_rows = list(csv_reader)  # Skip header

    current_group = None
    for row in csv_rows:
        row_type = row.get("Type")
        if row_type == "Comment":
            continue

        if row_type == "Group":
            if current_group is not None:
                group_info_list.append(current_group)

            camera_pose = ActorPose(
                float(row["X"]),
                float(row["Y"]),
                float(row["Z"]),
                float(row["Yaw"]),
                float(row["Pitch"]),
                float(row["Roll"]),
                get_float(row, "Shift", 0.0),
            )
            group_config = parse_comment(row.get("Comment", ""))

            cameraroot_location = None
            if (
                "cameraroot_x" in group_config
                and "cameraroot_y" in group_config
                and "cameraroot_z" in group_config
            ):
                cameraroot_location = {
                    "x": float(group_config["cameraroot_x"]),
                    "y": float(group_config["cameraroot_y"]),
                    "z": float(group_config["cameraroot_z"]),
                }

            current_group = {
                "group_config": group_config,
                "camera_pose": camera_pose,
                "sequence_name": group_config.get("sequence_name"),
                "sequence_frames": int(group_config["frames"])
                if "frames" in group_config
                else None,
                "hdri": group_config.get("hdri"),
                "camera_hfov": float(group_config["camera_hfov"])
                if "camera_hfov" in group_config
                else None,
                "time": float(group_config["time"]) if "time" in group_config else None,
                "cameraroot_yaw": float(group_config["cameraroot_yaw"])
                if "cameraroot_yaw" in group_config
                else None,
                "cameraroot_location": cameraroot_location,
                "bodies": [],
                "objects": [],
            }
            continue

        if current_group is None:
            continue

        if row_type == "Body":
            body = row["Body"]
            x = float(row["X"])
            y = float(row["Y"])
            z = float(row["Z"])
            yaw = float(row["Yaw"])
            pitch = float(row["Pitch"])
            roll = float(row["Roll"])

            body_config = parse_comment(row.get("Comment", ""))
            start_frame = int(body_config.get("start_frame", 0))

            subject = "undefined"
            animation_id = "undefined"
            if body.startswith("moyo"):
                subject = body.split("_", maxsplit=1)[0]
                animation_id = body
            elif "_" in body:
                subject = body.rsplit("_", maxsplit=1)[0]
                animation_id = body.rsplit("_", maxsplit=1)[1]

            body_info = {
                "index": int(row["Index"]) if row.get("Index") else None,
                "body": body,
                "subject": subject,
                "animation_id": animation_id,
                "pose": ActorPose(x, y, z, yaw, pitch, roll),
                "start_frame": start_frame,
                "texture_body": body_config.get("texture_body"),
                "texture_clothing": body_config.get("texture_clothing"),
                "texture_clothing_overlay": body_config.get("texture_clothing_overlay"),
                "hair": body_config.get("hair"),
                "haircolor": body_config.get("haircolor"),
                "shoe": body_config.get("shoe"),
                "body_config": body_config,
            }
            current_group["bodies"].append(body_info)
            continue

        if row_type == "Object":
            x = float(row["X"])
            y = float(row["Y"])
            z = float(row["Z"])
            yaw = float(row["Yaw"])
            pitch = float(row["Pitch"])
            roll = float(row["Roll"])
            shift = get_float(row, "Shift", 0.0)

            object_config = parse_comment(row.get("Comment", ""))
            per_frame_motion = None
            per_frame_motion_raw = str(row.get("PerFrameMotion", "")).strip()
            if per_frame_motion_raw:
                try:
                    per_frame_motion = json.loads(per_frame_motion_raw)
                except Exception:
                    per_frame_motion = None
            object_info = {
                "index": int(row["Index"]) if row.get("Index") else None,
                "body": row.get("Body"),
                "object_id": object_config.get("object_id"),
                "object_group": object_config.get("object_group"),
                "anim_name": object_config.get("anim_name"),
                "pose": ActorPose(x, y, z, yaw, pitch, roll, shift),
                "per_frame_motion": per_frame_motion,
                "object_config": object_config,
            }
            current_group["objects"].append(object_info)

    if current_group is not None:
        group_info_list.append(current_group)

    return group_info_list

def exr_to_array(filepath: Path):
    exrfile = exr.InputFile(filepath.as_posix())
    raw_bytes = exrfile.channel('Depth', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

SCENE_NAME_LIST = [
    # "arena",
    # "castle",
    # "cyber_bald",
    # "hospital",
    # "middleeast_bald",
    # "planet_bald",
    # "space_bald",
    # "winter",
    # "bigoffice_v1",
    # "countryside",
    # "office_bald",
    # "scifiroom_bald",
    # "village",
    # "flying_group",
    # "post_bald",
    "undefine_monocular_1_1_starterpack_obj3",
]

invalid_seqs = [
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000042",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000059",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000079",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000978",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000081",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000268",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000089",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000189",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000034",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000889",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000293",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000067",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000904",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000434",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000044",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000013",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000396",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000012",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000082",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000120",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000324",
    "20221013_3_250_batch01hand_static_bigOffice_seq_000038",
    "20221012_3-10_500_batch01hand_zoom_highSchoolGym_seq_000486",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000421",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000226",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000012",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000149",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000311",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000080",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000122",
    "20221012_3-10_500_batch01hand_zoom_highSchoolGym_seq_000079",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000077",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000095",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000062",
    "20221013_3_250_batch01hand_static_bigOffice_seq_000015",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000095",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000119",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000297",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000011",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000196",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000316",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000283",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000085",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000287",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000163",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000804",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000842",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000027",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000182",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000982",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000029",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000031",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000025",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000250",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000785",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000069",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000122",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000246",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000352",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000425",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000192",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000900",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000043",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000063",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000096",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000091",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000013",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000309",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000114",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000969",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000361",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000267",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000083",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000383",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000890",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000003",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000045",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000317",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000076",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000082",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000907",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000279",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000076",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000004",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000061",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000811",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000800",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000841",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000794",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000308",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000064",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000284",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000752",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000269",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000036",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000419",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000290",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000322",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000818",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000327",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000326",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000002",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000060",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000348",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000059",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000016",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000817",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000332",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000094",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000193",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000779",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000177",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000368",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000023",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000024",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000310",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000086",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000038",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000071",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000768",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000017",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000053",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000097",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000856",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000827",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000161",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000084",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000106",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000207",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000007",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000013",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000251",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000796",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000105",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000251",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000046",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000334",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000453",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000373",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000283",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000249",
]

hdri_scenes = [
    "20221010_3_1000_batch01hand",
    "20221017_3_1000_batch01hand",
    "20221018_3-8_250_batch01hand",
    "20221019_3_250_highbmihand",
]


class MultiRootPath:
    def __init__(self, roots, branch=Path()):
        self.roots = [Path(root) for root in roots if root]
        self.branch = Path(branch)

    def __truediv__(self, child):
        return MultiRootPath(self.roots, self.branch / child)

    def candidates(self):
        return [root / self.branch for root in self.roots]

    def resolve(self):
        for candidate in self.candidates():
            if candidate.exists():
                return candidate
        tried = ", ".join(str(candidate) for candidate in self.candidates())
        raise FileNotFoundError(f"Could not find metadata path. Tried: {tried}")


def build_metadata_path(metadata_root, fallback_metadata_root=None):
    roots = [metadata_root]
    if fallback_metadata_root:
        roots.append(fallback_metadata_root)
    return MultiRootPath(roots)


class Syn4D_Track(BaseStereoDynamicViewDataset):
    def __init__(self,
                 *args, 
                 dataset_root=None,
                 metadata_root=None,
                 fallback_metadata_root="/mnt/d/data_objs/obj_glbs",
                 scene_name_list=None,
                 track_query_idx=None,
                 dset='training',
                 use_augs=False,
                 S=16, # length of video
                 N=16,
                 strides=[5],
                 clip_step=2,
                 verbose=False,
                 dist_type=None,
                 clip_step_last_skip = 0,
                 min_interval=1,
                 max_interval=1,
                 debug=False,
                 rgb_source="auto",
                 tracking_format="auto",
                 debug_faceid_timing=False,
                 caption='Estimate the 3D location of each pixel',
                 **kwargs
                 ):
        self.metadata_root = metadata_root
        self.rgb_source = str(rgb_source or "auto").lower()
        if self.rgb_source not in ("auto", "png", "mp4"):
            raise ValueError(f"Unsupported rgb_source={rgb_source!r}; use auto, png, or mp4")
        self.tracking_format = str(tracking_format or "auto").lower()
        if self.tracking_format not in ("auto", "dense", "compact_faceid", "safetensor"):
            raise ValueError(
                f"Unsupported tracking_format={tracking_format!r}; use auto, dense, compact_faceid, or safetensor"
        )
        self.debug_faceid_timing = bool(debug_faceid_timing)
        self._tracking_safetensor_cache = {}
        self._env_mask_safetensor_cache = {}
        self._mp4_capture_cache = OrderedDict()
        self._mp4_capture_cache_size = 4
        self.metadata_path = build_metadata_path(metadata_root, fallback_metadata_root)
        self.root_smpl_npz_path = self.metadata_path / "bedlam2_smpl_npz"
        self.root_obj_npz_path = self.metadata_path / "new_weight_bone"
        self.cloth_tar_root = (
            self.metadata_path / "b2_assetdata_download" / "clothing" / "npz"
        ).resolve()
        cloth_json_path = os.path.join(self.cloth_tar_root, "archive_map.json")
        with open(cloth_json_path, 'r') as f:
            cloth_json = json.load(f)
        self.cloth_json = cloth_json
        self.cloth_npz_to_tar = {}
        for tar_name, npz_entries in cloth_json.items():
            for entry in npz_entries:
                self.cloth_npz_to_tar[entry] = tar_name
            
        self.caption = caption
        dataset_location = dataset_root
        annotation_root = dataset_location
        # Prefer explicit split from caller; keep dset in sync for legacy code.
        if "split" in kwargs:
            dset = kwargs["split"]
        print('loading BEDLAM dataset...')
        if "split" not in kwargs:
            kwargs["split"] = dset
        if "allow_repeat" not in kwargs:
            kwargs["allow_repeat"] = False
        # consume optional debugging flag so it won't leak to base class kwargs
        kwargs.pop("debug", None)
        super().__init__(*args, **kwargs)
        self.dataset_label = 'bedlam'
        self.split = dset
        self.video = True
        self.is_metric = True
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.S = S # stride
        self.N = N # min num points
        self.verbose = verbose

        self.use_augs = use_augs
        self.dset = dset
        self.dset = ''
        # If caller does not provide a list, keep current module-level default behavior.
        self.scene_name_list = SCENE_NAME_LIST if scene_name_list is None else scene_name_list

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.seg_paths = []
        self.track_npz_paths = []
        self.annotation_paths = []
        self.rgb_filenames = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides
        self.track_query_idx_override = track_query_idx

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location))
        be_seq_info_dict = {}
        transformation_json_dict = {}
        transformation_scale_dict = {}

        def scene_file(scene, preferred, fallback):
            preferred_path = os.path.join(scene, preferred)
            if os.path.exists(preferred_path):
                return preferred_path
            fallback_path = os.path.join(scene, fallback)
            if os.path.exists(fallback_path):
                return fallback_path
            raise FileNotFoundError(f"Missing {preferred} or {fallback} under {scene}")

        for subdir in self.subdirs:
            for scene in glob.glob(os.path.join(subdir, "*/")):
                scene_name = scene.split('/')[-2]
                if self.scene_name_list and scene_name not in self.scene_name_list:
                    continue
                be_seq_csv_path = scene_file(scene, "be_seq.csv", "be_seq_global_motion.csv")
                transform_json_path = scene_file(
                    scene,
                    "be_seq_transform_channels.json",
                    "be_seq_global_motion_transform_channels.json",
                )
                be_seq_info_dict[scene_name] = read_seq_csv(be_seq_csv_path)
                transformation_json = json.load(open(transform_json_path))
                transformation_json_dict[scene_name] = transformation_json

                # Build sequence_id -> [scale...] map sorted by sequence_body_index.
                scale_map = {}
                for item in transformation_json:
                    sequence_id = str(item.get("sequence_id", ""))
                    sequence_body_index = item.get("sequence_body_index", -1)
                    transform_channels = item.get("transform_channels", {})
                    map_camera_target_actor_location_x = float(item.get("map_camera_target_actor_location_x", 0.0))
                    map_camera_target_actor_location_y = float(item.get("map_camera_target_actor_location_y", 0.0))
                    map_camera_target_actor_location_z = float(item.get("map_camera_target_actor_location_z", 0.0))
                    map_camera_target_actor_location = np.array([map_camera_target_actor_location_x, map_camera_target_actor_location_y, map_camera_target_actor_location_z], dtype=np.float32)
                    map_camera_target_actor_location = map_camera_target_actor_location / 100
                    scale_x = float(transform_channels.get("scale_x", 1.0))
                    scale_map.setdefault(sequence_id, []).append((sequence_body_index, scale_x, map_camera_target_actor_location))
                for sequence_id, scale_items in scale_map.items():
                    scale_items.sort(key=lambda x: x[0] if isinstance(x[0], int) else 10**9)
                    scale_map[sequence_id] = [(scale, map_camera_target_actor_location) for _, scale, map_camera_target_actor_location in scale_items]
                transformation_scale_dict[scene_name] = scale_map
                png_root = os.path.join(scene, 'png')
                mp4_root = os.path.join(scene, 'mp4')
                use_png_index = self.rgb_source in ("auto", "png") and os.path.isdir(png_root)
                use_mp4_index = (
                    (self.rgb_source == "mp4" or (self.rgb_source == "auto" and not use_png_index))
                    and os.path.isdir(mp4_root)
                )
                if use_png_index:
                    seq_iter = glob.glob(os.path.join(png_root, "*/"))
                elif use_mp4_index:
                    seq_bases = set()
                    for mp4_file in glob.glob(os.path.join(mp4_root, "*.mp4")):
                        camera_seq = os.path.splitext(os.path.basename(mp4_file))[0]
                        seq_bases.add("_".join(camera_seq.split("_")[:-1]))
                    seq_iter = [os.path.join(mp4_root, seq_base) for seq_base in sorted(seq_bases)]
                else:
                    seq_iter = []

                for seq in seq_iter:
                    seq_name_all = os.path.basename(str(seq).rstrip("/"))
                    if use_mp4_index:
                        seq_name = seq_name_all
                    else:
                        seq_name = seq_name_all.split('_')[:-1]
                        seq_name = '_'.join(seq_name)

                    scene_name = Path(seq).parents[1].name
                    scene_name_no_fps = scene_name
                    if scene_name_no_fps + '_' + seq_name in invalid_seqs:
                        continue
                    if scene_name_no_fps in hdri_scenes:
                        continue
                    if "closeup" in scene_name_no_fps:
                        continue
                    if "orbit_stadium" in scene_name_no_fps:
                        continue
                    if "pitchUp52" in scene_name_no_fps:
                        continue
                    seq = os.path.join(scene, 'png' if use_png_index else 'mp4', seq_name)
                    if seq not in self.sequences:
                        self.sequences.append(seq)
        self.sequences = sorted(self.sequences)
        self.be_seq_info_dict = be_seq_info_dict
        self.be_seq_info_dict_map = {}
        for scene_name, be_seq_info_dict_item in self.be_seq_info_dict.items():
            self.be_seq_info_dict_map[scene_name] = {}
            for be_seq_info_item in be_seq_info_dict_item:
                self.be_seq_info_dict_map[scene_name][be_seq_info_item["sequence_name"]] = be_seq_info_item
        self.transformation_json_dict = transformation_json_dict
        self.transformation_scale_dict = transformation_scale_dict
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if debug:
           self.sequences = self.sequences[0:1] 
        
        for seq_all in self.sequences:
            if self.verbose: 
                print('seq_all', seq_all)
            
            png_path_filtered_all = []
            annotations_path_all = []
            seq_name_all = []
            seq_all_list = []

            for view_idx in range(8):
                is_mp4_sequence = Path(seq_all).parent.name == "mp4"
                seq = seq_all + "_" + str(view_idx) + "/"
                if is_mp4_sequence:
                    mp4_path = seq.rstrip("/") + ".mp4"
                    if not os.path.exists(mp4_path):
                        continue
                elif not os.path.exists(seq):
                    continue

                seq_all_list.append(seq)
                scene_name = seq.split('/')[-4]
                scene_name_no_fps = scene_name
                seq_name = seq.split('/')[-2]
                annotations_path = os.path.join(annotation_root, scene_name_no_fps, 'ground_truth', 'meta_exr_csv', seq_name + '_camera.csv')
                if is_mp4_sequence:
                    png_path_filtered = self._frame_names_from_mp4_or_csv(mp4_path, annotations_path, seq_name)
                else:
                    png_path = os.listdir(seq)
                    png_path_filtered = [x for x in png_path if x.endswith('.png')] # here we only select the 1280 * 720 png files
                png_path_filtered = sorted(png_path_filtered)
                png_path_filtered_all.append(png_path_filtered)
                seq_name_all.append(seq_name)
                annotations_path_all.append(annotations_path)
                # if scene_name_no_fps + '_' + seq_name in 

            if not annotations_path_all:
                if self.verbose:
                    print(f"rejecting seq for missing views: {seq_all}")
                continue

            if os.path.isfile(annotations_path_all[0]):
                for stride in strides:
                    full_idx = np.arange(0, len(png_path_filtered_all[0]), stride)
                    if full_idx[-1] >= len(png_path_filtered_all[0]):
                        continue
                    for seq_name_idx, seq in enumerate(seq_all_list):
                        rgb_path = []
                        rgb_filenames = []
                        depth_path = []
                        seg_path = []
                        track_npz_path = []
                        rgb_path.append([os.path.join(seq, png_path_filtered_all[seq_name_idx][idx]) for idx in full_idx])
                        rgb_filenames.append([rgb_path_i.split('/')[-1] for rgb_path_i in rgb_path[-1]])
                        depth_path.append([os.path.join(dataset_location, scene_name_no_fps, 'exr_layers', 'depth', seq_name_all[seq_name_idx], rgb_filename[:-4] + '_depth.exr') for rgb_filename in rgb_filenames[-1]])
                        seg_path_dict = {}
                        base_name = os.path.join(dataset_location, scene_name_no_fps, 'exr_layers', 'masks', seq_name_all[seq_name_idx])
                        env_shard_path = os.path.join(
                            dataset_location,
                            scene_name_no_fps,
                            'env_mask_safetensors',
                            f"{seq_name_all[seq_name_idx]}.safetensors",
                        )
                        first_env_png = os.path.join(base_name, rgb_filenames[-1][0][:-4] + '_env.png')
                        has_env_mask = os.path.exists(first_env_png) or os.path.exists(env_shard_path)
                        for rgb_filename in rgb_filenames[-1]:
                            if has_env_mask:
                                if "_env" not in seg_path_dict:
                                    seg_path_dict["_env"] = []
                                seg_path_dict["_env"].append(os.path.join(base_name, rgb_filename[:-4] + '_env.png'))
                        seg_path.append(seg_path_dict)
                        track_npz_path.append([os.path.join(dataset_location, scene_name_no_fps, 'npz', seq_name_all[seq_name_idx], rgb_filename[:-4]) for rgb_filename in rgb_filenames[-1]])
                        self.track_npz_paths.append(track_npz_path)
                        self.seg_paths.append(seg_path)
                        self.rgb_paths.append(rgb_path)
                        self.rgb_filenames.append(rgb_filenames)
                        self.depth_paths.append(depth_path)
                        seq_name = seq.split('/')[-2]
                        annotations_path = os.path.join(annotation_root, scene_name_no_fps, 'ground_truth', 'meta_exr_csv', seq_name + '_camera.csv')
                        self.annotation_paths.append(annotations_path)
                        self.full_idxs.append(full_idx)
                        self.sample_stride.append(stride)
                    if self.verbose:
                        sys.stdout.write('.')
                        sys.stdout.flush()
            elif self.verbose:
                print('rejecting seq for missing info or anno')
        
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)
        
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()
        
        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        # self.normal_paths = [self.normal_paths[i] for i in resampled_idxs]
        self.annotation_paths = [self.annotation_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)

    def _frame_names_from_mp4_or_csv(self, mp4_path, annotations_path, seq_name):
        if os.path.isfile(annotations_path):
            try:
                camera_params = pd.read_csv(annotations_path)
                if "name" in camera_params:
                    frame_names = [
                        os.path.basename(str(name))
                        for name in camera_params["name"].tolist()
                        if str(name).strip()
                    ]
                    frame_names = [name for name in frame_names if name.endswith(".png")]
                    if frame_names:
                        return sorted(frame_names)
            except Exception as exc:
                print(f"warning: failed to read camera csv for mp4 frames {annotations_path}: {exc}")

        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open mp4 for frame count: {mp4_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count <= 0:
            raise ValueError(f"Could not infer frame count from mp4 or csv: {mp4_path}")
        return [f"{seq_name}_{idx:04d}.png" for idx in range(frame_count)]

    def _get_mp4_capture(self, mp4_path):
        if mp4_path in self._mp4_capture_cache:
            cap = self._mp4_capture_cache[mp4_path]
            self._mp4_capture_cache.move_to_end(mp4_path)
            return cap
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open RGB mp4: {mp4_path}")
        self._mp4_capture_cache[mp4_path] = cap
        self._mp4_capture_cache.move_to_end(mp4_path)
        while len(self._mp4_capture_cache) > self._mp4_capture_cache_size:
            _, old_cap = self._mp4_capture_cache.popitem(last=False)
            old_cap.release()
        return cap

    def _load_rgb_image(self, impath):
        if os.path.exists(impath):
            return imread_cv2(impath)

        frame_name = os.path.basename(impath)
        frame_stem = os.path.splitext(frame_name)[0]
        try:
            frame_idx = int(frame_stem.rsplit("_", 1)[1])
        except Exception as exc:
            raise ValueError(f"Cannot parse frame index from pseudo mp4 image path: {impath}") from exc

        seq_dir = os.path.dirname(impath.rstrip("/"))
        camera_seq_name = os.path.basename(seq_dir)
        scene_dir = os.path.dirname(os.path.dirname(seq_dir))
        mp4_path = os.path.join(scene_dir, "mp4", f"{camera_seq_name}.mp4")
        cap = self._get_mp4_capture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise IOError(f"Could not decode frame {frame_idx} from RGB mp4: {mp4_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _build_cloth_npz_rel_path(self, subject, animation_id):
        if not subject:
            return None
        subject_str = str(subject)
        anim_str = None
        if animation_id and animation_id != "undefined":
            anim_str = str(animation_id)
        else:
            parts = subject_str.split("_")
            if len(parts) >= 3 and parts[-1].isdigit():
                anim_str = parts[-1]
                subject_str = "_".join(parts[:-1])
        if not anim_str:
            return None
        return f"{subject_str}/{anim_str}/{anim_str}.npz"

    def _find_cloth_member(self, tar_file, npz_rel_path, tar_name):
        candidates = [
            npz_rel_path,
            f"{tar_name}/{npz_rel_path}",
            f"{tar_name}.tar/{npz_rel_path}",
        ]
        for candidate in candidates:
            try:
                return tar_file.getmember(candidate)
            except KeyError:
                continue
        for member in tar_file.getmembers():
            if member.name.endswith(npz_rel_path):
                return member
        return None

    def _load_cloth_npz(self, subject, animation_id):
        npz_rel_path = self._build_cloth_npz_rel_path(subject, animation_id)
        if not npz_rel_path:
            return None
        tar_name = self.cloth_npz_to_tar.get(npz_rel_path)
        if not tar_name:
            return None
        tar_path = os.path.join(self.cloth_tar_root, f"{tar_name}.tar")
        if not os.path.exists(tar_path):
            return None
        cloth_npz = None
        try:
            with tarfile.open(tar_path, "r") as tar_file:
                member = self._find_cloth_member(tar_file, npz_rel_path, tar_name)
                if member is None:
                    return None
                extracted = tar_file.extractfile(member)
                if extracted is None:
                    return None
                with extracted:
                    npz_data = np.load(extracted, allow_pickle=True)
                    cloth_npz = {key: npz_data[key] for key in npz_data.files}
        except Exception as exc:
            print(f"Failed to load cloth npz {npz_rel_path}: {exc}")
            cloth_npz = None
        return cloth_npz

    def euler_to_rotation_matrix(self,yaw, pitch, roll):
        # Convert degrees to radians
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        # Compute rotation matrices for each axis
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        zxy_xyz = np.array([
            [0,0,1],
            [1,0,0],
            [0,-1,0],
        ])

        # Combine the rotation matrices
        R = R_yaw @ R_pitch @ R_roll @ zxy_xyz
        # R = R_roll @ R_pitch @ R_yaw
        return R

    def euler_to_rotation_matrix_obj(self,yaw, pitch, roll):
        # Convert degrees to radians
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        trans = np.array([
            [1,0,0],
            [0,0,1],
            [0,1,0],
        ])

        # Compute rotation matrices for each axis
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        zxy_xyz = np.array([
            [1,0,0],
            [0,0,1],
            [0,-1,0],
        ])

        # Combine the rotation matrices
        # R = R_yaw @ R_pitch @ R_roll @ zxy_xyz
        R = trans @ R_yaw @ R_pitch @ R_roll @ zxy_xyz
        # R = R_roll @ R_pitch @ R_yaw
        return R

    def euler_to_rotation_matrix_obj_v1(self,yaw, pitch, roll):
        # Convert degrees to radians
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        trans = np.array([
            [1,0,0],
            [0,0,1],
            [0,-1,0],
        ])

        # Compute rotation matrices for each axis
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        zxy_xyz = np.array([
            [1,0,0],
            [0,0,1],
            [0,1,0],
        ])

        # Combine the rotation matrices
        # R = R_yaw @ R_pitch @ R_roll @ zxy_xyz
        R = R_yaw @ R_pitch @ R_roll @ zxy_xyz
        # R = R_roll @ R_pitch @ R_yaw
        return R

    def euler_to_rotation_matrix_obj_objverse(self,yaw, pitch, roll):
        # Convert degrees to radians
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        roll = np.radians(roll)

        trans = np.array([
            [1,0,0],
            [0,0,1],
            [0,-1,0],
        ])

        # Compute rotation matrices for each axis
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)]
        ])

        zxy_xyz = np.array([
            [1,0,0],
            [0,0,1],
            [0,1,0],
        ])

        # Combine the rotation matrices
        # R = R_yaw @ R_pitch @ R_roll @ zxy_xyz
        R = R_yaw @ R_pitch @ R_roll
        # R = R_roll @ R_pitch @ R_yaw
        return R

    def _load_npz_as_dict(self, npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}

    def _slice_bedlam_clip_vertices(self, vertices_seq, start_frame, sequence_frames, label):
        start = 1 if start_frame is None else int(start_frame)
        end = start + int(sequence_frames)
        if start < 0 or end > len(vertices_seq):
            print(
                f"[vis_track] skipping {label}: requested BEDLAM frames "
                f"[{start}:{end}] from {len(vertices_seq)} available frames"
            )
            return None
        return vertices_seq[start:end]

    def _load_vertex_indices_npz(self, npz_path):
        data = self._load_npz_as_dict(npz_path)
        if "valid_flat_idx" not in data:
            if self.tracking_format == "compact_faceid":
                raise ValueError(f"Expected compact_faceid NPZ but found dense schema: {npz_path}")
            return data

        if "height" in data and "width" in data and "face_ids" in data:
            if self.tracking_format == "dense":
                raise ValueError(f"Expected dense NPZ but found compact_faceid schema: {npz_path}")
            return {
                "format": "compact_faceid",
                "height": int(np.asarray(data["height"]).reshape(-1)[0]),
                "width": int(np.asarray(data["width"]).reshape(-1)[0]),
                "valid_flat_idx": np.asarray(data["valid_flat_idx"], dtype=np.int64).reshape(-1),
                "face_ids": np.asarray(data["face_ids"], dtype=np.int64).reshape(-1),
                "npz_path": npz_path,
            }

        image_shape = np.asarray(data.get("image_shape"), dtype=np.int64).reshape(-1)
        if image_shape.size < 2:
            raise ValueError(f"sparse vertex indices missing image_shape/height/width: {npz_path}")
        height, width = int(image_shape[0]), int(image_shape[1])

        valid_flat_idx = np.asarray(data["valid_flat_idx"], dtype=np.int64).reshape(-1)
        masked_valid_flat = np.zeros(height * width, dtype=bool)
        masked_valid_flat[valid_flat_idx] = True
        masked_valid = masked_valid_flat.reshape(height, width)

        face_indices_flat = np.full(height * width, -1, dtype=np.int32)
        face_indices_flat[valid_flat_idx] = np.asarray(
            data["face_indices_valid"],
            dtype=np.int32,
        ).reshape(-1)
        face_indices = face_indices_flat.reshape(height, width, 1)

        return {
            "face_indices": face_indices,
            "weights": np.asarray(data["weights"], dtype=np.float32).reshape(1, -1, 3),
            "distances": np.asarray(data["distances"], dtype=np.float32).reshape(1, -1, 1),
            "orientation_sign": np.asarray(data["orientation_sign"], dtype=np.float32).reshape(-1, 1),
            "masked_valid": masked_valid,
        }

    def _get_tracking_safetensor(self, scene_root, seq_name):
        key = (os.path.abspath(scene_root), seq_name)
        if key in self._tracking_safetensor_cache:
            return self._tracking_safetensor_cache[key]

        shard_path = os.path.join(scene_root, "tracking_safetensors", f"{seq_name}.safetensors")
        index_path = os.path.join(scene_root, "tracking_safetensors", f"{seq_name}_index.json")
        if not (os.path.isfile(shard_path) and os.path.isfile(index_path)):
            self._tracking_safetensor_cache[key] = None
            return None

        try:
            from safetensors import safe_open
            handle = safe_open(shard_path, framework="np")
            with open(index_path, "r") as f:
                index = json.load(f)
            offsets = np.asarray(handle.get_tensor("offsets"), dtype=np.int64).reshape(-1)
            record_hw = np.asarray(handle.get_tensor("record_hw"), dtype=np.int64).reshape(-1, 2)
        except Exception as exc:
            if self.tracking_format == "safetensor":
                raise RuntimeError(f"Failed to open tracking safetensor {shard_path}: {exc}") from exc
            self._tracking_safetensor_cache[key] = None
            return None

        cached = {
            "handle": handle,
            "index": index.get("record_key_to_id", {}),
            "offsets": offsets,
            "record_hw": record_hw,
            "shard_path": shard_path,
        }
        self._tracking_safetensor_cache[key] = cached
        return cached

    def _load_vertex_indices_safetensor(self, track_dir, mask_name):
        seq_dir = os.path.dirname(track_dir)
        npz_root = os.path.dirname(seq_dir)
        scene_root = os.path.dirname(npz_root)
        seq_name = os.path.basename(seq_dir)
        frame_name = os.path.basename(track_dir)
        record_key = f"{frame_name}/{mask_name}"

        cached = self._get_tracking_safetensor(scene_root, seq_name)
        if cached is None:
            return None
        record_id = cached["index"].get(record_key)
        if record_id is None:
            return None

        offsets = cached["offsets"]
        record_hw = cached["record_hw"]
        if record_id < 0 or record_id + 1 >= offsets.shape[0] or record_id >= record_hw.shape[0]:
            raise ValueError(f"Bad record id {record_id} for {record_key} in {cached['shard_path']}")

        start = int(offsets[record_id])
        end = int(offsets[record_id + 1])
        height, width = (int(record_hw[record_id, 0]), int(record_hw[record_id, 1]))
        handle = cached["handle"]

        valid_slice = handle.get_slice("valid_flat_idx_values")
        face_slice = handle.get_slice("face_ids_values")
        valid_flat_idx = np.asarray(valid_slice[start:end], dtype=np.int64).reshape(-1)
        face_ids = np.asarray(face_slice[start:end], dtype=np.int64).reshape(-1)

        return {
            "format": "compact_faceid",
            "height": height,
            "width": width,
            "valid_flat_idx": valid_flat_idx,
            "face_ids": face_ids,
            "source": "safetensor",
            "safetensor_path": cached["shard_path"],
            "record_key": record_key,
        }

    def _load_vertex_indices_tracking(self, track_dir, mask_name):
        npz_path = os.path.join(track_dir, f"vertex_indices_{mask_name}.npz")
        if self.tracking_format == "safetensor":
            data = self._load_vertex_indices_safetensor(track_dir, mask_name)
            return data

        if os.path.exists(npz_path):
            return self._load_vertex_indices_npz(npz_path)

        if self.tracking_format in ("auto", "compact_faceid"):
            return self._load_vertex_indices_safetensor(track_dir, mask_name)
        return None

    def _get_env_mask_safetensor(self, scene_root, seq_name):
        key = (os.path.abspath(scene_root), seq_name)
        if key in self._env_mask_safetensor_cache:
            return self._env_mask_safetensor_cache[key]

        timing_t0 = time.perf_counter()
        shard_path = os.path.join(scene_root, "env_mask_safetensors", f"{seq_name}.safetensors")
        index_path = os.path.join(scene_root, "env_mask_safetensors", f"{seq_name}_index.json")
        if not os.path.exists(shard_path) or not os.path.exists(index_path):
            self._env_mask_safetensor_cache[key] = None
            return None

        try:
            from safetensors import safe_open
            handle = safe_open(shard_path, framework="np")
            with open(index_path, "r") as f:
                index = json.load(f)
            offsets = np.asarray(handle.get_tensor("offsets"), dtype=np.int64)
        except Exception as exc:
            print(f"Failed to open env mask safetensor {shard_path}: {exc}")
            self._env_mask_safetensor_cache[key] = None
            return None

        cached = {
            "handle": handle,
            "index": index,
            "offsets": offsets,
            "shard_path": shard_path,
        }
        self._env_mask_safetensor_cache[key] = cached
        if self.debug_faceid_timing:
            print(
                f"[env-mask-debug] open seq={seq_name} "
                f"time={time.perf_counter() - timing_t0:.4f}s shard={shard_path}"
            )
        return cached

    def _load_env_mask(self, mask_path):
        timing_t0 = time.perf_counter()
        if os.path.exists(mask_path):
            env_mask = imread_cv2(mask_path)
            env_valid = np.any(env_mask > 0, axis=-1) if env_mask.ndim == 3 else (env_mask > 0)
            if self.debug_faceid_timing:
                print(
                    f"[env-mask-debug] load source=png "
                    f"time={time.perf_counter() - timing_t0:.4f}s path={mask_path}"
                )
            return env_valid

        path = Path(mask_path)
        if path.name.endswith("_env.png") and len(path.parents) >= 4:
            seq_name = path.parent.name
            scene_root = path.parent.parent.parent.parent
            frame_key = path.name[:-len("_env.png")]
            cached = self._get_env_mask_safetensor(scene_root.as_posix(), seq_name)
            if cached is not None:
                frame_id = cached["index"].get("frame_to_id", {}).get(frame_key)
                if frame_id is not None:
                    offsets = cached["offsets"]
                    start = int(offsets[int(frame_id)])
                    end = int(offsets[int(frame_id) + 1])
                    encoded = np.asarray(cached["handle"].get_slice("png_bytes_values")[start:end], dtype=np.uint8)
                    env_mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                    if env_mask is None:
                        raise RuntimeError(f"Failed to decode env mask bytes for {mask_path}")
                    env_valid = env_mask > 0
                    if self.debug_faceid_timing:
                        print(
                            f"[env-mask-debug] load source=safetensor seq={seq_name} frame={frame_key} "
                            f"bytes={end - start} time={time.perf_counter() - timing_t0:.4f}s"
                        )
                    return env_valid

        raise FileNotFoundError(f"Missing env mask PNG/safetensor record: {mask_path}")

    def _world_to_camera(self, points_world, camera_pose):
        world_to_cam = np.linalg.inv(camera_pose)
        rot = world_to_cam[:3, :3]
        trans = world_to_cam[:3, 3]
        return points_world @ rot.T + trans

    def _get_views(self, index, resolution, rng):
        annotations_path = self.annotation_paths[index]
        scene_name = annotations_path.split('/')[-4]
        camera_seq_name = annotations_path.split('/')[-1].split("_camera.csv")[0]
        seq_info = self.be_seq_info_dict_map[scene_name][camera_seq_name]
        sequence_name = str(seq_info["sequence_name"])

        sequence_scale_map = self.transformation_scale_dict.get(scene_name, {})
        sequence_scales = sequence_scale_map.get(sequence_name, [])
        if not sequence_scales:
            prefix = f"{sequence_name}_"
            matched_ids = sorted([k for k in sequence_scale_map.keys() if k.startswith(prefix)])
            for matched_id in matched_ids:
                sequence_scales.extend(sequence_scale_map[matched_id])

        rgb_paths = self.rgb_paths[index][0]
        depth_paths = self.depth_paths[index][0]
        track_npz_paths = self.track_npz_paths[index][0]
        full_idx = self.full_idxs[index]
        stride = self.sample_stride[index]
        camera_params = pd.read_csv(annotations_path)
        clip_len = len(rgb_paths)
        if clip_len < 2:
            raise ValueError(f"clip too short: {clip_len}")

        resolution, num_views = resolution
        if clip_len < num_views and not self.allow_repeat:
            raise ValueError(f"clip length {clip_len} < num_views {num_views}")
        all_ids = list(range(clip_len))
        start_id = all_ids[0]
        if self.debug_faceid_timing:
            print(f"start_id: {start_id}")
            print(f"all_ids: {all_ids}")
        pos, ordered_video, _ = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_ids,
            rng,
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
        )
        selected_ids = np.array(all_ids)[pos].tolist()
        track_query_idx = 0
        if self.track_query_idx_override is not None:
            track_query_idx = int(self.track_query_idx_override)
            if track_query_idx < 0 or track_query_idx >= num_views:
                raise ValueError(
                    f"track_query_idx {track_query_idx} is out of range for num_views {num_views}"
                )
        seg_path_dict = self.seg_paths[index][0] if self.seg_paths[index] else {}
        object_mask_offset = 1 if len(seq_info.get("bodies", [])) > 0 else 0
        sequence_frames = seq_info["sequence_frames"]
        vertices_seq_d_list = {}
        faces_list = {}

        bodies = seq_info.get("bodies", [])
        if bodies:
            body_info = bodies[0]
            cloth_npz = self._load_cloth_npz(
                body_info.get("subject"),
                body_info.get("animation_id"),
            )
            if cloth_npz is not None:
                vertices_seq = self._slice_bedlam_clip_vertices(
                    cloth_npz["vertices_seq"],
                    body_info.get("start_frame"),
                    sequence_frames,
                    f"clothing {body_info.get('subject')}_{body_info.get('animation_id')}",
                )
                if vertices_seq is not None and len(vertices_seq) == sequence_frames:
                    body_pose = body_info.get("pose")
                    body_pose_matrix = self.euler_to_rotation_matrix_obj_v1(body_pose.yaw, body_pose.pitch, body_pose.roll)
                    body_translation = np.array([body_pose.x, body_pose.y, body_pose.z], dtype=np.float32) / 100
                    body_scale, global_translation = sequence_scales[0]
                    body_translation = body_translation + global_translation
                    vertices_seq_d = vertices_seq @ body_pose_matrix.T + body_translation
                    vertices_seq_d = vertices_seq_d[full_idx][selected_ids]
                    vertices_seq_d_list["_00_clothing"] = vertices_seq_d.astype(np.float32)
                    faces_list["_00_clothing"] = cloth_npz["faces"]

            npz_path = self.root_smpl_npz_path / body_info.get("subject") / (
                f"{body_info.get('subject')}_{body_info.get('animation_id')}.npz"
            )
            try:
                npz_path = npz_path.resolve()
            except FileNotFoundError:
                npz_path = None
            if npz_path is not None:
                body_npz = self._load_npz_as_dict(npz_path)
                vertices_seq = self._slice_bedlam_clip_vertices(
                    body_npz["vertices_seq"],
                    body_info.get("start_frame"),
                    sequence_frames,
                    f"body {body_info.get('subject')}_{body_info.get('animation_id')}",
                )
                if vertices_seq is not None and len(vertices_seq) == sequence_frames:
                    body_pose = body_info.get("pose")
                    body_pose_matrix = self.euler_to_rotation_matrix_obj_v1(body_pose.yaw, body_pose.pitch, body_pose.roll)
                    body_translation = np.array([body_pose.x, body_pose.y, body_pose.z], dtype=np.float32) / 100
                    body_scale, global_translation = sequence_scales[0]
                    body_translation = body_translation + global_translation
                    trans = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
                    vertices_seq_d = vertices_seq @ trans.T @ body_pose_matrix.T + body_translation
                    vertices_seq_d = vertices_seq_d[full_idx][selected_ids]
                    vertices_seq_d_list["_00_body"] = vertices_seq_d.astype(np.float32)
                    faces_list["_00_body"] = body_npz["faces"]

        objects = seq_info.get("objects", [])
        for obj_id, obj_info in enumerate(objects):
            sequence_folder = obj_info["object_group"]
            object_name = obj_info["object_id"]
            npz_path = (
                self.root_obj_npz_path
                / sequence_folder
                / object_name
                / "vertices_sequence.npz"
            )
            try:
                npz_path = npz_path.resolve()
            except FileNotFoundError:
                continue
            data = self._load_npz_as_dict(npz_path)
            vertices_seq = data["vertices_seq"]
            obj_frames = vertices_seq.shape[0]
            repeat_number = math.ceil((sequence_frames + 1) / obj_frames)
            vertices_seq = np.tile(vertices_seq, (repeat_number, 1, 1))
            vertices_seq = vertices_seq[1:1 + sequence_frames]
            vertices_seq[..., 1] = -vertices_seq[..., 1]
            body_pose = obj_info.get("pose")
            body_pose_matrix = self.euler_to_rotation_matrix_obj_objverse(body_pose.yaw, body_pose.pitch, body_pose.roll)
            body_translation = np.array([body_pose.x, body_pose.y, body_pose.z], dtype=np.float32) / 100

            body_shift_start = np.array([0, 0, 0], dtype=np.float32)
            body_shift_end = np.array([0, body_pose.shift, 0], dtype=np.float32)
            body_shift = np.linspace(body_shift_start, body_shift_end, sequence_frames)
            body_shift = np.expand_dims(body_shift, axis = 1) # add axis for vertices
            body_shift = body_shift / 100 # local shift (linearly through time)

            body_scale, global_translation = sequence_scales[obj_id]
            body_translation = body_translation + global_translation
            per_frame_motion = obj_info.get("per_frame_motion")
            if per_frame_motion is not None and len(per_frame_motion) > 0:
                motion_by_frame = {}
                for motion in per_frame_motion:
                    try:
                        motion_by_frame[int(motion.get("frame"))] = motion
                    except Exception:
                        continue
                sorted_motion_frames = sorted(motion_by_frame)

                def get_motion_for_frame(frame_i):
                    motion = motion_by_frame.get(frame_i)
                    if motion is not None:
                        return motion
                    if not sorted_motion_frames:
                        return {}
                    if frame_i <= sorted_motion_frames[0]:
                        return motion_by_frame[sorted_motion_frames[0]]
                    if frame_i >= sorted_motion_frames[-1]:
                        return motion_by_frame[sorted_motion_frames[-1]]

                    prev_frame = sorted_motion_frames[0]
                    next_frame = sorted_motion_frames[-1]
                    for candidate in sorted_motion_frames:
                        if candidate < frame_i:
                            prev_frame = candidate
                            continue
                        next_frame = candidate
                        break

                    prev_motion = motion_by_frame[prev_frame]
                    next_motion = motion_by_frame[next_frame]
                    alpha = float(frame_i - prev_frame) / float(next_frame - prev_frame)
                    interpolated = {"frame": frame_i}
                    for key, default in (
                        ("x", body_pose.x),
                        ("y", body_pose.y),
                        ("z", body_pose.z),
                        ("yaw", body_pose.yaw),
                        ("pitch", body_pose.pitch),
                        ("roll", body_pose.roll),
                        ("scale", body_scale),
                    ):
                        prev_value = float(prev_motion.get(key, default))
                        next_value = float(next_motion.get(key, default))
                        interpolated[key] = prev_value + (next_value - prev_value) * alpha
                    return interpolated

                vertices_seq_d = np.empty_like(vertices_seq, dtype=np.float32)
                for frame_i in range(sequence_frames):
                    motion = get_motion_for_frame(frame_i)
                    frame_scale = float(motion.get("scale", body_scale))
                    frame_rotation = self.euler_to_rotation_matrix_obj_objverse(
                        float(motion.get("yaw", body_pose.yaw)),
                        float(motion.get("pitch", body_pose.pitch)),
                        float(motion.get("roll", body_pose.roll)),
                    )
                    frame_translation = (
                        np.array(
                            [
                                float(motion.get("x", body_pose.x)),
                                float(motion.get("y", body_pose.y)),
                                float(motion.get("z", body_pose.z)),
                            ],
                            dtype=np.float32,
                        )
                        / 100.0
                        + global_translation
                    )
                    vertices_seq_d[frame_i] = (
                        vertices_seq[frame_i] * frame_scale
                    ) @ frame_rotation.T + frame_translation
            else:
                vertices_seq_d = (vertices_seq * body_scale + body_shift) @ body_pose_matrix.T + body_translation
            vertices_seq_d = vertices_seq_d[full_idx][selected_ids]
            mask_name = f"obj_{(obj_id + object_mask_offset):02d}"
            vertices_seq_d_list[mask_name] = vertices_seq_d.astype(np.float32)
            faces_list[mask_name] = data["faces"]

        track_npz_selected = [track_npz_paths[i] for i in selected_ids]
        per_frame_tracking = []
        for track_dir in track_npz_selected:
            frame_data = {}
            for mask_name in vertices_seq_d_list.keys():
                tracking_data = self._load_vertex_indices_tracking(track_dir, mask_name)
                if tracking_data is not None:
                    frame_data[mask_name] = tracking_data
            per_frame_tracking.append(frame_data)
        loaded_counts = [len(frame_data) for frame_data in per_frame_tracking]
        if self.tracking_format == "safetensor" and not any(loaded_counts):
            print(
                "[tracking-warning] no tracking records loaded for selected clip. "
                f"scene={scene_name} seq={camera_seq_name} "
                f"requested_masks={list(vertices_seq_d_list.keys())[:10]} "
                f"track_dirs={track_npz_selected[:3]}"
            )
        if self.debug_faceid_timing:
            format_counts = {}
            for frame_data in per_frame_tracking:
                for tracking_data in frame_data.values():
                    fmt = tracking_data.get("format", "dense")
                    format_counts[fmt] = format_counts.get(fmt, 0) + 1
            print(
                "[faceid-debug] loaded tracking records per selected frame: "
                f"min={min(loaded_counts, default=0)} "
                f"max={max(loaded_counts, default=0)} "
                f"total={sum(loaded_counts)} "
                f"formats={format_counts}"
            )
            if per_frame_tracking and not per_frame_tracking[track_query_idx]:
                print(
                    "[faceid-debug] query frame has no loaded tracking records. "
                    f"track_dir={track_npz_selected[track_query_idx]}"
                )

        frame_geometry_cache = {}
        camera_row_map = {}
        for _, row in camera_params.iterrows():
            name = str(row["name"])
            camera_row_map[os.path.basename(name)] = row

        def _load_frame_geometry(local_view_idx, clip_idx):
            if clip_idx in frame_geometry_cache:
                return frame_geometry_cache[clip_idx]

            impath = rgb_paths[clip_idx]
            impath_basename = os.path.basename(impath)
            matching_row = camera_row_map.get(impath_basename)
            if matching_row is None:
                raise ValueError(f"camera row not found for {impath_basename}")

            rgb_image = self._load_rgb_image(impath)
            depthmap = exr_to_array(Path(depth_paths[clip_idx])) / 100.0
            # Filter unreliable regions by zeroing depth before lifting to 3D:
            # - non-obj masks: keep pixels with distance < 0.01
            # - obj_* masks: keep this object only if distance.std() < 0.01, else drop all its pixels
            frame_tracking = per_frame_tracking[local_view_idx]
            invalid_depth_mask = np.zeros(depthmap.shape, dtype=bool)
            for mask_type, tracking_data in frame_tracking.items():
                masked_valid = tracking_data.get("masked_valid")
                distances = tracking_data.get("distances")
                if masked_valid is None or distances is None:
                    continue
                masked_valid = np.asarray(masked_valid, dtype=bool)
                if masked_valid.ndim == 3 and masked_valid.shape[-1] == 1:
                    masked_valid = masked_valid[..., 0]
                if masked_valid.shape != depthmap.shape:
                    continue

                valid_y, valid_x = np.where(masked_valid)
                if len(valid_y) == 0:
                    continue
                distances = np.asarray(distances, dtype=np.float32).reshape(-1)
                n = min(len(valid_y), len(distances))
                if n == 0:
                    continue
                valid_y = valid_y[:n]
                valid_x = valid_x[:n]
                distances = distances[:n]

                if mask_type.startswith("obj_"):
                    if float(np.std(distances)) >= 0.01:
                        invalid_depth_mask[valid_y, valid_x] = True
                else:
                    bad = distances >= 0.01
                    if np.any(bad):
                        invalid_depth_mask[valid_y[bad], valid_x[bad]] = True

            invalid_depth_mask |= depthmap > 300.0
            invalid_depth_mask |= ~np.isfinite(depthmap)
            depthmap[invalid_depth_mask] = 0.0 # set as invalid

            focal_length = float(matching_row['focal_length'])
            sensor_width = float(matching_row['sensor_width'])
            sensor_height = float(matching_row['sensor_height'])
            image_width = rgb_image.shape[1]
            image_height = rgb_image.shape[0]
            fx = (focal_length / sensor_width) * image_width
            fy = (focal_length / sensor_height) * image_height
            cx = image_width / 2
            cy = image_height / 2
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            yaw = float(matching_row['yaw'])
            pitch = float(matching_row['pitch'])
            roll = float(matching_row['roll'])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = self.euler_to_rotation_matrix(yaw, pitch, roll).astype(np.float32)
            camera_pose[:3, 3] = np.array(
                [matching_row['x'], matching_row['y'], matching_row['z']],
                dtype=np.float32,
            ) / 100.0

            pts3d_world, valid_mask = depthmap_to_absolute_camera_coordinates(
                depthmap=depthmap,
                camera_intrinsics=intrinsics,
                camera_pose=camera_pose,
            )
            valid_mask = valid_mask & np.isfinite(pts3d_world).all(axis=-1)

            frame_geometry_cache[clip_idx] = (
                impath,
                rgb_image,
                depthmap,
                intrinsics,
                camera_pose,
                pts3d_world,
                valid_mask,
            )
            return frame_geometry_cache[clip_idx]

        query_clip_idx = selected_ids[track_query_idx]
        _, _, _, _, _, pts3d_world_query, depth_query_valid_map = _load_frame_geometry(track_query_idx, query_clip_idx)
        static_background_valid_map = np.zeros_like(depth_query_valid_map, dtype=bool)
        for mask_type, mask_paths in seg_path_dict.items():
            if "env" not in mask_type or query_clip_idx >= len(mask_paths):
                continue
            env_valid = self._load_env_mask(mask_paths[query_clip_idx])
            if env_valid.shape == static_background_valid_map.shape:
                static_background_valid_map |= env_valid

        # Build a query-consistency invalidation map and apply it to all views:
        # if query-view wrapped track point is >10cm away from query pts3d, mark invalid everywhere.
        query_wrapped_world, query_track_valid_mask = get_wrapped_pts3d(
            target_pts3d=pts3d_world_query,
            target_tracking_npz_dict=per_frame_tracking[track_query_idx],
            vertices_seq_d_list=vertices_seq_d_list,
            faces_list=faces_list,
            source_frame_idx=track_query_idx,
            query_frame_idx=track_query_idx,
            depth_query_valid_map=depth_query_valid_map,
            static_background_valid_map=static_background_valid_map,
            debug_faceid_timing=self.debug_faceid_timing,
        )
        query_wrapped_finite = np.isfinite(query_wrapped_world).all(axis=-1)
        query_pts3d_finite = np.isfinite(pts3d_world_query).all(axis=-1)
        query_check_mask = query_track_valid_mask & query_wrapped_finite & query_pts3d_finite
        query_track_l2 = np.linalg.norm(
            query_wrapped_world.astype(np.float32) - pts3d_world_query.astype(np.float32),
            axis=-1,
        )
        query_inconsistent_mask = query_check_mask & (query_track_l2 > 0.1)
        # query_inconsistent_mask = query_check_mask

        views = []
        track_valid_counts = []
        for local_view_idx, clip_idx in enumerate(selected_ids):
            impath, rgb_image, depthmap, intrinsics, camera_pose, _, _ = _load_frame_geometry(local_view_idx, clip_idx)

            # the target is the query frame, the source_frame_idx is the target idx
            wrapped_world, track_valid_mask = get_wrapped_pts3d(
                target_pts3d=pts3d_world_query,
                target_tracking_npz_dict=per_frame_tracking[track_query_idx],
                vertices_seq_d_list=vertices_seq_d_list,
                faces_list=faces_list,
                source_frame_idx=local_view_idx,
                query_frame_idx=track_query_idx,
                depth_query_valid_map=depth_query_valid_map,
                static_background_valid_map=static_background_valid_map,
                debug_faceid_timing=self.debug_faceid_timing,
            )

            track_camera = self._world_to_camera(wrapped_world, camera_pose).astype(np.float32)
            finite_track_mask = np.isfinite(track_camera).all(axis=-1)
            track_valid_mask = track_valid_mask & finite_track_mask & (~query_inconsistent_mask)
            track_camera[~track_valid_mask] = 0.0
            track_valid_counts.append(int(np.asarray(track_valid_mask, dtype=bool).sum()))

            rgb_image, depthmap, intrinsics, track_camera, track_valid_mask = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=impath,
                track=track_camera,
                track_valid_mask=track_valid_mask.astype(np.bool_),
                allow_random_transpose=False,
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    track=track_camera.astype(np.float32),
                    track_valid_mask=track_valid_mask.astype(np.bool_),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    track_query_idx=track_query_idx,
                    dataset="syn4d_track" + ("_filterdynamic" if self.is_filter_dynamic else ""),
                    label=rgb_paths[clip_idx].split('/')[-4] + rgb_paths[clip_idx].split('/')[-2],
                    instance=f"{index}_{clip_idx}",
                    caption=self.caption,
                    fps=24 // stride,
                    stride=stride,
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        if self.debug_faceid_timing:
            print(
                "[track-debug] final valid pixels per view: "
                f"min={min(track_valid_counts, default=0)} "
                f"max={max(track_valid_counts, default=0)} "
                f"total={sum(track_valid_counts)}"
            )
        elif self.tracking_format == "safetensor" and not any(track_valid_counts):
            print(
                "[tracking-warning] safetensor tracking produced no final valid pixels. "
                f"scene={scene_name} seq={camera_seq_name}"
            )
        return views
        

def get_wrapped_pts3d(
    target_pts3d,
    target_tracking_npz_dict,
    vertices_seq_d_list,
    faces_list,
    source_frame_idx,
    query_frame_idx,
    depth_query_valid_map,
    static_background_valid_map,
    debug_faceid_timing=False,
):
    # the target is the query frame, the source_frame_idx is the target idx
    def _tracking_source_path(tracking_data):
        return tracking_data.get("npz_path") or tracking_data.get("safetensor_path")

    compact_only = (
        bool(target_tracking_npz_dict)
        and all(
            data.get("format") == "compact_faceid"
            for data in target_tracking_npz_dict.values()
        )
    )
    if compact_only:
        pts3d_target_np = np.asarray(target_pts3d, dtype=np.float32).copy()
        h, w, _ = pts3d_target_np.shape
        wrapped_valid_mask_np = np.zeros((h, w), dtype=bool)

        static_background_valid_np = np.asarray(static_background_valid_map, dtype=bool)
        if static_background_valid_np.ndim == 3 and static_background_valid_np.shape[-1] == 1:
            static_background_valid_np = static_background_valid_np[:, :, 0]
        wrapped_valid_mask_np |= static_background_valid_np

        compact_recover_count = 0
        compact_recover_points = 0
        compact_recover_valid = 0
        compact_recover_time = 0.0
        pts3d_target_flat = pts3d_target_np.reshape(-1, 3)

        for mask_type, target_tracking_npz_data in target_tracking_npz_dict.items():
            compact_recover_count += 1
            if mask_type not in faces_list or mask_type not in vertices_seq_d_list:
                continue

            timing_t0 = time.perf_counter()
            faces = np.asarray(faces_list[mask_type], dtype=np.int64)
            original_mesh_sequence = np.asarray(vertices_seq_d_list[mask_type], dtype=np.float32)
            valid_flat_idx = np.asarray(
                target_tracking_npz_data["valid_flat_idx"], dtype=np.int64
            ).reshape(-1)
            nearest_indices = np.asarray(
                target_tracking_npz_data["face_ids"], dtype=np.int64
            ).reshape(-1)
            n0 = min(valid_flat_idx.size, nearest_indices.size)
            if n0 == 0:
                continue

            valid_flat_idx = valid_flat_idx[:n0]
            nearest_indices = nearest_indices[:n0]
            compact_recover_points += int(n0)
            in_bounds = (
                (valid_flat_idx >= 0)
                & (valid_flat_idx < h * w)
                & (nearest_indices >= 0)
                & (nearest_indices < faces.shape[0])
            )
            if not np.any(in_bounds):
                continue

            valid_flat_idx = valid_flat_idx[in_bounds]
            nearest_indices = nearest_indices[in_bounds]
            valid_y = valid_flat_idx // w
            valid_x = valid_flat_idx % w

            query_points = pts3d_target_flat[valid_flat_idx]
            projected_faces = faces[nearest_indices]
            tri_query = original_mesh_sequence[query_frame_idx, projected_faces]

            a = tri_query[:, 0]
            b = tri_query[:, 1]
            c = tri_query[:, 2]
            v0 = b - a
            v1 = c - a
            v2 = query_points - a
            d00 = np.sum(v0 * v0, axis=-1)
            d01 = np.sum(v0 * v1, axis=-1)
            d11 = np.sum(v1 * v1, axis=-1)
            d20 = np.sum(v2 * v0, axis=-1)
            d21 = np.sum(v2 * v1, axis=-1)
            denom = d00 * d11 - d01 * d01
            denom_valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)
            denom_safe = np.maximum(denom, 1e-12)

            bary_1 = (d11 * d20 - d01 * d21) / denom_safe
            bary_2 = (d00 * d21 - d01 * d20) / denom_safe
            bary_0 = 1.0 - bary_1 - bary_2
            bary_w = np.stack((bary_0, bary_1, bary_2), axis=-1).astype(np.float32, copy=False)

            face_normal_query = np.cross(b - a, c - a)
            norm_query = np.linalg.norm(face_normal_query, axis=-1, keepdims=True)
            face_normal_query = face_normal_query / np.maximum(norm_query, 1e-6)
            projected_query = np.sum(tri_query * bary_w[:, :, None], axis=1)
            signed_distance = np.sum((query_points - projected_query) * face_normal_query, axis=-1)
            orientation_sign = np.sign(signed_distance).astype(np.float32, copy=False)
            orientation_sign[orientation_sign == 0] = 1.0
            distances_flat = np.abs(signed_distance).astype(np.float32, copy=False)

            finite_valid = (
                denom_valid
                & np.isfinite(query_points).all(axis=-1)
                & np.isfinite(bary_w).all(axis=-1)
                & np.isfinite(distances_flat)
                & np.isfinite(orientation_sign)
            )
            distance_threshold = 0.1
            final_valid = finite_valid & (distances_flat <= distance_threshold)
            if not np.any(final_valid):
                if debug_faceid_timing:
                    elapsed = time.perf_counter() - timing_t0
                    compact_recover_time += elapsed
                    print(
                        f"[faceid-recover-np] mask={mask_type} points={n0} valid=0 "
                        f"time={elapsed:.4f}s path={_tracking_source_path(target_tracking_npz_data)}"
                    )
                continue

            valid_y = valid_y[final_valid]
            valid_x = valid_x[final_valid]
            bary_w = bary_w[final_valid]
            orientation_sign = orientation_sign[final_valid]
            distances_flat = distances_flat[final_valid]
            projected_faces = projected_faces[final_valid]
            tri_source = original_mesh_sequence[source_frame_idx, projected_faces]
            n = bary_w.shape[0]

            face_normal_source = np.cross(
                tri_source[:, 1] - tri_source[:, 0],
                tri_source[:, 2] - tri_source[:, 0],
            )
            norm_source = np.linalg.norm(face_normal_source, axis=-1, keepdims=True)
            face_normal_source = face_normal_source / np.maximum(norm_source, 1e-6)
            projected_points_bary_source = np.sum(tri_source * bary_w[:, :, None], axis=1)
            face_normal_source = face_normal_source * orientation_sign[:, None]
            corresponding_points_source = (
                projected_points_bary_source
                + distances_flat[:, None] * face_normal_source
            )

            pts3d_target_np[valid_y, valid_x, :] = corresponding_points_source.astype(np.float32, copy=False)
            wrapped_valid_mask_np[valid_y, valid_x] = True
            compact_recover_valid += int(n)
            if debug_faceid_timing:
                elapsed = time.perf_counter() - timing_t0
                compact_recover_time += elapsed
                print(
                    f"[faceid-recover-np] mask={mask_type} points={n0} valid={n} "
                    f"time={elapsed:.4f}s per_point={elapsed / max(n0, 1):.8f}s "
                    f"dist_mean={float(distances_flat.mean()):.6f} "
                    f"dist_max={float(distances_flat.max()):.6f} "
                    f"path={_tracking_source_path(target_tracking_npz_data)}"
                )

        depth_query_valid_np = np.asarray(depth_query_valid_map, dtype=bool)
        if depth_query_valid_np.ndim == 3 and depth_query_valid_np.shape[-1] == 1:
            depth_query_valid_np = depth_query_valid_np[:, :, 0]
        wrapped_valid_mask_np &= depth_query_valid_np
        pts3d_target_np[~wrapped_valid_mask_np] = 0.0
        if debug_faceid_timing:
            print(
                f"[faceid-recover-total-np] source_frame_idx={source_frame_idx} "
                f"compact_masks={compact_recover_count} "
                f"points={compact_recover_points} valid={compact_recover_valid} "
                f"time={compact_recover_time:.4f}s"
            )
        return pts3d_target_np, wrapped_valid_mask_np

    has_compact = any(
        data.get("format") == "compact_faceid"
        for data in target_tracking_npz_dict.values()
    )
    if has_compact:
        raise ValueError(
            "Mixed compact_faceid and dense tracking schemas are not supported "
            "within one query frame."
        )

    # wrap target frame pts3d to the timestep of the source frame
    pts3d_target = torch.as_tensor(target_pts3d, dtype=torch.float32).clone()
    h, w, _ = pts3d_target.shape
    wrapped_valid_mask = torch.zeros((h, w), dtype=torch.bool)
    static_background_valid_map = torch.as_tensor(static_background_valid_map, dtype=torch.bool)
    if static_background_valid_map.ndim == 3 and static_background_valid_map.shape[-1] == 1:
        static_background_valid_map = static_background_valid_map[:, :, 0]
    wrapped_valid_mask = wrapped_valid_mask | static_background_valid_map

    for mask_type in target_tracking_npz_dict:
        target_tracking_npz_data = target_tracking_npz_dict[mask_type]
        masked_valid = torch.as_tensor(target_tracking_npz_data["masked_valid"], dtype=torch.bool)
        if masked_valid.ndim == 3 and masked_valid.shape[-1] == 1:
            masked_valid = masked_valid[:, :, 0]
        if mask_type not in faces_list or mask_type not in vertices_seq_d_list:
            continue
        weights = torch.as_tensor(target_tracking_npz_data["weights"], dtype=torch.float32)
        orientation_sign = torch.as_tensor(target_tracking_npz_data["orientation_sign"], dtype=torch.float32)
        distances = torch.as_tensor(target_tracking_npz_data["distances"], dtype=torch.float32)
        face_index_map_np = target_tracking_npz_data.get("face_index_map", target_tracking_npz_data.get("face_indices"))
        face_index_map = torch.as_tensor(face_index_map_np, dtype=torch.long)
        if face_index_map.ndim == 3 and face_index_map.shape[-1] == 1:
            face_index_map = face_index_map[:, :, 0]
        faces = torch.as_tensor(faces_list[mask_type], dtype=torch.long)
        original_mesh_sequence = torch.as_tensor(vertices_seq_d_list[mask_type], dtype=torch.float32)

        nearest_indices = face_index_map[masked_valid]
        n = nearest_indices.numel()
        if n == 0:
            continue

        distance_threshold = 0.1  # 10cm threshold, adjust as needed
        distances_flat = distances.reshape(-1)[:n]  # [N]
        valid_distance_mask = distances_flat <= distance_threshold

        # Invalidate face indices where the distance is too large.
        face_index_map = face_index_map.clone()
        valid_y, valid_x = torch.where(masked_valid)
        if valid_y.numel() == valid_distance_mask.numel():
            invalid_mask = ~valid_distance_mask
            if invalid_mask.any():
                invalid_y = valid_y[invalid_mask]
                invalid_x = valid_x[invalid_mask]
                face_index_map[invalid_y, invalid_x] = -1

        projected_faces = faces[nearest_indices.flatten()]  # [N, 3]
        projected_faces_point = original_mesh_sequence[:, projected_faces.flatten()]
        projected_faces_point = projected_faces_point.reshape(-1, n, 3, 3)
        # cross product normal from projected faces point
        face_normal_all = torch.cross(
            projected_faces_point[:, :, 1] - projected_faces_point[:, :, 0],
            projected_faces_point[:, :, 2] - projected_faces_point[:, :, 0],
            dim=-1,
        )  # [T, N, 3]
        face_normal_all = torch.nn.functional.normalize(face_normal_all, eps=1e-6, dim=-1)

        # Barycentric projected points on each frame: [T, N, 3]
        bary_w = weights[0].reshape(-1, 3)[:n]  # [N, 3]
        projected_points_bary_all = torch.sum(
            projected_faces_point * bary_w.unsqueeze(0).unsqueeze(-1),
            dim=2,
        )
        orientation_sign_flat = orientation_sign.reshape(-1)[:n]
        face_normal_all = face_normal_all * orientation_sign_flat.view(1, n, 1)

        corresponding_points_all = projected_points_bary_all + torch.abs(distances_flat).view(1, n, 1) * face_normal_all

        vertex_track_map = torch.zeros_like(pts3d_target)
        vertex_track_map[valid_y, valid_x, :] = corresponding_points_all[source_frame_idx]

        valid_mask_with_vertex = (face_index_map >= 0) & masked_valid
        if valid_mask_with_vertex.any():
            valid_y2, valid_x2 = torch.where(valid_mask_with_vertex)
            pts3d_target[valid_y2, valid_x2, :] = vertex_track_map[valid_y2, valid_x2, :]
            wrapped_valid_mask[valid_y2, valid_x2] = True

    # track mask must ve valid in depth map
    depth_query_valid_map = torch.as_tensor(depth_query_valid_map, dtype=torch.bool)
    if depth_query_valid_map.ndim == 3 and depth_query_valid_map.shape[-1] == 1:
        depth_query_valid_map = depth_query_valid_map[:, :, 0]

    wrapped_valid_mask = wrapped_valid_mask & depth_query_valid_map

    pts3d_target[~wrapped_valid_mask] = 0.0
    return pts3d_target.cpu().numpy(), wrapped_valid_mask.cpu().numpy()
