import numpy as np
import PIL
import torch
import itertools
import random

from utils import depthmap_to_absolute_camera_coordinates, geotrf, inv, ImgNorm, crop_image_depthmap, rescale_image_depthmap, camera_matrix_of_crop, bbox_from_intrinsics_in_out


class BaseStereoViewDataset():
    """Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=ImgNorm,
        aug_crop=False,
        seed=None,
        num_track_points=768,
        is_filter_dynamic=False,
        filter_dynamic_threshold=80,
    ):
        self.num_views = 48
        self.split = split
        self._set_resolutions(resolution)

        if isinstance(transform, str):
            transform = eval(transform)
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed
        self.num_track_points = num_track_points
        self.is_filter_dynamic = is_filter_dynamic
        self.filter_dynamic_threshold = filter_dynamic_threshold

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def _synced_transform(self, imgs):
        """Apply the same random transform to all images by stacking them."""
        w, h = imgs[0].size
        combined_img = PIL.Image.new(imgs[0].mode, (w * len(imgs), h))
        for i, img in enumerate(imgs):
            combined_img.paste(img, (i * w, 0))
        
        combined_tensor = self.transform(combined_img)
        # split images
        return torch.split(combined_tensor, w, dim=-1)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)

        # dynmic sample num_views from max_num_views
        resolution, num_views = resolution
        if self._rng.random() < 0.0:
            # Prefer larger view counts with linearly increasing probability
            max_num_views = num_views
            candidates = np.arange(2, max_num_views + 1)
            probs = candidates / candidates.sum()
            num_views = self._rng.choice(candidates, p=probs)
        resolution = (resolution, num_views)

        try:
            views = self._get_views(idx, resolution, self._rng)
        except Exception as e:
            print(f"get_views error: {e}")
            return self.__getitem__(((idx + 1) % self.__len__(), ar_idx))

        has_valid_pts3d = False
        has_valid_track = False

        # modified by yihang: with 30% probability, apply transform to all views together
        perform_synced_transform = False
        if len(views) > 0 and random.random() < 0.3:
            perform_synced_transform = True
            transformed_imgs = self._synced_transform([view["img"] for view in views])

        # check data-types
        track_query_idx = 0 if "track_query_idx" not in views[0] else views[0]["track_query_idx"]

        for v, view in enumerate(views):
            assert (
                "pts3d" not in view
            ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view["idx"] = (idx, ar_idx, v)

            # encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            
            if perform_synced_transform:
                view["img"] = transformed_imgs[v]
            else:
                view["img"] = self.transform(view["img"])

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(
                    view["camera_pose"]
                ).all(), f"NaN in camera pose for view {view_name(view)}"
            assert "pts3d" not in view
            assert "valid_mask" not in view
            if not np.isfinite(
                view["depthmap"]
            ).all():
                print(f"NaN in depthmap for view {view_name(view)}")
                return self.__getitem__(((idx + 1) % self.__len__(), ar_idx))

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            if np.any(view["valid_mask"]):
                has_valid_pts3d = True

        for v, view in enumerate(views):
            if "track" in view:
                view["track_valid_mask"] = view["track_valid_mask"] & views[track_query_idx]["valid_mask"]
                if np.any(view["track_valid_mask"]):
                    has_valid_track = True

                # we use track in the global coordinate system
                camera_pose = view["camera_pose"]
                R_cam2world = camera_pose[:3, :3]
                t_cam2world = camera_pose[:3, 3]

                # Express in absolute coordinates (invalid depth values)
                view["track"] = (
                    np.einsum("ik, vuk -> vui", R_cam2world, view["track"]) + t_cam2world[None, None, :]
                )

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

        if not has_valid_pts3d or ("track" in view and not has_valid_track):
            print("no valid pts3d or track", has_valid_pts3d, has_valid_track)
            return self.__getitem__(((idx + 1) % self.__len__(), ar_idx))

        anchor_camera_inv = inv(views[0]["camera_pose"])
        for view in views:
            gt_extrinsic = view["camera_pose"]
            # inv is because dust3r frammework, dataset gt extrinsic is cam2world, but vggt predicts world2cam
            view['w2c_aligned'] = inv(anchor_camera_inv @ gt_extrinsic)
            view['c2w_aligned'] = anchor_camera_inv @ gt_extrinsic
            view['pts3d_global_aligned'] = geotrf(anchor_camera_inv, view["pts3d"])
            view['track_global_aligned'] = geotrf(anchor_camera_inv, view["track"])

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

            if "is_2d_track" not in view:
                view["is_2d_track"] = False

        if self.is_filter_dynamic:
            ref_track = views[track_query_idx]["track"].copy()
            for i, view in enumerate(views):
                if i == track_query_idx:
                    valid_mask = view["track_valid_mask"]
                    valid_mask_flat = valid_mask.flatten()
                    num_valid = np.sum(valid_mask_flat)
                    num_keep = int(num_valid * (1 - self.filter_dynamic_threshold / 100.0))
                    keep_indices = np.random.choice(np.where(valid_mask_flat)[0], num_keep, replace=False)
                    new_valid_mask_flat = np.zeros_like(valid_mask_flat, dtype=valid_mask.dtype)
                    new_valid_mask_flat[keep_indices] = True
                    view["track_valid_mask"] = new_valid_mask_flat.reshape(valid_mask.shape)
                else:
                    diff = np.linalg.norm(view["track"] - ref_track, axis=-1)
                    valid_mask = view["track_valid_mask"]
                    valid_diffs = diff[valid_mask]
                    threshold = np.percentile(valid_diffs, self.filter_dynamic_threshold)
                    view["track_valid_mask"] = valid_mask & (diff >= threshold)

                # TODO: remove this, just for visualization debug
                # view["track"][~view["track_valid_mask"]] = 0

        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                (width, height), num_views = resolution
            assert isinstance(
                width, int
            ), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(
                height, int
            ), f"Bad type for {height=} {type(height)=}, should be int"
            assert isinstance(
                num_views, int
            ), f"Bad type for {num_views=} {type(num_views)=}, should be int"
            assert width >= height
            self._resolutions.append(((width, height), num_views))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None, track=None, track_valid_mask=None, allow_random_transpose=True, track2d_01=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # in current implementation, track2d_01 should be used with track for the following crop/resize
        if track2d_01 is not None:
            assert track is not None

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)

        if track is not None:
            image, depthmap, intrinsics, track, track_valid_mask, track2d_01 = crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox, track=track, track_valid_mask=track_valid_mask, track2d_01=track2d_01
            )
        else:
            image, depthmap, intrinsics = crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if allow_random_transpose and rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        
        if track is not None:
            track_dim = track.shape[-1]
            if track2d_01 is not None:
                nearest_value_to_resize = np.concatenate([track, track2d_01], axis=-1)
                image, depthmap, intrinsics, nearest_value_to_resize, track_valid_mask = rescale_image_depthmap(
                    image, depthmap, intrinsics, target_resolution, track=nearest_value_to_resize, track_valid_mask=track_valid_mask
                )
                track, track2d_01 = nearest_value_to_resize[..., :track_dim], nearest_value_to_resize[..., track_dim:]
            else:
                image, depthmap, intrinsics, track, track_valid_mask = rescale_image_depthmap(
                    image, depthmap, intrinsics, target_resolution, track=track, track_valid_mask=track_valid_mask
                )
        else:
            image, depthmap, intrinsics = rescale_image_depthmap(
                image, depthmap, intrinsics, target_resolution
            )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        if track is not None:
            image, depthmap, intrinsics2, track, track_valid_mask, track2d_01 = crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox, track=track, track_valid_mask=track_valid_mask, track2d_01=track2d_01
            )
        else:
            image, depthmap, intrinsics2 = crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )

        if track is not None:
            if track2d_01 is not None:
                return image, depthmap, intrinsics2, track, track_valid_mask, track2d_01
            else:
                return image, depthmap, intrinsics2, track, track_valid_mask
        else:
            return image, depthmap, intrinsics2


class BaseStereoDynamicViewDataset(BaseStereoViewDataset):
    def __init__(
        self,
        *,
        split=None,
        resolution=None,
        transform=ImgNorm,
        aug_crop=False,
        seed=None,
        allow_repeat=True, # modified by yihang
        num_track_points=768,
        is_filter_dynamic=False,
        filter_dynamic_threshold=80,
    ):
        super().__init__(
            split=split,
            resolution=resolution,
            transform=transform,
            aug_crop=aug_crop,
            seed=seed,
            num_track_points=num_track_points,
            is_filter_dynamic=is_filter_dynamic,
            filter_dynamic_threshold=filter_dynamic_threshold,
        )
        self.allow_repeat = allow_repeat

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True, list(range(num_views))
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        pos_ref_zero = [int(x) - pos[0] for x in pos]
        return pos, is_video, pos_ref_zero


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None

def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"

def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        if "track" in view:
            assert view["track"].shape == (height, width, 3)
            view["track"] = view["track"].swapaxes(0, 1)
            assert view["track_valid_mask"].shape == (height, width)
            view["track_valid_mask"] = view["track_valid_mask"].swapaxes(0, 1)

        if "pts3d_global_aligned" in view:
            assert view["pts3d_global_aligned"].shape == (height, width, 3)
            view["pts3d_global_aligned"] = view["pts3d_global_aligned"].swapaxes(0, 1)
            assert view["track_global_aligned"].shape == (height, width, 3)
            view["track_global_aligned"] = view["track_global_aligned"].swapaxes(0, 1)

        if "track2d" in view:
            assert view["track2d"].shape == (height, width, 2)
            view["track2d"] = view["track2d"].swapaxes(0, 1)
            assert view["track2d_valid_mask"].shape == (height, width)
            view["track2d_valid_mask"] = view["track2d_valid_mask"].swapaxes(0, 1)
            assert view["track2d_visibility"].shape == (height, width)
            view["track2d_visibility"] = view["track2d_visibility"].swapaxes(0, 1)

        if "instance_mask" in view:
            assert view["instance_mask"].shape == (height, width)
            view["instance_mask"] = view["instance_mask"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]

        if "tap_track2d" in view:
            assert view["tap_track2d"].shape[-1] == 2
            view["tap_track2d"] = view["tap_track2d"][..., [1, 0]]
            view["tap_queries"] = view["tap_queries"][..., [0, 2, 1]]

        if "tap_track3d" in view:
            assert view["tap_track2d"].shape[-1] == 3
            view["tap_track3d"] = view["tap_track3d"][..., [1, 0, 2]]

def flip_views_horizontally(views):
    for view in views:
        width, height = view["img"].size

        view["img"] = ImageOps.mirror(view["img"])

        if "tap_track2d" in view:
            assert view["tap_track2d"].shape[-1] == 2
            tap_xy = view["tap_track2d"].copy()
            tap_xy[..., 0] = width - tap_xy[..., 0]
            view["tap_track2d"] = tap_xy
        if "tap_queries" in view:
            # [t, x, y]
            tq = view["tap_queries"].copy()
            tq[..., 1] = width - tq[..., 1]
            view["tap_queries"] = tq