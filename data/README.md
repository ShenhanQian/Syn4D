# Syn4D Data Subsets

This folder contains the released Syn4D dataset subsets and shared metadata.

| Folder | Contents | Intended Use |
| --- | --- | --- |
| `syn4d_v1_stride_1/` | Syn4D V1 with every frame available as a possible tracking reference frame. | Use when dense temporal reference-frame coverage is needed. |
| `syn4d_v1_stride_5/` | Syn4D V1 with every 5th frame available as a possible tracking reference frame. | Use when stride-5 reference-frame coverage is sufficient. |
| `syn4d_sim/` | Additional Kubric-style simulation subset packaged in the same safetensor/MP4 layout. | Use as an extra synthetic subset alongside Syn4D V1. |
| `metadata/` | Shared object mesh, rig, and auxiliary metadata used by the loaders and visualizer. | Download with the subsets when running code that reconstructs geometry or visualizes tracks. |

## Files and Layout

The dataset subsets are distributed as per-scene archives plus selected convenience files:

| Path / File | Meaning |
| --- | --- |
| `{dataset_subset}/*.tar.zst` | Per-scene compressed archives under `syn4d_v1_stride_1/`, `syn4d_v1_stride_5/`, or `syn4d_sim/`. Each archive contains the packaged scene data, including camera metadata, depth/mask layers, and safetensor tracking shards. |
| `{dataset_subset}/mp4/{scene}/*.mp4` | Per-sequence RGB videos for quick preview and video-based loading. |
| `{dataset_subset}/sequence_to_asset_mapping.csv` | Mapping from sequence names to the underlying object/body assets used in each Syn4D V1 sequence. |
| `metadata/new_weight_bone/` | Object mesh, rig, and skinning metadata used by the loaders/visualizer for geometry reconstruction. |

Inside extracted scene archives, the main components are:

| Component | Meaning |
| --- | --- |
| `ground_truth/meta_exr_csv/` | Camera and frame metadata. |
| `tracking_safetensors/` | Compact point-to-surface tracking records stored as safetensor shards with JSON indices. |
| `env_mask_safetensors/` | Packed environment masks, when available. |
| `exr_layers/` | Rendered depth and mask layers retained for geometry and visibility processing. |
| `mp4/` | RGB video files for the scene, when included in the extracted layout. |

The V1 subsets include `sequence_to_asset_mapping.csv`, which maps sequences to their underlying third-party assets. License details for those assets are summarized in the repository-level `THIRD_PARTY_NOTICES.md`.
