---
license: cc-by-4.0
arxiv: 2605.05207
pretty_name: "Syn4D: A Multiview Synthetic 4D Dataset"
viewer: false
---

# Syn4D: A Multiview Synthetic 4D Dataset

Syn4D is a synthetic 4D dataset with multi-view RGB videos, depth, masks, tracking geometry, and supporting object mesh metadata.

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://jzr99.github.io/Syn4D/)
[![Paper](https://img.shields.io/badge/arXiv-2605.05207-b31b1b)](https://arxiv.org/abs/2605.05207)
[![Gallery](https://img.shields.io/badge/Space-Syn4D%20Gallery-blue)](https://huggingface.co/spaces/Syn4D/Syn4D_Gallery)

![Syn4D teaser](resources/syn4d_teaser_v2.png)

## Layout

```text
data/
  syn4d_v1_stride_1/       # Syn4D V1, every frame can be a tracking reference frame
  syn4d_v1_stride_5/       # Syn4D V1, every 5th frame can be a tracking reference frame
  syn4d_sim/               # Kubric-style simulation subset
  metadata/new_weight_bone/

code/visualizer/
  README.md
  requirements.txt
  run.py

code/bedlam2/
  README.md
  smplx_anim_to_alembic_batch.py
  smplx_anim_to_objs_batch.py

LICENSE
README.md
THIRD_PARTY_NOTICES.md
LICENSES/
resources/
```

Each Syn4D V1 split includes `sequence_to_asset_mapping.csv`, which maps dataset sequences to their underlying assets. The same mapping files are also referenced in `THIRD_PARTY_NOTICES.md` together with third-party asset notices and license references.

## Using The Dataset

1. Clone or download this Hugging Face dataset repository.

```bash
git lfs install
git clone https://huggingface.co/datasets/Syn4D/Syn4D
cd Syn4D
```

2. Choose a dataset subset and extract its scene archives.

The main subsets are:

- `data/syn4d_v1_stride_1/`: Syn4D V1 with every frame available as a possible tracking reference frame.
- `data/syn4d_v1_stride_5/`: Syn4D V1 with every 5th frame available as a possible tracking reference frame.
- `data/syn4d_sim/`: additional Kubric-style simulation subset packaged in the same safetensor/MP4 layout.

To extract all scene archives directly under the same subset folder:

```bash
SUBSET=data/syn4d_v1_stride_5

for archive in "$SUBSET"/*.tar.zst; do
  tar -I zstd -xf "$archive" -C "$SUBSET"
done
```

Equivalent decompression form if your `tar` does not support `-I zstd`:

```bash
tar --use-compress-program=unzstd -xf <archive>.tar.zst -C <output_dir>
```

For full geometry/body reconstruction workflows, use the metadata under `data/metadata/`. Object metadata is provided under `data/metadata/new_weight_bone/`. Processed BEDLAM2 metadata is not redistributed due to BEDLAM2 license restrictions; if needed, follow `code/bedlam2/README.md` to download BEDLAM2 from the official source and convert SMPL-X motion files.

BEDLAM2 conversion is CPU-heavy, so running it on machines with fast CPUs is recommended.

3. Run the visualizer or integrate the dataset.

For visualization instructions, see `code/visualizer/README.md`.
