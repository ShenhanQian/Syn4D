# Syn4D Multi-View Track Visualizer

Interactive Viser visualizer for Syn4D tracking, geometry, and RGB sequences.

## Files

- `run.py`: command-line entry point.
- `syn4d_track.py`: Syn4D dataset reader used by the visualizer.
- `base_dataset.py`: shared crop, resize, camera, and view formatting logic.
- `utils.py`: image, camera, geometry, and tensor helpers.
- `viser_visualizer_track.py`: Viser server and 3D visualization UI.

## Requirements

Install the requirements for this visualizer:

```bash
pip install -r requirements.txt
```

The local `requirements.txt` lists only the packages used by this folder:

```text
torch
torchvision
numpy
opencv-python
Pillow
tqdm
matplotlib
scipy
imageio
viser
pandas
pause
safetensors
OpenEXR
Imath
```

## Basic Usage

Run from the downloaded dataset repository root:

```bash
python code/visualizer/run.py \
  --dataset-root /scratch/shared/beegfs/kelvin/Syn4D/subsets/sim_fixed_re_safetensor \
  --metadata-root /scratch/shared/beegfs/zeren/Syn4D/metadata \
  --scene-name downtown_sim \
  --tracking-format safetensor \
  --rgb-source auto \
  --resolution "(512,512)" \
  --num-frames 10 \
  --host 0.0.0.0 \
  --port 8020
```

Open the printed Viser URL in a browser. If running on a remote machine, forward the selected port or connect through the machine address exposed by your cluster setup.

## Common Options

- `--dataset-root`: Syn4D dataset root.
- `--metadata-root`: primary metadata root.
- `--fallback-metadata-root`: optional fallback metadata root; pass `""` to disable it.
- `--scene-name`: scene filter. Repeat the flag or pass comma-separated scene names.
- `--select-idx-view0`: dataset index for the primary sequence.
- `--select-idx-view1`: optional dataset index for a secondary sequence.
- `--track-query-idx`: override the query/reference frame.
- `--stride`: frame stride. Use `5` for stride-5 tracks and `1` for stride-1 tracks.
- `--num-frames`: number of sampled frames to visualize.
- `--resolution`: square size like `512`, tuple string like `"(512,512)"`, or landscape form like `512x384`.
- `--rgb-source`: `auto`, `png`, or `mp4`.
- `--tracking-format`: `auto`, `dense`, `compact_faceid`, or `safetensor`.
- `--debug-faceid-timing`: print loading and face-id recovery timing diagnostics.

## Safetensor / MP4 Dataset Notes

For the packed datasets, prefer:

```bash
--tracking-format safetensor --rgb-source auto
```

`auto` uses PNG frames when present and falls back to MP4 when PNG frames are omitted. The safetensor tracking reader expects the packed per-sequence tracking shards and their JSON indices to be present under the dataset root.

## Example: Stride-1 Syn4D v1

```bash
python code/visualizer/run.py \
  --dataset-root /scratch/shared/beegfs/kelvin/Syn4D/subsets/syn4d_v1_safetensor_stride_1 \
  --metadata-root /scratch/shared/beegfs/zeren/Syn4D/metadata \
  --scene-name downtown_bald \
  --tracking-format safetensor \
  --rgb-source auto \
  --stride 1 \
  --resolution "(512,512)" \
  --num-frames 10 \
  --host 0.0.0.0 \
  --port 8020
```

## Troubleshooting

- If Bash errors on `(`, quote the resolution: `--resolution "(512,512)"`.
- If no tracks appear, rerun with `--debug-faceid-timing` and check whether tracking records are loaded for the selected sequence.
- If the selected scene has many clips, use `--select-idx-view0` to move through candidates.
- If port `8020` is in use, pass another `--port`.
