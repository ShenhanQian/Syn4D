# BEDLAM2 SMPL-X to Vertex NPZ Conversion

This folder contains the BEDLAM2 conversion scripts used to generate per-animation
vertex-cache `.npz` files from BEDLAM2 source SMPL-X motion `.npz` files.

The conversion is intentionally split into two stages:

```text
BEDLAM2 motion .npz -> Alembic .abc -> vertex-cache .npz
```

The final `.npz` files contain:

- `vertices_seq`: `(T, 10475, 3)`, `float32`
- `faces`: `(20908, 3)`, `int32`
- `start_frame`
- `end_frame`
- `target_fps`
- `source_fps`

## Required Files

Keep these files together in the same folder:

- `smplx_anim_to_alembic.py`
- `smplx_anim_to_alembic_batch.py`
- `smplx_anim_to_objs.py`
- `smplx_anim_to_objs_batch.py`
- `supervised_blender_batch.py`
- `requirements.txt`

## System Requirements

- Python 3.10+
- Blender 4.5+ recommended
- SMPL-X Blender add-on installed in Blender
- Enough disk space for the intermediate `.abc` files

The batch drivers use only the Python standard library. The Blender worker that
writes `.npz` files uses NumPy, which is normally included with Blender Python.
If NumPy is missing from Blender Python, install it into Blender Python:

```powershell
"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe" -m pip install -r requirements.txt
```

Adjust that Python path for your Blender installation.

## Install Blender

On Linux, install Blender under a local app folder:

```bash
mkdir -p ~/apps
cd ~/apps
tar -xf blender-4.5.0-linux-x64.tar.xz
export BLENDER="$(pwd)/blender-4.5.0-linux-x64/blender"
"$BLENDER" --background --version
```

On Windows, the batch scripts default to:

```text
C:\Program Files\Blender Foundation\Blender 5.0\blender.exe
```

If Blender is elsewhere, pass it with `--blender`.

## Install the SMPL-X Blender Add-on

`pip install smplx` is not enough for this conversion. The first stage uses the
official SMPL-X Blender add-on through Blender operators.

Download the official SMPL-X Blender add-on zip after logging into the SMPL-X
site and accepting the license. Place it under the same app folder, for example:

```text
~/apps/smplx_blender_addon-1.0.3-20260511.zip
```

The current SMPL-X package is a Blender extension, not a legacy add-on. Install
and enable it with Blender's extension command:

```bash
"$BLENDER" --command extension install-file \
  --repo user_default \
  --enable \
  "$HOME/apps/smplx_blender_addon-1.0.3-20260511.zip"
```

Alternatively, in the Blender UI, use **Edit > Preferences > Get Extensions >
Install from Disk**, select the downloaded zip, and ensure **SMPL-X for Blender**
is enabled under **Preferences > Add-ons**.

Do not use `bpy.ops.preferences.addon_install` for this package. That legacy
installer may extract the files but does not register the extension, resulting
in `has smplx_tool: False` and an `add-on missing 'bl_info'` warning. If the
package was already installed that way, move the incorrectly installed copy out
of the legacy add-on directory before running the extension command:

```bash
mv "$HOME/.config/blender/4.5/scripts/addons/smplx_blender_addon" \
  "$HOME/apps/smplx_blender_addon-legacy-backup"
```

Verify that the add-on is available:

```bash
"$BLENDER" --background --python-expr "
import bpy
print('has smplx_tool:', hasattr(bpy.context.window_manager, 'smplx_tool'))
"
```

Expected output:

```text
has smplx_tool: True
```

## Download Needed BEDLAM2 Files

Use the same `METADATA_ROOT` where you store the metadata downloaded from the
Syn4D Hugging Face dataset. This keeps the BEDLAM2 files, converted vertex-cache
files, and other provided Syn4D metadata under one root. In the examples below,
assume you are already inside that metadata root:

```bash
cd METADATA_ROOT
export METADATA_ROOT="$(pwd)"
```

Download these files from the BEDLAM2 download page after logging in and
accepting the license: https://bedlam2.is.tuebingen.mpg.de/download.php

- `b2_motions_npz_training.tar`: under **Body and Motion Ground Truth**
- `b2_assetdata_download.zip`: under **Simulated clothing cache: NPZ format**

Place both files directly under `METADATA_ROOT`.

Extract them into same-name folders:

```bash
mkdir -p b2_motions_npz_training
tar -xf b2_motions_npz_training.tar -C b2_motions_npz_training

mkdir -p b2_assetdata_download
unzip b2_assetdata_download.zip -d b2_assetdata_download
```

After extraction, the motion files should look like:

```text
METADATA_ROOT/b2_motions_npz_training/motions_npz_training/*.npz
```

The BEDLAM2 asset download helper should be here:

```text
METADATA_ROOT/b2_assetdata_download/download_script.sh
```

To download the clothing NPZ assets, run:

```bash
cd b2_assetdata_download
bash download_script.sh clothing_npz
```

The script prompts for the BEDLAM2 website email and password. Password input is
silent. The clothing download should produce:

```text
METADATA_ROOT/b2_assetdata_download/clothing/npz/b2_clothing_npz_XXX.tar
```

To extract the clothing NPZ assets, run:

```bash
cd b2_assetdata_download/clothing/npz
for archive in *.tar; do
  # Extract the folder name by removing the '.tar' extension
  dir="${archive%.tar}"
  
  # Create the folder and extract into it
  mkdir -p "$dir" && tar -xvf "$archive" -C "$dir"
done
```

Important distinction:

- `b2_motions_npz_training/motions_npz_training/*.npz` is required for the
  SMPL-X motion-to-vertex-cache conversion.
- `b2_assetdata_download/clothing/npz/` is part of the broader BEDLAM2/Syn4D
  metadata setup, and should live under the same metadata root as the other
  provided metadata.

## Expected Metadata Layout

`METADATA_ROOT` should ideally be the same metadata root where you placed the
metadata downloaded from the Syn4D Hugging Face dataset. A completed setup
should look like:

```text
METADATA_ROOT/
  animations/
    training/
      *.npz

  b2_assetdata_download/
    download_script.sh
    clothing/
      npz/
        archive_map.json
        checksum.xxh128
        b2_clothing_npz_000.tar
        ...
        b2_clothing_npz_275.tar

  bedlam2_smpl_npz/
    {subject}/
      {motion}.npz

  new_weight_bone/
    ...
```

The conversion scripts produce:

```text
METADATA_ROOT/bedlam2_smpl_npz/{subject}/{motion}.npz
```

The standard raw BEDLAM2 motion input layout is:

```text
METADATA_ROOT/animations/training/*.npz
```

If you extracted `b2_motions_npz_training.tar` into:

```text
METADATA_ROOT/b2_motions_npz_training/motions_npz_training/*.npz
```

you can either pass that folder directly to the converter, or move/symlink it to
`METADATA_ROOT/animations/training` for compatibility with the standard metadata
layout.

## Smoke Test: One Motion File

Open a terminal in the folder that contains these scripts before running the
commands below.

Stage 1 converts one BEDLAM2 motion `.npz` to Alembic:

```bash
"$BLENDER" --background \
  --python smplx_anim_to_alembic.py -- \
  --input "$METADATA_ROOT"/b2_motions_npz_training/motions_npz_training/it_4001_XL_2400.npz \
  --output "$METADATA_ROOT"/linux_smoke_abc/it_4001_XL/it_4001_XL_2400.abc
```

Stage 2 converts that Alembic file to the final vertex-cache `.npz`:

```bash
"$BLENDER" --background \
  --python smplx_anim_to_objs.py -- \
  --input "$METADATA_ROOT"/linux_smoke_abc/it_4001_XL/it_4001_XL_2400.abc \
  --output "$METADATA_ROOT"/linux_smoke_vertices_npz/it_4001_XL/it_4001_XL_2400.npz
```

You can also smoke-test stage 2 through the batch wrapper:

```bash
python3 smplx_anim_to_objs_batch.py \
  "$METADATA_ROOT"/linux_smoke_abc \
  "$METADATA_ROOT"/linux_smoke_vertices_npz \
  1 \
  --blender "$BLENDER" \
  --timeout-seconds 600 \
  --retries 1
```

## Full Batch Conversion

Stage 1 converts all raw BEDLAM2 motion `.npz` files to intermediate `.abc`
files:

```bash
python3 smplx_anim_to_alembic_batch.py \
  "$METADATA_ROOT"/b2_motions_npz_training/motions_npz_training \
  "$METADATA_ROOT"/bedlam2_smpl_abc \
  4 \
  --blender "$BLENDER" \
  --timeout-seconds 600 \
  --retries 1 \
  --skip-existing
```

Stage 2 converts all `.abc` files to final vertex-cache `.npz` files:

```bash
python3 smplx_anim_to_objs_batch.py \
  "$METADATA_ROOT"/bedlam2_smpl_abc \
  "$METADATA_ROOT"/bedlam2_smpl_npz \
  4 \
  --blender "$BLENDER" \
  --timeout-seconds 600 \
  --retries 1 \
  --skip-existing
```

On Windows, the same commands work with Windows paths and the default Blender
path, for example:

```powershell
python smplx_anim_to_alembic_batch.py C:\bedlam2\animations\training C:\bedlam2\animations\abc 4 --timeout-seconds 600 --retries 1 --skip-existing
python smplx_anim_to_objs_batch.py C:\bedlam2\animations\abc C:\bedlam2_smpl_npz 4 --timeout-seconds 600 --retries 1 --skip-existing
```

## Output Layout

For an input like:

```text
it_4006_M_2306.npz
```

the stage 1 output is:

```text
bedlam2_smpl_abc/it_4006_M/it_4006_M_2306.abc
```

the stage 2 output is:

```text
bedlam2_smpl_npz/it_4006_M/it_4006_M_2306.npz
```

## Timeout and Retry Options

`--timeout-seconds 600` means each individual Blender child process gets up to
600 seconds to convert one file. If that child is still running after the limit,
the supervisor kills that Blender process tree so the whole batch does not hang.
Use `--timeout-seconds 0` only if you want to disable timeout protection.

`--retries 1` means a failed or timed-out file gets one additional attempt. A
file marked `timeout_after_export` is not retried because the output file exists
and the log contains the expected success markers.

Use `--skip-existing` when resuming a partial run. It skips non-empty output
files that are already present.

## Logs, Report, and Stop File

Each batch run writes:

- `conversion_report.csv`
- per-file logs under `_logs/`
- a default stop-file path at `OUTPUT_DIR/STOP`

To stop a run cleanly, create the stop file in another terminal:

```bash
touch "$METADATA_ROOT"/bedlam2_smpl_abc/STOP
```

or for stage 2:

```bash
touch "$METADATA_ROOT"/bedlam2_smpl_npz/STOP
```

The supervisor kills active Blender children and stops queued work. If a process
times out after the output file was written and the log contains success markers,
the result is recorded as `timeout_after_export`; that output can be treated as
usable, but it is still visible in `conversion_report.csv`.

## Recommended Process Count

This conversion is CPU-heavy, especially the SMPL-X motion to Alembic stage. It
is useful to run it on machines with fast CPUs; GPU availability is not expected
to be the main bottleneck.

Start with 4 processes. Increase only after checking RAM, CPU load, and whether
timeouts increase. On shared Linux/Slurm nodes, do not exceed the CPU count
allocated to the job.

As a rough guide:

- 3-4 processes on about 46-64 GiB RAM
- 6 processes on about 128 GiB RAM
- 8 processes on 200 GiB+ RAM

The full intermediate ABC cache can be hundreds of GB, so confirm disk space
before running the full stage 1 conversion.
