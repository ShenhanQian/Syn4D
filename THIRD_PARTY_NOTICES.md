# License and Third-Party Asset Notice

This dataset contains rendered videos/images, annotations, camera parameters, trajectories, and metadata generated using third-party assets. The components created by the dataset authors are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/. All third-party assets remain governed by their original licenses.

## BEDLAM2

Some sequences depend on BEDLAM2. BEDLAM2 data is not redistributed with this dataset. Users must obtain BEDLAM2 separately from the official source and agree to the BEDLAM2 license before using any BEDLAM2-dependent sequences.

## Fab / Unreal Scenes

Some scenes use Fab assets under the Fab Standard License: https://www.fab.com/eula

The Fab assets used in this dataset were selected from Fab listings under the Fab Standard License that indicated usage with AI is allowed. We do not redistribute original Fab source assets, including Unreal project files, `.uasset` files, meshes, textures, materials, or standalone asset packages.

## Objaverse Objects

Some objects are sourced from Objaverse. Objaverse is licensed under ODC-By v1.0, while individual objects retain their original licenses as specified in the Objaverse metadata.

We provide a sequence-to-asset mapping CSV with the following format:

```csv
scene,sequence_name,asset_type,asset,object_group,object_id
```

For rows where `asset_type` is `objaverse_object`, `object_id` is the Objaverse object UID.

## Google Scanned Objects

Some objects are sourced from the Google Scanned Objects dataset, which is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Third-Party Metadata

Third-party asset information is provided through:

- `data/syn4d_v1_stride_1/sequence_to_asset_mapping.csv`
- `data/syn4d_v1_stride_5/sequence_to_asset_mapping.csv`
- `LICENSES/` for relevant license texts or archived license pages where applicable

This dataset license does not grant additional rights to BEDLAM2, Fab, Objaverse, Google Scanned Objects, or any other third-party assets. Users are responsible for complying with the corresponding original licenses.
