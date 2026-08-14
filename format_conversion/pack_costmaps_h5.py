"""
Stage 5 - Pack Raw Costmaps into gs2-style HDF5 (No Key Prefix)
==============================================================
Packs the per-frame raw costmaps (costmaps_raw/*.npy, metric distances in
metres) produced by generate_gt_costmap.py into a single HDF5 file matching
the schema expected by the mast3r-nav training config:

    graphs_path: "<some_dir>"
    precomputed_filename: "_costmaps.h5"
    datasets:
      my_dataset:
        ...

  ->  file written to:  <graphs_path>/<dataset_name>_costmaps.h5

Schema:

    <h5file>
        <traj_name>_0/
            pls_pixels   (60, 80) float32   # raw metric cost, resized down
        <traj_name>_1/
            pls_pixels   (60, 80) float32
        ...

  - Top-level key = "{traj_name}_{frame_idx}"
    (frame_idx is 0-indexed, NOT zero-padded, contiguous)
  - Each group has exactly one dataset "pls_pixels", shape (60, 80), float32
  - Values are raw metric distances (metres), NOT normalised to [0,1] —
    do NOT use the costmaps/*.png (those are the colour visualisation only)
  - traj_name here MUST match the trajectory folder name used in the
    ViNT-format dataset (i.e. the same name you passed as --traj_name /
    used as the scene folder name in convert_to_vint_format.py), since
    the training loader will look up costmap keys using that name.

Usage:
    # Pack every scene under out/ that has costmaps_raw/
    python pack_costmaps_h5.py \\
        --out out \\
        --scenes_glob "out/*" \\
        --dataset_name my_dataset \\
        --graphs_path /scratch2/utkarsh.malaiya/datasets/gs2_modified

    # Pack a specific list of scenes
    python pack_costmaps_h5.py \\
        --out out \\
        --scenes world00_traj00 world00_traj01 \\
        --dataset_name my_dataset \\
        --graphs_path /path/to/graphs

    # Custom output path instead of <graphs_path>/<dataset_name>_costmaps.h5
    python pack_costmaps_h5.py --out out --scenes_glob "out/*" \\
        --dataset_name my_dataset --output_path /tmp/my_dataset_costmaps.h5
"""

import argparse
import os
import glob

import numpy as np
import h5py
import cv2


def pack_scene(h5file: h5py.File,
                scene_root: str,
                traj_name: str,
                target_h: int,
                target_w: int) -> int:
    """
    Resize + write every costmaps_raw/*.npy for one scene into the h5 file.

    Returns the number of frames written.
    """
    raw_dir = os.path.join(scene_root, "costmaps_raw")
    if not os.path.isdir(raw_dir):
        print(f"[WARN] Skipping {scene_root}: no costmaps_raw/ found "
              f"(run generate_gt_costmap.py first).")
        return 0

    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))
    if not raw_files:
        print(f"[WARN] Skipping {scene_root}: costmaps_raw/ is empty.")
        return 0

    for frame_idx, raw_path in enumerate(raw_files):
        cost = np.load(raw_path).astype(np.float32)   # (H, W) metric metres

        # Resize down to target resolution (default 60x80). cv2.resize takes (W, H) order.
        cost_resized = cv2.resize(
            cost, (target_w, target_h), interpolation=cv2.INTER_AREA
        ).astype(np.float32)

        # Key strictly formatted without any prefix string
        key = f"{traj_name}_{frame_idx}"
        grp = h5file.create_group(key)
        grp.create_dataset("pls_pixels", data=cost_resized, dtype=np.float32)

    print(f"[INFO] Packed {len(raw_files)} frame(s) from {scene_root} "
          f"as trajectory '{traj_name}'")
    return len(raw_files)


def main(args):
    if args.scenes:
        scene_names = args.scenes
        scene_roots = [os.path.join(args.out, s) for s in scene_names]
    elif args.scenes_glob:
        scene_roots = sorted(glob.glob(args.scenes_glob))
        scene_roots = [s for s in scene_roots if os.path.isdir(s)]
        scene_names = [os.path.basename(os.path.normpath(s)) for s in scene_roots]
    else:
        raise ValueError("Provide either --scenes or --scenes_glob.")

    if not scene_roots:
        raise FileNotFoundError("No matching scene directories found.")

    if args.output_path:
        output_path = args.output_path
    else:
        if not args.graphs_path:
            raise ValueError(
                "Provide --graphs_path (to write "
                "<graphs_path>/<dataset_name>_costmaps.h5) or --output_path directly."
            )
        os.makedirs(args.graphs_path, exist_ok=True)
        output_path = os.path.join(args.graphs_path,
                                   f"{args.dataset_name}_costmaps.h5")

    if os.path.isfile(output_path) and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it, "
            f"or delete it manually first (existing keys are not merged)."
        )

    print(f"[INFO] Packing {len(scene_roots)} scene(s) -> {output_path}")
    print(f"[INFO] Target resolution: ({args.target_h}, {args.target_w})")

    total_frames = 0
    with h5py.File(output_path, "w") as h5file:
        for scene_root, traj_name in zip(scene_roots, scene_names):
            total_frames += pack_scene(
                h5file=h5file,
                scene_root=scene_root,
                traj_name=traj_name,
                target_h=args.target_h,
                target_w=args.target_w,
            )

    print(f"[INFO] Done. {total_frames} total frame(s) packed into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pack raw costmaps into gs2-style HDF5 without any key prefixes."
    )
    parser.add_argument("--out", default="out",
                        help="Root directory containing scene folders "
                             "(same --out used in the collection pipeline).")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenes", nargs="+", default=None,
                       help="Explicit list of scene/trajectory names "
                            "(folders under --out), e.g. world00_traj00 world00_traj01")
    group.add_argument("--scenes_glob", default=None,
                       help='Glob pattern for scene dirs, e.g. "out/*"')

    parser.add_argument("--dataset_name", required=True,
                        help="Dataset name used in data_config.yaml / "
                             "training config (e.g. 'my_dataset'). Used to "
                             "build the default output filename.")
    parser.add_argument("--graphs_path", default=None,
                        help="Directory to write "
                             "<dataset_name>_costmaps.h5 into (matches the "
                             "training config's graphs_path).")
    parser.add_argument("--output_path", default=None,
                        help="Explicit output .h5 path, overrides "
                             "--graphs_path/--dataset_name naming.")

    parser.add_argument("--target_h", type=int, default=60,
                        help="Output costmap height (default 60).")
    parser.add_argument("--target_w", type=int, default=80,
                        help="Output costmap width (default 80).")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output file if it already exists.")

    main(parser.parse_args())
