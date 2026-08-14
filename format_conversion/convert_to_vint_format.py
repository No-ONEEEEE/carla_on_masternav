"""
Stage 4 – Convert CARLA Scene to ViNT/GNM/NoMaD Training Format (Pure H,W)
===========================================================================
Converts the output of collect_dataset(_grp).py + compute_3d_points.py +
generate_costmaps.py into the dataset format expected by the General
Navigation Models (GNM / ViNT / NoMaD) training code:

    https://github.com/robodhruv/visualnav-transformer

Per the repo's required structure:

    <dataset_name>/
        <traj_name>/
            0.jpg
            1.jpg
            ...
            T.jpg
            traj_data.pkl      # {"position": (T,2) xy,  "yaw": (T,) }

IMPORTANT — The costmap image is used in place of the RGB image as the network's 
visual input. Per your training setup requirement, these are saved strictly as 
single-channel (H, W) grayscale JPEGs.

position / yaw come directly from agent_states.npy (x, y, yaw), matching
the repo's odometry convention (planar position + heading).
"""

import argparse
import os
import glob
import pickle
import shutil

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Core conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_scene(scene_root: str,
                  dataset_root: str,
                  traj_name: str,
                  cost_type: str,
                  copy_rgb: bool = True) -> None:
    """
    Convert one CARLA scene folder into one ViNT-format trajectory folder.
    """
    costmap_dir = os.path.join(scene_root, "costmaps")
    rgb_dir     = os.path.join(scene_root, "images")
    states_path = os.path.join(scene_root, "agent_states.npy")

    if not os.path.isdir(costmap_dir):
        raise FileNotFoundError(
            f"costmaps/ not found at {costmap_dir}. "
            f"Run generate_costmaps.py on this scene first."
        )
    if not os.path.isfile(states_path):
        raise FileNotFoundError(f"agent_states.npy not found at {states_path}")

    costmap_files = sorted(glob.glob(os.path.join(costmap_dir, "*.png")))
    if not costmap_files:
        raise FileNotFoundError(f"No costmap PNGs found in {costmap_dir}")

    agent_states = np.load(states_path)   # (N, 6) — x y z roll pitch yaw

    if len(costmap_files) != len(agent_states):
        raise ValueError(
            f"Frame count mismatch in {scene_root}: "
            f"{len(costmap_files)} costmaps vs {len(agent_states)} agent states. "
        )

    out_traj_dir = os.path.join(dataset_root, traj_name)
    os.makedirs(out_traj_dir, exist_ok=True)
    if copy_rgb:
        out_rgb_dir = os.path.join(out_traj_dir, "rgb")
        os.makedirs(out_rgb_dir, exist_ok=True)

    # ── Copy frames, renumbered 0.jpg, 1.jpg, ... T.jpg ────────────────────────
    for i, costmap_path in enumerate(costmap_files):
        # Read strictly as a 1-channel grayscale (H, W) matrix
        img_gray = cv2.imread(costmap_path, cv2.IMREAD_GRAYSCALE)
        
        # Save directly as a 1-channel (H, W) JPEG image file
        out_path = os.path.join(out_traj_dir, f"{i}.jpg")
        cv2.imwrite(out_path, img_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if copy_rgb:
            stem = os.path.splitext(os.path.basename(costmap_path))[0]
            rgb_src = os.path.join(rgb_dir, f"{stem}.png")
            if os.path.isfile(rgb_src):
                shutil.copy(rgb_src, os.path.join(out_rgb_dir, f"{i}.jpg"))

    # ── Build traj_data.pkl ───────────────────────────────────────────────────
    position = agent_states[:, 0:2].astype(np.float64)
    position[:, 1] = -position[:, 1]          # Y_ros = -Y_carla
    yaw_deg  = agent_states[:, 5].astype(np.float64)
    yaw_rad  = -np.radians(yaw_deg)           # yaw_ros = -yaw_carla

    traj_data = {
        "position": position,
        "yaw":      yaw_rad,
    }

    pkl_path = os.path.join(out_traj_dir, "traj_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(traj_data, f)

    print(f"[INFO] Converted {scene_root} -> {out_traj_dir} "
          f"({len(costmap_files)} frames, cost_type={cost_type})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.dataset_root, exist_ok=True)

    if args.scene_root:
        scene_roots = [args.scene_root]
    elif args.scenes_glob:
        scene_roots = sorted(glob.glob(args.scenes_glob))
        scene_roots = [s for s in scene_roots if os.path.isdir(s)]
    else:
        raise ValueError("Provide either --scene_root or --scenes_glob.")

    if not scene_roots:
        raise FileNotFoundError("No matching scene directories found.")

    print(f"[INFO] Converting {len(scene_roots)} scene(s) into "
          f"'{args.dataset_root}' (ViNT/GNM/NoMaD format)")

    for scene_root in scene_roots:
        traj_name = args.traj_name or os.path.basename(os.path.normpath(scene_root))
        try:
            convert_scene(
                scene_root=scene_root,
                dataset_root=args.dataset_root,
                traj_name=traj_name,
                cost_type=args.cost,
                copy_rgb=not args.no_rgb_copy,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] Skipping {scene_root}: {e}")

    print(f"[INFO] Done. Dataset ready at: {args.dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CARLA costmap dataset to pure (H, W) ViNT training format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene_root", default=None,
                       help="Path to a single out/<scene_name> directory.")
    group.add_argument("--scenes_glob", default=None,
                       help='Glob pattern matching multiple scene dirs.')

    parser.add_argument("--dataset_root", required=True,
                        help="Output root for the ViNT-format dataset.")
    parser.add_argument("--traj_name", default=None,
                        help="Trajectory folder name (single-scene mode only).")
    parser.add_argument("--cost", default="groundplane",
                        choices=["euclidean3d", "groundplane", "geodesic"],
                        help="Which costmaps subfolder to pull from.")
    parser.add_argument("--no_rgb_copy", action="store_true",
                        help="Skip copying original RGB frames into rgb/ subfolder.")

    main(parser.parse_args())