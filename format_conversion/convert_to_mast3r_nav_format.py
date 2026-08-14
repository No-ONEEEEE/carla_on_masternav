"""
Stage 4 (v2) - Convert CARLA Scene(s) to RGB + Consolidated-Costmap Format
============================================================================
Alternate output format to convert_to_vint_format.py. Here the RGB frames
are the visual input (not the costmap), and costmaps are stored separately
as consolidated arrays rather than one grayscale JPEG per frame:
    <dataset_root>/
        compiled_costmaps.h5              # WHOLE-DATASET costmaps
            costmaps   (total_T, H, W) float32   -- all trajectories concatenated
            metadata   empty (h5py.Empty) -- placeholder, not yet populated
        <traj_name>/
            images/
                0.jpg
                1.jpg
                ...
                T.jpg                      # original RGB frames, renumbered
            traj_data.pkl                  # {"position": (T,2) xy, "yaw": (T,1)}
            costmap_<traj_name>.npz
                costmaps   (T, H, W) float32  -- this trajectory's raw metric costs
                metadata   empty array         -- placeholder, not yet populated
Costmaps are pulled from costmaps_raw/*.npy (raw metric floats), NOT the
costmaps/*.png visualisations -- same convention as pack_costmaps_h5.py.
NOTE: costmaps_raw/ is not split by cost type -- it always holds whatever
--cost was used the last time generate_gt_costmap.py was run for that scene.
The --cost flag below is a label for bookkeeping only, it does not select
between multiple precomputed cost types (same limitation as the existing
convert_to_vint_format.py).
Frame ordering in compiled_costmaps.h5 follows the order scene_roots are
processed in (sorted, for --scenes_glob). There is currently no per-frame
trajectory-id stored alongside it since "metadata" was specified as blank --
flag if you want frame->trajectory traceability added later.

POSITION/YAW FRAME: traj_data.pkl stores position and yaw EGOCENTRIC to the
trajectory's own start pose -- position[0] == (~0, ~0) and yaw[0] == ~0.
This is done by converting CARLA -> ROS coordinates first (Y flip, yaw
negation), then rotating/translating so frame 0 sits at the origin with
zero heading. See make_relative_to_start().
"""
import argparse
import os
import glob
import pickle
import numpy as np
import h5py
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ──────────────────────────────────────────────────────────────────────────────
def make_relative_to_start(position: np.ndarray, yaw: np.ndarray):
    """
    Re-express (position, yaw) -- already in ROS convention -- in an
    egocentric frame anchored at frame 0: position[0] -> (0, 0),
    yaw[0] -> 0. Standard CCW-positive world-to-body rotation, applied
    after the CARLA->ROS conversion.

    position: (T, 2) float64
    yaw:      (T,)   float64, radians
    Returns (rel_position (T,2), rel_yaw (T,)).
    """
    x0, y0 = position[0]
    theta0 = yaw[0]

    dx = position[:, 0] - x0
    dy = position[:, 1] - y0

    cos0, sin0 = np.cos(theta0), np.sin(theta0)
    rel_x = dx * cos0 + dy * sin0
    rel_y = -dx * sin0 + dy * cos0

    rel_position = np.stack([rel_x, rel_y], axis=1)
    rel_yaw = yaw - theta0
    rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    return rel_position, rel_yaw


# ──────────────────────────────────────────────────────────────────────────────
# Per-scene conversion
# ──────────────────────────────────────────────────────────────────────────────
def load_traj_costmaps(scene_root: str):
    """
    Stack costmaps_raw/*.npy (sorted) into one (T, H, W) float32 array.
    Returns (costmaps, stems) where stems are the sorted frame stems, used
    to look up matching RGB frames by filename.
    """
    costraw_dir = os.path.join(scene_root, "costmaps_raw")
    if not os.path.isdir(costraw_dir):
        raise FileNotFoundError(
            f"costmaps_raw/ not found at {costraw_dir}. "
            f"Run costmap_gen/generate_gt_costmap.py on this scene first."
        )
    raw_files = sorted(glob.glob(os.path.join(costraw_dir, "*.npy")))
    if not raw_files:
        raise FileNotFoundError(f"No costmap .npy files found in {costraw_dir}")
    stems = [os.path.splitext(os.path.basename(p))[0] for p in raw_files]
    costmaps = np.stack([np.load(p) for p in raw_files], axis=0).astype(np.float32)
    return costmaps, stems


def convert_scene(scene_root: str,
                   dataset_root: str,
                   traj_name: str,
                   cost_type: str,
                   copy_images: bool = True) -> np.ndarray:
    """
    Convert one CARLA scene folder into one RGB+costmap trajectory folder.
    Returns the (T, H, W) costmap array for this trajectory (for accumulation
    into the dataset-wide compiled_costmaps.h5).
    """
    rgb_dir     = os.path.join(scene_root, "images")
    states_path = os.path.join(scene_root, "agent_states.npy")
    if not os.path.isfile(states_path):
        raise FileNotFoundError(f"agent_states.npy not found at {states_path}")

    costmaps, stems = load_traj_costmaps(scene_root)   # (T, H, W)
    agent_states = np.load(states_path)                 # (N, 6) — x y z roll pitch yaw
    if len(stems) != len(agent_states):
        raise ValueError(
            f"Frame count mismatch in {scene_root}: "
            f"{len(stems)} costmaps vs {len(agent_states)} agent states."
        )

    out_traj_dir   = os.path.join(dataset_root, traj_name)
    out_images_dir = os.path.join(out_traj_dir, "images")
    os.makedirs(out_images_dir, exist_ok=True)

    # ── Copy RGB frames, renumbered 0.jpg, 1.jpg, ... T.jpg ────────────────────
    if copy_images:
        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"images/ RGB dir not found at {rgb_dir}")
        for i, stem in enumerate(stems):
            rgb_src = os.path.join(rgb_dir, f"{stem}.png")
            if not os.path.isfile(rgb_src):
                raise FileNotFoundError(f"Missing RGB frame {rgb_src}")
            img = cv2.imread(rgb_src, cv2.IMREAD_COLOR)
            out_path = os.path.join(out_images_dir, f"{i}.jpg")
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ── traj_data.pkl, egocentric to frame 0 ────────────────────────────────────
    position = agent_states[:, 0:2].astype(np.float64)
    position[:, 1] = -position[:, 1]          # Y_ros = -Y_carla
    yaw_deg  = agent_states[:, 5].astype(np.float64)
    yaw_rad  = -np.radians(yaw_deg)           # yaw_ros = -yaw_carla

    position, yaw_rad = make_relative_to_start(position, yaw_rad)  # anchor to frame 0

    traj_data = {"position": position, "yaw": yaw_rad.reshape(-1, 1)}
    with open(os.path.join(out_traj_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f)

    # ── Per-trajectory consolidated costmap npz ─────────────────────────────────
    npz_path = os.path.join(out_traj_dir, f"costmap_{traj_name}.npz")
    np.savez(npz_path, costmaps=costmaps, metadata=np.array([]))

    print(f"[INFO] Converted {scene_root} -> {out_traj_dir} "
          f"({len(stems)} frames, cost_type={cost_type})")
    return costmaps


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
          f"'{args.dataset_root}' (RGB + consolidated-costmap format)")

    all_costmaps = []   # list of (T, H, W) arrays to concatenate
    ref_hw = None

    for scene_root in scene_roots:
        traj_name = args.traj_name or os.path.basename(os.path.normpath(scene_root))
        try:
            costmaps = convert_scene(
                scene_root=scene_root,
                dataset_root=args.dataset_root,
                traj_name=traj_name,
                cost_type=args.cost,
                copy_images=not args.no_image_copy,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] Skipping {scene_root}: {e}")
            continue

        if ref_hw is None:
            ref_hw = costmaps.shape[1:]
        elif costmaps.shape[1:] != ref_hw:
            print(f"[WARN] {scene_root} costmap shape {costmaps.shape[1:]} "
                  f"!= expected {ref_hw} -- excluding from compiled_costmaps.h5 "
                  f"(per-trajectory .npz was still written normally).")
            continue

        all_costmaps.append(costmaps)

    if not all_costmaps:
        print("[WARN] No trajectories with matching costmap shape were found -- "
              "compiled_costmaps.h5 was NOT written.")
        return

    combined = np.concatenate(all_costmaps, axis=0)   # (total_T, H, W)
    h5_path = args.h5_path or os.path.join(args.dataset_root, "compiled_costmaps.h5")
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("costmaps", data=combined, dtype=np.float32)
        h5f.create_dataset("metadata", data=h5py.Empty("f"))

    print(f"[INFO] Done. compiled_costmaps.h5 written to {h5_path} "
          f"(costmaps shape={combined.shape})")
    print(f"[INFO] Dataset ready at: {args.dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CARLA costmap dataset to RGB-input + "
                     "consolidated-costmap (.npz per trajectory + "
                     "dataset-wide .h5) training format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene_root", default=None,
                        help="Path to a single out/<scene_name> directory.")
    group.add_argument("--scenes_glob", default=None,
                        help='Glob pattern matching multiple scene dirs.')
    parser.add_argument("--dataset_root", required=True,
                         help="Output root for the converted dataset. Also "
                              "the default location of compiled_costmaps.h5 "
                              "unless --h5_path is given.")
    parser.add_argument("--traj_name", default=None,
                         help="Trajectory folder name (single-scene mode only).")
    parser.add_argument("--cost", default="groundplane",
                         choices=["euclidean3d", "groundplane", "geodesic"],
                         help="Label only -- costmaps_raw/ is not split by "
                              "cost type, this just records what was used.")
    parser.add_argument("--no_image_copy", action="store_true",
                         help="Skip copying RGB frames into images/ subfolder "
                              "(costmap npz/h5 and traj_data.pkl still written).")
    parser.add_argument("--h5_path", default=None,
                         help="Explicit path for the dataset-wide "
                              "compiled_costmaps.h5. Defaults to "
                              "<dataset_root>/compiled_costmaps.h5.")
    main(parser.parse_args())
