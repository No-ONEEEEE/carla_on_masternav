"""
Stage 3 – Ground-Truth Costmap Generation (Single Channel)
=========================================================
For every saved frame, transforms per-pixel 3D points (camera frame) into
world coordinates, computes a scalar cost to the final goal, and saves both
a single-channel (H, W) grayscale PNG visualization and a raw float32 .npy cost map.

The goal is the ego's own final position — `agent_states[-1, :3]` — not the
last anchor waypoint in `trajectory.npy`. This guarantees the minimum cost is
always located in the last saved frame, since that is by definition where the
ego ended up (distance from that point to itself is zero).

Three cost types are available:

  1) euclidean3d   – full 3D Euclidean distance  sqrt(dx²+dy²+dz²)
  2) groundplane   – 2D ground-plane distance     sqrt(dx²+dy²)   (recommended for driving)
  3) geodesic      – road-following distance via CARLA's GlobalRoutePlanner
"""

import argparse
import os
import glob
import sys
from multiprocessing import Pool

sys.path.insert(0, "/home2/adamya.singhal/Carla-0.10.0-Linux-Shipping/PythonAPI/carla")

import numpy as np
import cv2

# ──────────────────────────────────────────────────────────────────────────────
# Multiprocessing Worker Initialization
# ──────────────────────────────────────────────────────────────────────────────

_worker_grp = None
_worker_carla = None
_worker_map = None

def init_worker(host, port):
    """
    Initializes an isolated CARLA client connection and route planner 
    for each background worker thread exactly once when spawned.
    """
    global _worker_grp, _worker_carla, _worker_map
    try:
        import carla
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        _worker_carla = carla
        client = carla.Client(host, port)
        client.set_timeout(15.0)
        world = client.get_world()
        _worker_map = world.get_map()
        _worker_grp = GlobalRoutePlanner(_worker_map, sampling_resolution=2.0)
    except Exception as e:
        print(f"[ERROR] Worker process failed initialization connection to CARLA: {e}")


def compute_pixel_task(args_tuple):
    """
    Worker task function that calculates the geodesic route cost for a single coordinate,
    optionally appending a penalised lateral offset for off-road configurations.
    """
    r, c, px, goal, sky_pixel, lateral_penalty, lateral_penalty_weight = args_tuple

    if sky_pixel:
        return r, c, -1.0

    if _worker_grp is None or _worker_carla is None:
        return r, c, float(np.linalg.norm(px[:2] - goal[:2]))

    src_loc = _worker_carla.Location(x=float(px[0]), y=float(px[1]), z=float(px[2]))
    goal_loc = _worker_carla.Location(x=float(goal[0]), y=float(goal[1]), z=float(goal[2]))

    lateral_offset = 0.0
    if lateral_penalty and _worker_map is not None:
        try:
            snapped_wp = _worker_map.get_waypoint(
                src_loc,
                project_to_road=True,
                lane_type=_worker_carla.LaneType.Driving
            )
            if snapped_wp is not None:
                lateral_offset = src_loc.distance(snapped_wp.transform.location)
        except Exception:
            lateral_offset = 0.0

    try:
        route = _worker_grp.trace_route(src_loc, goal_loc)
        if len(route) < 2:
            base_cost = float(np.linalg.norm(px[:2] - goal[:2]))
            if lateral_penalty:
                return r, c, base_cost + (lateral_penalty_weight * lateral_offset)
            return r, c, base_cost
        
        total = 0.0
        for k in range(1, len(route)):
            a = route[k-1][0].transform.location
            b = route[k][0].transform.location
            total += np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

        if lateral_penalty:
            total += lateral_penalty_weight * lateral_offset

        return r, c, total
    except Exception:
        base_cost = float(np.linalg.norm(px[:2] - goal[:2]))
        if lateral_penalty:
            return r, c, base_cost + (lateral_penalty_weight * lateral_offset)
        return r, c, base_cost


# ──────────────────────────────────────────────────────────────────────────────
# Cost functions
# ──────────────────────────────────────────────────────────────────────────────

def cost_euclidean3d(pts_world: np.ndarray, goal: np.ndarray, sky_mask: np.ndarray = None) -> np.ndarray:
    diff = pts_world - goal[np.newaxis, np.newaxis, :]
    cost = np.linalg.norm(diff, axis=-1).astype(np.float32)
    if sky_mask is not None:
        cost[sky_mask] = -1.0
    return cost


def cost_groundplane(pts_world: np.ndarray, goal: np.ndarray, sky_mask: np.ndarray = None) -> np.ndarray:
    diff_xy = pts_world[:, :, :2] - goal[np.newaxis, np.newaxis, :2]
    cost = np.linalg.norm(diff_xy, axis=-1).astype(np.float32)
    if sky_mask is not None:
        cost[sky_mask] = -1.0
    return cost


def cost_geodesic_parallel(pts_world: np.ndarray,
                           goal: np.ndarray,
                           sky_mask: np.ndarray,
                           pool: Pool,
                           lateral_penalty: bool = False,
                           lateral_penalty_weight: float = 1.0) -> np.ndarray:
    H, W = pts_world.shape[:2]
    STRIDE = 2     

    rows = np.arange(0, H, STRIDE)
    cols = np.arange(0, W, STRIDE)

    tasks = []
    for r in rows:
        for c in cols:
            tasks.append((r, c, pts_world[r, c], goal, sky_mask[r, c], lateral_penalty, lateral_penalty_weight))

    results = pool.map(compute_pixel_task, tasks)

    sparse_H = len(rows)
    sparse_W = len(cols)
    cost_sparse = np.zeros((sparse_H, sparse_W), dtype=np.float32)

    row_to_ri = {r: ri for ri, r in enumerate(rows)}
    col_to_ci = {c: ci for ci, c in enumerate(cols)}

    for r, c, val in results:
        ri = row_to_ri[r]
        ci = col_to_ci[c]
        cost_sparse[ri, ci] = val

    cost_full = cv2.resize(cost_sparse, (W, H), interpolation=cv2.INTER_LINEAR)
    
    if sky_mask is not None:
        cost_full[sky_mask] = -1.0
        
    return cost_full.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Custom colormap & Scale Math
# ──────────────────────────────────────────────────────────────────────────────

def build_bgr_lut() -> np.ndarray:
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    idx = np.arange(256, dtype=np.float32)

    t1 = idx[:128] / 127.0
    lut[:128, 0, 0] = ((1 - t1) * 255)          # B: 255 -> 0
    lut[:128, 0, 1] = (t1 * 255)                # G: 0   -> 255
    lut[:128, 0, 2] = 0                         # R: 0

    t2 = (idx[128:] - 128.0) / 127.0
    lut[128:, 0, 0] = 0                         # B: 0
    lut[128:, 0, 1] = ((1 - t2) * 255)          # G: 255 -> 0
    lut[128:, 0, 2] = (t2 * 255)                # R: 0   -> 255

    return lut


BGR_COLORMAP = build_bgr_lut()


def compute_scale_ref(costs: np.ndarray, percentile: float) -> float:
    """Robust scale reference — ignores extreme outlier pixels."""
    valid = costs[costs != -1.0]
    if len(valid) == 0:
        return 1.0
    ref = float(np.percentile(valid, percentile))
    return ref if ref > 0 else 1.0


def cam_to_world(pts_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    H, W, _ = pts_cam.shape
    pts_flat = pts_cam.reshape(-1, 3)
    ones     = np.ones((pts_flat.shape[0], 1), dtype=np.float64)
    pts_h    = np.hstack([pts_flat.astype(np.float64), ones])
    pts_w    = (T @ pts_h.T).T
    return pts_w[:, :3].reshape(H, W, 3).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution Block
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    root            = os.path.join(args.out, args.scene)
    points_dir      = os.path.join(root, "3d_points")
    extrinsics_path = os.path.join(root, "camera_extrinsics.npy")
    agent_states_path = os.path.join(root, "agent_states.npy")
    costmap_dir     = os.path.join(root, "costmaps")
    costraw_dir     = os.path.join(root, "costmaps_raw")
    costcolor_dir   = os.path.join(root, "costmaps_color")

    dirs_to_reset = [costmap_dir, costraw_dir]
    if args.color:
        dirs_to_reset.append(costcolor_dir)

    for d in dirs_to_reset:
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)
        os.makedirs(d)

    if not os.path.isfile(agent_states_path):
        raise FileNotFoundError(f"agent_states.npy not found at {agent_states_path}")
    if not os.path.isfile(extrinsics_path):
        raise FileNotFoundError(f"camera_extrinsics.npy not found at {extrinsics_path}")

    agent_states = np.load(agent_states_path)
    extrinsics   = np.load(extrinsics_path)

    goal = agent_states[-1, :3].astype(np.float64)
    print(f"[INFO] Goal (world, last ego position): "
          f"x={goal[0]:.2f}  y={goal[1]:.2f}  z={goal[2]:.2f}")

    point_files = sorted(glob.glob(os.path.join(points_dir, "*.npy")))
    if not point_files:
        raise FileNotFoundError(f"No 3D point .npy files found in {points_dir}")

    print(f"[INFO] Cost type : {args.cost}")
    if args.cost == "geodesic":
        status_string = f"ON (weight={args.lateral_penalty_weight})" if args.lateral_penalty else "OFF (default)"
        print(f"[INFO] Lateral penalty : {status_string}")
    print(f"[INFO] Frames    : {len(point_files)}")

    pool = None
    if args.cost == "geodesic":
        NUM_WORKERS = 20
        print(f"[INFO] Initializing parallel Pool utilizing {NUM_WORKERS} local worker threads...")
        pool = Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(args.host, args.port))

    raw_costs_list = []
    try:
        for idx, pf in enumerate(point_files):
            pts_cam   = np.load(pf)
            T         = extrinsics[idx]
            pts_world = cam_to_world(pts_cam, T)

            DEPTH_FAR = 1000.0
            sky_mask = (pts_cam[..., 2] >= DEPTH_FAR)

            if args.cost == "euclidean3d":
                cost = cost_euclidean3d(pts_world, goal, sky_mask)
            elif args.cost == "groundplane":
                cost = cost_groundplane(pts_world, goal, sky_mask)
            elif args.cost == "geodesic":
                cost = cost_geodesic_parallel(
                    pts_world, goal, sky_mask, pool,
                    lateral_penalty=args.lateral_penalty,
                    lateral_penalty_weight=args.lateral_penalty_weight
                )
            else:
                raise ValueError(f"Unknown cost type: {args.cost}")

            raw_costs_list.append(cost)
            print(f" -> Evaluated frame {idx+1}/{len(point_files)}")
            
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Compute global scale reference if per-image mode is off
    all_valid_costs = np.concatenate([c[c != -1.0] for c in raw_costs_list])
    if len(all_valid_costs) > 0:
        global_ref = float(np.percentile(all_valid_costs, args.scale_percentile))
    else:
        global_ref = 1.0
    if global_ref == 0.0:
        global_ref = 1.0

    print(f"[INFO] Scale reference (Global {args.scale_percentile}th percentile): {global_ref:.2f} m")
    print(f"[INFO] Intensity scale : {'per-image' if args.per_image_scale else 'global'}")
    print(f"[INFO] Gamma applied   : {args.gamma}")

    # Pass 2: Normalize, apply Gamma Correction, and Save
    for idx, (pf, cost) in enumerate(zip(point_files, raw_costs_list)):
        stem = os.path.splitext(os.path.basename(pf))[0]

        # Save raw un-scaled float metrics
        np.save(os.path.join(costraw_dir, f"{stem}.npy"), cost)

        # Dynamic range mapping selection
        if args.per_image_scale:
            frame_ref = compute_scale_ref(cost, percentile=args.scale_percentile)
        else:
            frame_ref = global_ref

        # Core scaling configuration
        cost_norm = np.clip(cost / frame_ref, 0.0, 1.0)
        
        # Gamma tracking modification (Gamma < 1 expansions lift near values away from blue)
        if args.gamma != 1.0:
            cost_norm = cost_norm ** args.gamma

        # Discretize map range to standard pixel intensities
        cost_gray = (cost_norm * 255).astype(np.uint8)
        cost_gray[cost == -1.0] = 0  # Re-mask structural sky bounds to absolute black

        cv2.imwrite(os.path.join(costmap_dir, f"{stem}.png"), cost_gray)

        if args.color:
            cost_color = cv2.applyColorMap(cost_gray, BGR_COLORMAP)
            cost_color[cost == -1.0] = [0, 0, 0]
            cv2.imwrite(os.path.join(costcolor_dir, f"{stem}.png"), cost_color)

    print(f"[INFO] Single channel (H, W) costmaps saved to {costmap_dir}")
    print(f"[INFO] Raw float32 costs saved to {costraw_dir}")
    if args.color:
        print(f"[INFO] Colored (H, W, 3) costmaps saved to {costcolor_dir}")


if __name__ == "__main__":
    COST_TYPES = ["euclidean3d", "groundplane", "geodesic"]

    parser = argparse.ArgumentParser(
        description="Generate single-channel GT costmaps from saved 3D point clouds."
    )
    parser.add_argument("--out",   default="out")
    parser.add_argument("--scene", default="scene_00")
    parser.add_argument("--cost",  default="euclidean3d", choices=COST_TYPES)
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  default=2000, type=int)
    parser.add_argument("--color", action="store_true",
                        help="Also save RGB colored costmaps under costmaps_color/.")
    parser.add_argument("--per_image_scale", action="store_true",
                        help="Normalize each frame's costmap by its own reference profile.")
    parser.add_argument("--lateral_penalty", action="store_true",
                        help="Geodesic cost only. Explicitly penalise off-road points.")
    parser.add_argument("--lateral_penalty_weight", default=1.0, type=float,
                        help="Multiplier on the lateral offset distance.")
    
    # Newly introduced visibility args
    parser.add_argument("--scale_percentile", default=95.0, type=float,
                        help="Percentile used as the normalization reference instead "
                             "of the true max to isolate near-field contrast. Default: 95.")
    parser.add_argument("--gamma", default=0.6, type=float,
                        help="Gamma correction applied after normalization. "
                             "<1.0 expands the low-cost range (more color spread "
                             "between road/sidewalk), 1.0 = no change. Default: 0.6.")

    main(parser.parse_args())
