"""
Open-loop evaluation of ObjRelLearntController against convert_to_mast3r_nav_format.py
trajectories, feeding the pre-computed per-frame costmaps directly to the controller
(no mast3r call anywhere in this path).

For every trajectory directory of the form:
    <traj_dir>/
        images/                 0.jpg, 1.jpg, ..., T-1.jpg
        traj_data.pkl           {"position": (T,2), "yaw": (T,1)}  -- egocentric to frame 0
        costmap_<traj_name>.npz {"costmaps": (T,H,W) float32}

this script:
  1. Runs controller.predict(rgb, costmap) at every timestep t (open loop: the real
     recorded rgb/costmap is used at every step, never the model's own rollout).
  2. Re-expresses the GT trajectory in the frame egocentric to t (same rotation math
     as make_relative_to_start(), just anchored at t instead of 0), so it's directly
     comparable to action_pred (which is egocentric to the current frame).
  3. Matches predicted waypoint k -> GT frame t + (k+1)*waypoint_stride, and computes
     the L2 (Euclidean) distance between predicted (dx, dy) and GT (dx, dy).
  4. Writes evaluation/<traj_name>/{loss.json, loss_curve.png, video.mp4}.

Assumptions worth double-checking against your training code:
  - L2 loss is computed on position (dx, dy) only, not heading. Per-timestep loss is
    the mean L2 over the valid prediction horizon; total trajectory loss is reported
    as both sum and mean over timesteps (see loss.json).
  - waypoint_stride (default 1) controls the GT-frame spacing used to match each of
    the len_traj_pred predicted waypoints; change via --waypoint_stride if your model
    was trained with a different horizon spacing.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from natsort import natsorted

# repo-relative imports -- run this script from the mast3r-nav repo root
from libs.control.learnt_controller import ObjRelLearntController
from notebooks.viz_utils import gen_bearings_from_waypoints


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate helpers (mirrors convert_to_mast3r_nav_format.make_relative_to_start,
# but anchored at an arbitrary frame index t instead of frame 0)
# ──────────────────────────────────────────────────────────────────────────────
def relative_to_frame(position: np.ndarray, yaw: np.ndarray, t: int):
    """
    Re-express (position, yaw) -- already egocentric to frame 0 -- in the frame
    egocentric to index t: position[t] -> (0, 0), yaw[t] -> 0.

    position: (T, 2) float64
    yaw:      (T,)   float64, radians
    Returns (rel_position (T,2), rel_yaw (T,)).
    """
    x0, y0 = position[t]
    theta0 = yaw[t]

    dx = position[:, 0] - x0
    dy = position[:, 1] - y0

    cos0, sin0 = np.cos(theta0), np.sin(theta0)
    rel_x = dx * cos0 + dy * sin0
    rel_y = -dx * sin0 + dy * cos0

    rel_position = np.stack([rel_x, rel_y], axis=1)
    rel_yaw = yaw - theta0
    rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi
    return rel_position, rel_yaw


def gt_waypoints_for_frame(position, yaw, t, len_traj_pred, stride):
    """
    Build the (K, 4) GT waypoint array -- (dx, dy, cos(dtheta), sin(dtheta)) -- in
    the frame egocentric to t, for whichever of the len_traj_pred horizon steps are
    still inside the trajectory. K <= len_traj_pred.
    """
    T = position.shape[0]
    rel_pos, rel_yaw = relative_to_frame(position, yaw, t)

    rows = []
    valid_k = []
    for k in range(len_traj_pred):
        gt_idx = t + (k + 1) * stride
        if gt_idx >= T:
            break
        dx, dy = rel_pos[gt_idx]
        dtheta = rel_yaw[gt_idx]
        rows.append([dx, dy, np.cos(dtheta), np.sin(dtheta)])
        valid_k.append(k)
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), []
    return np.array(rows, dtype=np.float32), valid_k


def l2_loss(pred_wp: np.ndarray, gt_wp: np.ndarray):
    """
    pred_wp, gt_wp: (K, >=2) arrays, first two columns are (dx, dy).
    Returns (per_waypoint_losses (K,), mean_loss).
    """
    if len(pred_wp) == 0:
        return np.array([]), float("nan")
    diff = pred_wp[:, :2] - gt_wp[:, :2]
    per_wp = np.linalg.norm(diff, axis=1)
    return per_wp, float(per_wp.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory-vs-GT plotting (reuses the notebook's exact plot_traj conventions,
# just drawn twice on one axis with distinct colors + legend)
# ──────────────────────────────────────────────────────────────────────────────
def plot_pred_vs_gt(ax, pred_wp, gt_wp, quiver_freq=1):
    ax.grid(False)
    ax.axis("off")
    ax.set_ylim(-1, 12)
    ax.set_xlim(-4, 4)
    ax.invert_xaxis()
    ax.set_aspect("equal", "box")

    if len(pred_wp) > 0:
        ax.plot(pred_wp[:, 1], pred_wp[:, 0], color="c", alpha=0.8, marker="o", label="predicted")
        bearings = gen_bearings_from_waypoints(pred_wp)
        ax.quiver(
            pred_wp[::quiver_freq, 1], pred_wp[::quiver_freq, 0],
            -bearings[::quiver_freq, 1], bearings[::quiver_freq, 0],
            color="y", scale=1.0,
        )
    if len(gt_wp) > 0:
        ax.plot(gt_wp[:, 1], gt_wp[:, 0], color="m", alpha=0.8, marker="o", label="ground truth")
        bearings = gen_bearings_from_waypoints(gt_wp)
        ax.quiver(
            gt_wp[::quiver_freq, 1], gt_wp[::quiver_freq, 0],
            -bearings[::quiver_freq, 1], bearings[::quiver_freq, 0],
            color="orange", scale=1.0,
        )
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def plot_predicted_only(ax, pred_wp, quiver_freq=1):
    """Exact reproduction of plot_traj() from viz_utils.py / learnt_controller.py."""
    ax.plot(pred_wp[:, 1], pred_wp[:, 0], color="c", alpha=0.5, marker="o")
    bearings = gen_bearings_from_waypoints(pred_wp)
    ax.quiver(
        pred_wp[::quiver_freq, 1], pred_wp[::quiver_freq, 0],
        -bearings[::quiver_freq, 1], bearings[::quiver_freq, 0],
        color="y", scale=1.0,
    )
    ax.grid(False)
    ax.axis("off")
    ax.set_ylim(-1, 12)
    ax.set_xlim(-4, 4)
    ax.invert_xaxis()
    ax.set_aspect("equal", "box")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def discover_trajectories(dataset_root: Path):
    """
    Returns a list of trajectory directories under dataset_root. Supports both:
      - dataset_root itself being a single trajectory dir (has traj_data.pkl), and
      - dataset_root being a dataset with several <traj_name>/ subdirs.
    """
    if (dataset_root / "traj_data.pkl").is_file():
        return [dataset_root]
    traj_dirs = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and (child / "traj_data.pkl").is_file():
            traj_dirs.append(child)
    if not traj_dirs:
        raise FileNotFoundError(
            f"No trajectory directories (containing traj_data.pkl) found under {dataset_root}"
        )
    return traj_dirs


def load_trajectory(traj_dir: Path):
    traj_name = traj_dir.name
    img_dir = traj_dir / "images"
    img_paths = natsorted(img_dir.iterdir(), key=lambda p: int(p.stem))

    npz_path = traj_dir / f"costmap_{traj_name}.npz"
    if not npz_path.is_file():
        candidates = list(traj_dir.glob("costmap_*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No costmap_*.npz found in {traj_dir}")
        npz_path = candidates[0]
    costmaps = np.load(npz_path)["costmaps"]  # (T, H, W) float32

    with open(traj_dir / "traj_data.pkl", "rb") as f:
        traj_data = pickle.load(f)
    position = np.asarray(traj_data["position"], dtype=np.float64)  # (T, 2)
    yaw = np.asarray(traj_data["yaw"], dtype=np.float64).reshape(-1)  # (T,)

    T = len(img_paths)
    if not (T == costmaps.shape[0] == position.shape[0] == yaw.shape[0]):
        raise ValueError(
            f"[{traj_name}] length mismatch: images={T}, costmaps={costmaps.shape[0]}, "
            f"position={position.shape[0]}, yaw={yaw.shape[0]}"
        )
    return traj_name, img_paths, costmaps, position, yaw


# ──────────────────────────────────────────────────────────────────────────────
# Per-timestep 6-panel frame rendering
# ──────────────────────────────────────────────────────────────────────────────
def render_frame(rgb, costmap, pred_wp, gt_wp, timestep_losses, t, use_percentile=True):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("white")

    # Panel 1: RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB - Step {t}")
    axes[0, 0].axis("off")

    # Panel 2: grayscale costmap
    axes[0, 1].imshow(costmap, cmap="gray")
    axes[0, 1].set_title(f"Costmap (grayscale) - Step {t}")
    axes[0, 1].axis("off")

   # 1. Clear out NaNs and infs if any exist
    clean_costmap = np.nan_to_num(costmap, nan=0.0, posinf=0.0, neginf=0.0)
    valid = clean_costmap[clean_costmap > 0] # focus only on positive costs

    if valid.size > 0:
        # 2. Aggressively clip the ceiling (e.g., 75th percentile). 
        # This forces huge sky/building costs to saturate completely to max color,
        # leaving the full visual spectrum for the fine ground details.
        vmin = np.percentile(valid, 1)
        vmax = np.percentile(valid, 75) # Adjust to 60 or 80 if ground is still too dark
    else:
        vmin, vmax = 0.0, 1.0

    if vmin == vmax:
        vmax += 1e-5

    # 3. Render using 'inferno' or 'viridis'. 
    # Ground nuances will now pop, while sky/roofs will be a solid bright highlight.
    im = axes[0, 2].imshow(clean_costmap, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"Costmap (heatmap) - Step {t}")

    # Panel 4: predicted waypoint visualization (matches notebook's plot_traj exactly)
    plot_predicted_only(axes[1, 0], pred_wp)
    axes[1, 0].set_title(f"Predicted waypoints - Step {t}")

    # Panel 5: loss graph so far
    steps_so_far = np.arange(len(timestep_losses))
    axes[1, 1].plot(steps_so_far, timestep_losses, color="tab:blue", marker=".")
    if len(timestep_losses) > 0 and not np.isnan(timestep_losses[-1]):
        axes[1, 1].plot(t, timestep_losses[-1], "ro", markersize=8)
    axes[1, 1].set_title("L2 loss per timestep")
    axes[1, 1].set_xlabel("timestep")
    axes[1, 1].set_ylabel("L2 loss")
    axes[1, 1].grid(True, alpha=0.3)

    # Panel 6: predicted vs GT overlay
    plot_pred_vs_gt(axes[1, 2], pred_wp, gt_wp)
    axes[1, 2].set_title(f"Predicted vs GT - Step {t}")

    plt.tight_layout()
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    else:
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Per-trajectory evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_trajectory(controller, traj_dir: Path, out_root: Path, waypoint_stride: int, fps: int):
    traj_name, img_paths, costmaps, position, yaw = load_trajectory(traj_dir)
    T = len(img_paths)
    len_traj_pred = controller.config["len_traj_pred"]

    out_dir = out_root / traj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    controller.reset_params()

    timestep_losses = []
    per_waypoint_losses_all = []
    frames = []

    for t in range(T):
        bgr = cv2.imread(str(img_paths[t]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        costmap = costmaps[t]

        # Controller called directly on the precomputed costmap -- mast3r is never invoked.
        controller.predict(rgb, costmap)
        pred_wp = controller.action_pred.copy()  # (len_traj_pred, 4)

        # Unscale the normalized x and y coordinates back into meters
        pred_wp[:, :2] = pred_wp[:, :2] * 2.0
        #don't use if you are using the controller with the waypoint_stride argument
        '''
        if t == 0:
            print(f"\n[DEBUG] Raw Pred (first 2 waypoints): \n{pred_wp[:2, :2]}")
            gt_wp_debug, _ = gt_waypoints_for_frame(position, yaw, t, len_traj_pred, waypoint_stride)
            print(f"[DEBUG] Raw GT (first 2 waypoints): \n{gt_wp_debug[:2, :2]}\n")
        '''
        
        gt_wp, valid_k = gt_waypoints_for_frame(position, yaw, t, len_traj_pred, waypoint_stride)
        pred_wp_matched = pred_wp[valid_k] if len(valid_k) > 0 else pred_wp[:0]

        per_wp_loss, mean_loss = l2_loss(pred_wp_matched, gt_wp)
        timestep_losses.append(mean_loss)
        per_waypoint_losses_all.append(per_wp_loss.tolist())

        frame = render_frame(rgb, costmap, pred_wp, gt_wp, timestep_losses, t)
        frames.append(frame)
        print(f"[{traj_name}] step {t+1}/{T}  loss={mean_loss:.4f}" if not np.isnan(mean_loss)
              else f"[{traj_name}] step {t+1}/{T}  loss=n/a (no GT horizon left)")

    valid_losses = [l for l in timestep_losses if not np.isnan(l)]
    total_loss_sum = float(np.sum(valid_losses)) if valid_losses else float("nan")
    total_loss_mean = float(np.mean(valid_losses)) if valid_losses else float("nan")

    # ── loss.json ────────────────────────────────────────────────────────────
    loss_record = {
        "traj_name": traj_name,
        "num_timesteps": T,
        "waypoint_stride": waypoint_stride,
        "per_timestep_loss": timestep_losses,
        "per_timestep_per_waypoint_loss": per_waypoint_losses_all,
        "total_loss_sum": total_loss_sum,
        "total_loss_mean": total_loss_mean,
    }
    with open(out_dir / "loss.json", "w") as f:
        json.dump(loss_record, f, indent=2)

    # ── standalone loss curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(T), timestep_losses, marker=".", color="tab:blue")
    ax.set_xlabel("timestep")
    ax.set_ylabel("L2 loss")
    ax.set_title(f"{traj_name} -- per-timestep L2 loss (total mean={total_loss_mean:.4f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # ── mp4 ──────────────────────────────────────────────────────────────────
    h, w = frames[0].shape[:2]
    h, w = (h // 2) * 2, (w // 2) * 2  # even dims for libx264
    video_path = out_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame[:h, :w], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

    print(f"[{traj_name}] done. total_loss_sum={total_loss_sum:.4f} "
          f"total_loss_mean={total_loss_mean:.4f} -> {out_dir}")
    return loss_record


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Open-loop eval of ObjRelLearntController directly on "
                    "convert_to_mast3r_nav_format.py output (no mast3r call)."
    )
    parser.add_argument("--dataset_root", required=True,
                         help="Either a single trajectory dir, or a dataset dir "
                              "containing multiple <traj_name>/ subdirs.")
    parser.add_argument("--config_path", default="configs/controller",
                         help="Hydra config dir (relative), same as the notebook.")
    parser.add_argument("--config_name", default="carla_waypixel",
                         help="Hydra config name, same as the notebook.")
    parser.add_argument("--output_root", default="evaluation",
                         help="Root folder to write evaluation/<traj_name>/ into.")
    parser.add_argument("--waypoint_stride", type=int, default=1,
                         help="GT-frame spacing per predicted horizon step.")
    parser.add_argument("--fps", type=int, default=4, help="Output video FPS.")
    args = parser.parse_args()

    from hydra import initialize, compose
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path=args.config_path):
        config = dict(compose(config_name=args.config_name))
    print(OmegaConf.to_yaml(config))

    config["boost_final_goal"] = config.get("boost_final_goal", False)
    controller = ObjRelLearntController(config=config, boost_final_goal=config["boost_final_goal"])

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    traj_dirs = discover_trajectories(dataset_root)
    print(f"Found {len(traj_dirs)} trajectory dir(s) under {dataset_root}")

    all_records = []
    for traj_dir in traj_dirs:
        record = evaluate_trajectory(controller, traj_dir, out_root, args.waypoint_stride, args.fps)
        all_records.append(record)

    summary = {
        "num_trajectories": len(all_records),
        "per_trajectory_total_loss_mean": {r["traj_name"]: r["total_loss_mean"] for r in all_records},
        "overall_mean_loss": float(np.mean([r["total_loss_mean"] for r in all_records
                                             if not np.isnan(r["total_loss_mean"])])),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll done. Summary written to {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()