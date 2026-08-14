# Full Script Reference

Detailed reference for every script in the pipeline: what it does, every
argument it accepts, and the manual step-by-step sequence if you don't want
to use the two orchestrators (`generate_mass_dataset.py`, `postprocess.py`).

## Contents

- [Manual execution order](#manual-execution-order)
- [`generate_mass_dataset.py`](#generate_mass_datasetpy)
- [`world_gen/generate_world.py`](#world_gengenerate_worldpy)
- [`world_gen/generate_navmesh.py`](#world_gengenerate_navmeshpy)
- [`data_gen/generate_dataset.py`](#data_gengenerate_datasetpy)
- [`data_gen/generate_3d_points.py`](#data_gengenerate_3d_pointspy)
- [`data_gen/generate_gt_costmap.py`](#data_gengenerate_gt_costmappy)
- [`postprocess.py`](#postprocesspy)
- [`format_conversion/convert_to_mast3r_nav_format.py`](#format_conversionconvert_to_mast3r_nav_formatpy)
- [`format_conversion/convert_to_vint_format.py`](#format_conversionconvert_to_vint_formatpy)
- [`format_conversion/fix_numpy_pickle_compat.py`](#format_conversionfix_numpy_pickle_compatpy)
- [`format_conversion/pack_costmaps_h5.py`](#format_conversionpack_costmaps_h5py)
- [`eval_open_loop.py`](#eval_open_looppy)

---

## Manual execution order

If you skip the orchestrators, run scripts in this order:

1. **Start CARLA** (v0.10.0, Unreal Engine 5 build).
2. `python world_gen/generate_world.py [...]` — spawns traffic + pedestrians. Requires CARLA.
3. `python data_gen/generate_dataset.py [...]` — drives the ego and captures RGB/depth. Requires CARLA.
4. `python data_gen/generate_3d_points.py [...]` — back-projects depth to 3D. Offline, no CARLA needed.
5. `python data_gen/generate_gt_costmap.py [...]` — computes cost-to-goal maps. Requires CARLA **only** for `--cost geodesic`.
6. Pick one training format:
   - `python format_conversion/convert_to_mast3r_nav_format.py [...]` for mast3r-nav, **or**
   - `python format_conversion/convert_to_vint_format.py [...]` for ViNT/GNM.
7. `python format_conversion/fix_numpy_pickle_compat.py [...]` — only if your dataset was written under NumPy 2.x but your training environment uses NumPy 1.x.
8. `python format_conversion/pack_costmaps_h5.py [...]` — only needed if your training config expects a consolidated HDF5 costmap file (the mast3r-nav config in `configs/` does).

---

## `generate_mass_dataset.py`

Top-level orchestrator. For every world index `w` in `range(num_worlds)` and
trajectory index `t` in `range(trajectories_per_world)`, it runs, in order:
`world_gen/generate_world.py` (once per world) → the collection script →
`data_gen/generate_3d_points.py` → `data_gen/generate_gt_costmap.py`. If a
step fails, that scene is recorded as failed and the loop moves on; if world
generation itself fails, every trajectory planned for that world is skipped.

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | CARLA server address |
| `--port` | `2000` | CARLA server port |
| `--out` | `out` | Root output directory for all generated scenes |
| `--num_worlds` | `1` | Number of distinct static worlds to generate |
| `--trajectories_per_world` | `10` | Trajectories to collect per world |
| `--cost` | `groundplane` | `euclidean3d`, `groundplane`, or `geodesic` — forwarded to `generate_gt_costmap.py` |
| `--color` | off | Forwarded to `generate_gt_costmap.py` — also save RGB colorized costmaps |
| `--per_image_scale` | off | Forwarded — normalize each frame's costmap independently instead of by a global reference |
| `--lateral_penalty` | off | Forwarded — geodesic-only, penalizes off-road lateral offset |
| `--lateral_penalty_weight` | `1.0` | Forwarded — multiplier on the lateral penalty |
| `--scale_percentile` | `95.0` | Forwarded — percentile used as the normalization reference instead of the true max |
| `--gamma` | `0.6` | Forwarded — gamma curve applied after normalization (<1.0 spreads low-cost contrast) |
| `--world_seed_base` | `1000` | World seed = `world_seed_base + world_index` |
| `--traj_seed_base` | `0` | Trajectory seed = `traj_seed_base + world_index * trajectories_per_world + traj_index` |

Seeds are deterministic: re-running with the same base seeds and `--num_worlds`/`--trajectories_per_world` reproduces the same worlds and routes.

Seed lifecycle note: `geodesic` costs need CARLA running through the costmap stage (it queries the route planner per pixel via a multiprocessing pool of 20 workers). `euclidean3d` and `groundplane` are pure geometry — safe to close CARLA right after the collection step finishes.

---

## `world_gen/generate_world.py`

Spawns static traffic vehicles and pedestrians into an already-running CARLA
world. Destroys any existing `vehicle.*`/`walker.*` actors first so repeated
calls (e.g. from the orchestrator) don't accumulate actors across worlds —
this will also destroy an ego vehicle if one happens to already be spawned,
so only run this before the collection script, not after.

Traffic Manager is intentionally never engaged — enabling it would make the
world dynamic and break the deterministic seeding this pipeline relies on.

| Flag | Default | Description |
|---|---|---|
| `--host` | `localhost` | CARLA server address |
| `--port` | `2000` | CARLA server port |
| `--seed` | `42` | Controls vehicle/pedestrian spawn point shuffling |
| `--walker_seed` | `42` | Passed to `world.set_pedestrians_seed()` if available in your CARLA build; falls back to a warning otherwise |
| `--num_vehicles` | `20` | Number of traffic vehicles to spawn |
| `--num_pedestrians` | `50` | Number of pedestrians to spawn |

Known TODOs left in the source: pedestrian-seed support isn't confirmed for
this CARLA build, ego collision-avoidance/sidestepping isn't implemented, and
there's no way yet to restrict spawning to a sub-region of the map.

---

## `world_gen/generate_navmesh.py`

Standalone utility built during early ideation for this project — **not
called by either orchestrator and not consumed by any other script in this
repo.** Samples a 400×400 m grid (0.5 m resolution by default, hardcoded in
the script) via `world.cast_ray()`, keeps points whose semantic label is
road/sidewalk/crosswalk/parking/shoulder, discards points too close to
non-drivable geometry or to spawned vehicles/pedestrians, and writes the
survivors to `navmesh.npy` (`(N, 6)`: xyz + normal) in the current directory.

It predates the road-graph random-walk approach that `generate_dataset.py`
ended up using to pick the ego's route, and was never plugged into the
pipeline. It's left in the repo because the underlying idea is still
plausible: a filtered grid of drivable-surface points like this could be the
basis for an alternative ego-travel method — e.g. sampling or planning routes
directly over the navmesh points instead of walking the CARLA road graph —
if you want to explore that instead of (or alongside) the current approach.

| Flag | Default | Description |
|---|---|---|
| `--visualize` | off | Draw the resulting points as green debug markers in the CARLA viewport for 60 seconds |

There's no `--host`/`--port` flag — it always connects to `localhost:2000`
with a 20 s timeout. Grid bounds, resolution, and clearance radii are set as
constants at the top of the file if you need to tweak them.

---

## `data_gen/generate_dataset.py`

Drives an ego vehicle (`vehicle.lincoln.mkz`) through one continuous route
and captures synchronized RGB, metric depth, and pose at 10 Hz
(`fixed_delta_seconds = 0.1`, synchronous mode). The route is built by
picking `NUM_ANCHORS` (1, so 2 stops total including the start) waypoints via
a random walk of the road graph — taking a random branch at every junction —
spaced 150–200 m apart by road distance, then connecting the start and
anchors with CARLA's `GlobalRoutePlanner` so the dense path follows real road
geometry through turns and junctions. Frames where a collision fires, or
where a sensor read times out, are skipped rather than saved.

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | CARLA server address |
| `--port` | `2000` | CARLA server port |
| `--out` | `out` | Root output directory |
| `--scene` | `scene_00` | Scene folder name, e.g. `world00_traj00` |
| `--seed` | `42` | RNG seed for spawn point and route branch selection |

Writes to `out/<scene>/`: `images/*.png` (320×240 RGB), `images_depth/*.npy`
(metric depth in meters, decoded from CARLA's raw depth encoding),
`trajectory.npy` (anchor waypoints), `agent_states.npy` (x, y, z, roll,
pitch, yaw per saved frame), `camera_intrinsics.npy`, and
`camera_extrinsics.npy` (camera-to-world per saved frame). Camera is mounted
1.5 m forward, 2.4 m up, no rotation offset, 90° FOV.

> Contains a hardcoded `sys.path.insert(...)` pointing at the original
> author's CARLA install — update it to your own path, or delete the line if
> `carla` is already importable in your environment.

---

## `data_gen/generate_3d_points.py`

Offline, no CARLA connection needed. Loads each `images_depth/*.npy` and
`camera_intrinsics.npy`, back-projects every pixel through the pinhole model,
and writes `3d_points/*.npy` — `(H, W, 3)` float32, camera-frame coordinates
where X = right, Y = down, Z = forward.

| Flag | Default | Description |
|---|---|---|
| `--out` | `out` | Root output directory |
| `--scene` | `scene_00` | Scene folder to process |

---

## `data_gen/generate_gt_costmap.py`

For every frame, transforms the saved 3D camera-frame points to world
coordinates using `camera_extrinsics.npy`, computes a per-pixel scalar cost
to a goal, and writes both a raw float32 cost map and an 8-bit grayscale
visualization (optionally also an RGB colorized one). The goal is always
`agent_states[-1, :3]` — the ego's own final recorded position — **not** the
last anchor in `trajectory.npy`, so the minimum cost always lands in the last
saved frame by construction.

Three cost types:

| `--cost` value | Meaning |
|---|---|
| `euclidean3d` | Full 3D Euclidean distance to goal |
| `groundplane` | 2D ground-plane distance to goal (recommended for driving) |
| `geodesic` | Road-following distance via `GlobalRoutePlanner`, computed per-pixel on a 20-worker multiprocessing pool with each worker holding its own CARLA client connection |

| Flag | Default | Description |
|---|---|---|
| `--out` | `out` | Root output directory |
| `--scene` | `scene_00` | Scene folder to process |
| `--cost` | `euclidean3d` | `euclidean3d`, `groundplane`, or `geodesic` |
| `--host` | `127.0.0.1` | CARLA server address (geodesic only) |
| `--port` | `2000` | CARLA server port (geodesic only) |
| `--color` | off | Also save RGB colorized costmaps to `costmaps_color/` |
| `--per_image_scale` | off | Normalize each frame's costmap by its own percentile reference instead of a global one |
| `--lateral_penalty` | off | Geodesic-only. Adds a penalty proportional to the pixel's lateral offset from the nearest drivable lane center, so sidewalks/off-road cost more than the raw route distance alone would suggest |
| `--lateral_penalty_weight` | `1.0` | Multiplier on the lateral penalty term |
| `--scale_percentile` | `95.0` | Percentile (not true max) used as the normalization reference, so a few extreme-outlier pixels don't wash out contrast |
| `--gamma` | `0.6` | Gamma curve applied after normalization; values below 1.0 expand contrast in the near-field/low-cost range |

Pixels with no valid depth (sky) are marked `-1.0` in the raw map and pure
black in the visualizations. Writes to `out/<scene>/`: `costmaps/*.png`,
`costmaps_raw/*.npy`, and (with `--color`) `costmaps_color/*.png`. Running
this script again for a scene overwrites all three directories — it does not
version by cost type, so `costmaps_raw/` always holds whichever `--cost` was
used most recently for that scene.

> Also contains the same hardcoded CARLA `sys.path.insert(...)` as
> `generate_dataset.py` — same fix applies.

---

## `postprocess.py`

Orchestrates the mast3r-nav conversion path only (not the ViNT/GNM path —
see [`convert_to_vint_format.py`](#format_conversionconvert_to_vint_formatpy)
if that's what you need). Always runs all three steps in order and aborts on
the first failure:

1. `format_conversion/convert_to_mast3r_nav_format.py`
2. `format_conversion/fix_numpy_pickle_compat.py`
3. `format_conversion/pack_costmaps_h5.py`

Because `--dataset_name` is required at the top level, step 3 always runs —
there's no flag to skip HDF5 packing when going through this orchestrator.

Scenes that are missing required files (e.g. an incomplete trajectory that
never got costmaps) are skipped with a `[WARN]` rather than aborting the
whole run — `convert_to_mast3r_nav_format.py` catches `FileNotFoundError`/
`ValueError` per scene internally. This also means `postprocess.py` (and the
individual `format_conversion/` scripts) can be pointed at a **custom
dataset** that wasn't produced by `generate_mass_dataset.py` at all, as long
as each scene directory follows the same layout: `images/`,
`agent_states.npy`, `camera_extrinsics.npy`, and `costmaps_raw/*.npy`.

| Flag | Default | Description |
|---|---|---|
| `--out` | `out` | Root directory to search for raw scenes (used by step 3) |
| `--dataset_root` | *(required)* | Output directory for the converted dataset |
| `--scene_root` | — | Mutually exclusive with `--scenes_glob`. Convert one scene folder |
| `--scenes_glob` | — | Mutually exclusive with `--scene_root`. Glob pattern matching multiple scene folders, e.g. `"out/world00_*"` |
| `--traj_name` | none | Override the output trajectory folder name (single-scene mode only) |
| `--cost` | `groundplane` | Bookkeeping label only — see the note in step 1 below |
| `--no_image_copy` | off | Skip copying RGB frames into the converted dataset |
| `--dry_run` | off | Step 2 only — report what the pickle fix would change without writing anything |
| `--no_backup` | off | Step 2 only — skip writing `.pkl.bak` safety copies before rewriting pickles |
| `--dataset_name` | *(required)* | Step 3 — name used to build the default `<dataset_name>_costmaps.h5` filename |
| `--graphs_path` | none | Step 3 — directory to write the packed HDF5 file into |
| `--output_path` | none | Step 3 — explicit output path, overrides `--graphs_path`/`--dataset_name` naming |
| `--target_h` | `60` | Step 3 — downsampled costmap height |
| `--target_w` | `80` | Step 3 — downsampled costmap width |
| `--overwrite` | off | Step 3 — overwrite an existing HDF5 file |

---

## `format_conversion/convert_to_mast3r_nav_format.py`

Converts one or more raw scenes into the format mast3r-nav training expects:
RGB frames stay the visual input, and costmaps are stored as consolidated
arrays rather than one grayscale image per frame.

Running this script alone writes directly into `--dataset_root`:

```
<dataset_root>/
├── compiled_costmaps.h5                (total_T, H, W) float32, all trajectories concatenated
├── world00_traj00/
│   ├── images/                         0.jpg … T.jpg, renumbered RGB frames
│   ├── traj_data.pkl                   {"position": (T, 2) xy, "yaw": (T, 1)}
│   └── costmap_world00_traj00.npz      (T, H, W) raw metric costs for this trajectory
├── world00_traj01/
│   └── ...                             same layout as world00_traj00/
└── ...                                 one folder per converted trajectory
```

When this script is run as part of `postprocess.py`, `fix_numpy_pickle_compat.py`
adds a `traj_data.pkl.bak` next to each `traj_data.pkl` (unless `--no_backup`
is passed), and `pack_costmaps_h5.py` writes a separate
`<graphs_path>/<dataset_name>_costmaps.h5` alongside — see
[mast3r-nav conversion output structure](README.md#mast3r-nav-conversion-output-structure)
in the main README for the combined picture.

Costmaps are always pulled from `costmaps_raw/*.npy` (raw metric floats),
never from the `costmaps/*.png` visualizations. Since `costmaps_raw/` isn't
split by cost type, `--cost` here is a bookkeeping label only — it records
what was used, it does not let you select between multiple precomputed cost
types for the same scene.

Position/yaw in `traj_data.pkl` are first converted from CARLA to ROS
convention (`Y_ros = -Y_carla`, `yaw_ros = -yaw_carla`), then made
egocentric to the trajectory's own start pose, so `position[0] ≈ (0, 0)` and
`yaw[0] ≈ 0`.

Frame order in `compiled_costmaps.h5` follows the order scene roots are
processed in (sorted, for `--scenes_glob`) — there's currently no per-frame
trajectory-id stored alongside it.

| Flag | Default | Description |
|---|---|---|
| `--scene_root` | — | Mutually exclusive with `--scenes_glob`. Path to a single `out/<scene_name>` directory |
| `--scenes_glob` | — | Mutually exclusive with `--scene_root`. Glob pattern matching multiple scene directories |
| `--dataset_root` | *(required)* | Output root for the converted dataset |
| `--traj_name` | none | Trajectory folder name override (single-scene mode only) |
| `--cost` | `groundplane` | Bookkeeping label only (see above) |
| `--no_image_copy` | off | Skip copying RGB frames; costmap npz/h5 and `traj_data.pkl` are still written |
| `--h5_path` | `<dataset_root>/compiled_costmaps.h5` | Explicit path override for the dataset-wide HDF5 file |

---

## `format_conversion/convert_to_vint_format.py`

Alternate output format for training [GNM / ViNT / NoMaD](https://github.com/robodhruv/visualnav-transformer)
instead of mast3r-nav. Per that repo's expected layout:

```
<dataset_root>/
    <traj_name>/
        0.jpg … T.jpg
        traj_data.pkl      # {"position": (T,2) xy, "yaw": (T,)}
```

Important difference from the mast3r-nav path: here the **cost map image
replaces the RGB image** as the network's visual input, saved as
single-channel grayscale JPEGs. Position/yaw come directly from
`agent_states.npy` (x, y, yaw). This script is not called by `postprocess.py`
— run it directly if you want ViNT/GNM output.

| Flag | Default | Description |
|---|---|---|
| `--scene_root` | — | Mutually exclusive with `--scenes_glob`. Path to a single scene directory |
| `--scenes_glob` | — | Mutually exclusive with `--scene_root`. Glob pattern matching multiple scenes |
| `--dataset_root` | *(required)* | Output root for the ViNT-format dataset |
| `--traj_name` | none | Trajectory folder name override (single-scene mode only) |
| `--cost` | `groundplane` | Which `costmaps/` subfolder to pull the grayscale images from |
| `--no_rgb_copy` | off | Skip copying the original RGB frames into an `rgb/` subfolder |

---

## `format_conversion/fix_numpy_pickle_compat.py`

NumPy 2.0 moved internal modules from `numpy.core` to `numpy._core`, so a
`traj_data.pkl` written under NumPy 2.x fails with
`ModuleNotFoundError: No module named 'numpy._core'` when loaded under NumPy
1.x. This script must be run **in the NumPy 2.x environment** (the one that
produced the pickle) — it loads each pickle, converts every array to a plain
Python list and back, and rewrites it, producing a pickle that loads cleanly
under both NumPy versions.

| Flag | Default | Description |
|---|---|---|
| `--dataset_root` | — | Mutually exclusive with `--traj_dir`. Convert every `traj_data.pkl` under this dataset root |
| `--traj_dir` | — | Mutually exclusive with `--dataset_root`. Convert a single trajectory folder |
| `--dry_run` | off | Report what would be converted without writing anything |
| `--no_backup` | off | Skip writing a `.pkl.bak` safety copy before rewriting |

---

## `format_conversion/pack_costmaps_h5.py`

Packs per-frame raw costmaps (`costmaps_raw/*.npy`, metric distances in
meters — **not** the `costmaps/*.png` visualizations) into a single HDF5 file
matching the schema a mast3r-nav training config expects at
`graphs_path/<dataset_name>_costmaps.h5`:

```
<h5file>
    <traj_name>_0/
        pls_pixels   (target_h, target_w) float32   # raw metric cost
    <traj_name>_1/
        pls_pixels   (target_h, target_w) float32
    ...
```

Top-level keys are `{traj_name}_{frame_idx}` (0-indexed, not zero-padded).
`traj_name` here must exactly match the trajectory folder name used in the
converted dataset, since the training loader looks up costmap keys by that
name.

| Flag | Default | Description |
|---|---|---|
| `--out` | `out` | Root directory containing raw scene folders |
| `--scenes` | — | Mutually exclusive with `--scenes_glob`. Explicit list of scene names, e.g. `--scenes world00_traj00 world00_traj01` |
| `--scenes_glob` | — | Mutually exclusive with `--scenes`. Glob pattern for scene directories, e.g. `"out/*"` |
| `--dataset_name` | *(required)* | Used to build the default output filename `<dataset_name>_costmaps.h5` |
| `--graphs_path` | none | Directory to write the packed file into (matches the training config's `graphs_path`) |
| `--output_path` | none | Explicit output path, overrides `--graphs_path`/`--dataset_name` naming |
| `--target_h` | `60` | Output costmap height |
| `--target_w` | `80` | Output costmap width |
| `--overwrite` | off | Overwrite the output file if it already exists |

---

## `eval_open_loop.py`

Open-loop evaluation of a trained mast3r-nav `ObjRelLearntController` against
trajectories produced by `convert_to_mast3r_nav_format.py`. "Open loop" means
the real recorded RGB/costmap is fed to the controller at every timestep —
the model's own predicted actions never feed back into what it sees next.

For every trajectory directory it finds, the script:

1. Calls `controller.predict(rgb, costmap)` at every timestep `t`.
2. Re-expresses the ground-truth trajectory in the frame egocentric to `t`
   (same rotation math the converter used to anchor at frame 0, just
   re-anchored at `t`), so it's directly comparable to the model's
   egocentric-to-current-frame prediction.
3. Matches predicted waypoint `k` to GT frame `t + (k+1) * waypoint_stride`
   and computes L2 distance between predicted and GT `(dx, dy)`.
4. Writes `evaluation/<traj_name>/{loss.json, loss_curve.png, video.mp4}`.

L2 loss is computed on position only (not heading); per-timestep loss is the
mean L2 over the valid prediction horizon, and total trajectory loss is
reported as both the sum and mean over timesteps in `loss.json`.

This script imports `libs.control.learnt_controller` and
`notebooks.viz_utils`, which live in the **mast3r-nav repository, not this
one** — run it from the mast3r-nav repo root (or otherwise put that repo on
your `PYTHONPATH`), with the `configs/` YAML files in this repo pointed at
your own checkpoint and dataset paths.

| Flag | Default | Description |
|---|---|---|
| `--dataset_root` | *(required)* | Root of the converted mast3r-nav-format dataset to evaluate against |
| `--config_path` | `configs/controller` | Hydra config directory |
| `--config_name` | `carla_waypixel` | Config name to load |
| `--output_root` | `evaluation` | Where to write per-trajectory results |
| `--waypoint_stride` | `1` | GT-frame spacing used to match each predicted waypoint — change this if your model was trained with a different horizon spacing |
| `--fps` | `4` | Output video FPS |

`configs/carla_waypixel.yaml` and `configs/carla_learnt.yaml` in this repo
are the controller configs referenced above (`--config_path` defaults to
`configs/controller`, matching where they need to live). They belong inside
`configs/controller/` of the [mast3r-nav](https://github.com/vanshg1729/mast3r-nav)
repository checkout — copy them there — and both still contain the original
author's machine paths (`load_run`, `graphs_path`) that need to be edited to
point at your own checkpoint directory and packed-costmap HDF5 location
before use. See [`TRAINING.md`](TRAINING.md) for the full setup sequence.
