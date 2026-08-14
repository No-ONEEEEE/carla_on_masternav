"""
Post-Processing Orchestrator
============================
Sequentially calls format conversion scripts in the required logical order:
    1. convert_to_mast3r_nav_format.py
    2. fix_numpy_pickle_compat.py
    3. pack_costmaps_h5.py

It exposes all internal argument parameters with identical default behaviors.
"""

import argparse
import subprocess
import sys
import os


def run_script(script_name: str, args_list: list[str]) -> bool:
    cmd = [sys.executable, os.path.join("format_conversion", script_name)] + args_list
    print(f"\n{'='*80}\n[POST-PROCESS] Running: {' '.join(cmd)}\n{'='*80}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified Orchestrator for CARLA dataset post-processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ──── Global / Shared Settings ────
    parser.add_argument("--out", default="out", help="Pipeline raw data directory (used to search for raw scenes).")
    parser.add_argument("--dataset_root", required=True, help="Target destination for output training structure format.")

    # ──── 1. convert_to_mast3r_nav_format.py Arguments ────
    c_group = parser.add_mutually_exclusive_group(required=True)
    c_group.add_argument("--scene_root", default=None, help="Process one single source scene dir folder explicitly.")
    c_group.add_argument("--scenes_glob", default=None, help="Glob match query string pattern identifying raw scenes.")
    
    parser.add_argument("--traj_name", default=None, help="Specific track overrides directory tag (single-mode only).")
    parser.add_argument("--cost", default="groundplane", choices=["euclidean3d", "groundplane", "geodesic"], help="Bookkeeping text descriptor parameter tracking context.")
    parser.add_argument("--no_image_copy", action="store_true", help="Omit file copies of captured PNG arrays into output.")

    # ──── 2. fix_numpy_pickle_compat.py Arguments ────
    parser.add_argument("--dry_run", action="store_true", help="Simulate execution run cycle tracking transformations.")
    parser.add_argument("--no_backup", action="store_true", help="Bypass creating '.pkl.bak' safeguard copies.")

    # ──── 3. pack_costmaps_h5.py Arguments ────
    parser.add_argument("--dataset_name", required=True, help="Training model schema tracking reference string tag ID.")
    parser.add_argument("--graphs_path", default=None, help="Destination directory layout to house generated global .h5.")
    parser.add_argument("--output_path", default=None, help="Explicit file target override path specifying final .h5 location.")
    parser.add_argument("--target_h", type=int, default=60, help="Downsampled cost dimension height resolution.")
    parser.add_argument("--target_w", type=int, default=80, help="Downsampled cost dimension width resolution.")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite pre-existing h5 tracking artifacts.")

    parsed_args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Run convert_to_mast3r_nav_format.py
    # ──────────────────────────────────────────────────────────────────────────
    step1_args = ["--dataset_root", parsed_args.dataset_root, "--cost", parsed_args.cost]
    if parsed_args.scene_root:
        step1_args += ["--scene_root", parsed_args.scene_root]
    if parsed_args.scenes_glob:
        step1_args += ["--scenes_glob", parsed_args.scenes_glob]
    if parsed_args.traj_name:
        step1_args += ["--traj_name", parsed_args.traj_name]
    if parsed_args.no_image_copy:
        step1_args += ["--no_image_copy"]

    if not run_script("convert_to_mast3r_nav_format.py", step1_args):
        print("[CRITICAL] Step 1 (Format Conversion) failed. Aborting pipeline.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Run fix_numpy_pickle_compat.py
    # ──────────────────────────────────────────────────────────────────────────
    step2_args = ["--dataset_root", parsed_args.dataset_root]
    if parsed_args.dry_run:
        step2_args += ["--dry_run"]
    if parsed_args.no_backup:
        step2_args += ["--no_backup"]

    if not run_script("fix_numpy_pickle_compat.py", step2_args):
        print("[CRITICAL] Step 2 (Pickle Compatibility Fix) failed. Aborting pipeline.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Run pack_costmaps_h5.py
    # ──────────────────────────────────────────────────────────────────────────
    step3_args = ["--out", parsed_args.out, "--dataset_name", parsed_args.dataset_name,
                  "--target_h", str(parsed_args.target_h), "--target_w", str(parsed_args.target_w)]
    
    if parsed_args.scene_root:
        step3_args += ["--scenes", os.path.basename(os.path.normpath(parsed_args.scene_root))]
    elif parsed_args.scenes_glob:
        step3_args += ["--scenes_glob", parsed_args.scenes_glob]
        
    if parsed_args.graphs_path:
        step3_args += ["--graphs_path", parsed_args.graphs_path]
    if parsed_args.output_path:
        step3_args += ["--output_path", parsed_args.output_path]
    if parsed_args.overwrite:
        step3_args += ["--overwrite"]

    if not run_script("pack_costmaps_h5.py", step3_args):
        print("[CRITICAL] Step 3 (Costmap Packing) failed.")
        sys.exit(1)

    print("\n[SUCCESS] Entire data post-processing stack completed successfully.")


if __name__ == "__main__":
    main()
