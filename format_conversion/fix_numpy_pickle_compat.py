"""
NumPy Pickle Compatibility Converter
======================================
Converts traj_data.pkl files written with NumPy 2.x to be compatible with
NumPy 1.x by deserialising the arrays and rewriting them cleanly.

The issue: NumPy 2.0 moved internal modules from numpy.core to numpy._core.
Pickles serialised under NumPy 2.x contain references to numpy._core which
do not exist in NumPy 1.x, causing:
    ModuleNotFoundError: No module named 'numpy._core'

The fix: load each pkl (requires NumPy 2.x to be currently active), convert
every numpy array to a plain Python list and back to a numpy array, then
rewrite the pkl. The resulting file loads cleanly under both NumPy 1.x and 2.x.

This script must be run in the environment that has NumPy 2.x (e.g. your
'carla' env), immediately after convert_to_vint_format.py, before switching
to the training environment.

Usage:
    # Convert all traj_data.pkl files under a dataset root
    python fix_pickle_numpy_compat.py --dataset_root vint_dataset/my_dataset

    # Convert a single trajectory folder
    python fix_pickle_numpy_compat.py --traj_dir vint_dataset/my_dataset/world00_traj00

    # Dry run — report what would be converted without writing anything
    python fix_pickle_numpy_compat.py --dataset_root vint_dataset/my_dataset --dry_run
"""

import argparse
import os
import glob
import pickle
import shutil

import numpy as np


def convert_pkl(pkl_path: str, dry_run: bool = False, backup: bool = True) -> bool:
    """
    Load a traj_data.pkl, convert all numpy arrays to ensure NumPy 1.x
    compatibility, and overwrite the file.

    Parameters
    ----------
    pkl_path : path to traj_data.pkl
    dry_run  : if True, only report what would happen — do not write
    backup   : if True, save a .bak copy before overwriting

    Returns
    -------
    True if file was (or would be) converted, False if skipped.
    """
    if not os.path.isfile(pkl_path):
        print(f"[WARN] Not found, skipping: {pkl_path}")
        return False

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except ModuleNotFoundError as e:
        print(f"[ERROR] Cannot even load {pkl_path} in this environment: {e}")
        print(f"        Make sure you are running in the NumPy 2.x environment.")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error loading {pkl_path}: {e}")
        return False

    if not isinstance(data, dict):
        print(f"[WARN] {pkl_path} does not contain a dict — skipping.")
        return False

    # Rebuild dict: convert every numpy array via list round-trip so the
    # pickle references numpy's public API rather than numpy._core internals
    converted = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            converted[key] = np.array(value.tolist(), dtype=value.dtype)
        else:
            converted[key] = value

    if dry_run:
        keys_info = {k: (v.shape, str(v.dtype))
                     for k, v in converted.items()
                     if isinstance(v, np.ndarray)}
        print(f"[DRY RUN] Would convert: {pkl_path}  arrays={keys_info}")
        return True

    # Backup original before overwriting
    if backup:
        bak_path = pkl_path + ".bak"
        shutil.copy2(pkl_path, bak_path)

    with open(pkl_path, "wb") as f:
        pickle.dump(converted, f, protocol=2)  # protocol=2 — safe for Python 3 + NumPy 1.x

    print(f"[OK] Converted: {pkl_path}")
    return True


def main(args):
    # Collect all pkl files to process
    if args.traj_dir:
        pkl_files = [os.path.join(args.traj_dir, "traj_data.pkl")]
    elif args.dataset_root:
        pkl_files = sorted(
            glob.glob(os.path.join(args.dataset_root, "*", "traj_data.pkl"))
        )
    else:
        raise ValueError("Provide either --dataset_root or --traj_dir.")

    if not pkl_files:
        print("[INFO] No traj_data.pkl files found.")
        return

    print(f"[INFO] NumPy version in this environment: {np.__version__}")
    print(f"[INFO] Found {len(pkl_files)} traj_data.pkl file(s) to process.")
    if args.dry_run:
        print("[INFO] DRY RUN — no files will be written.\n")

    converted = 0
    failed    = 0

    for pkl_path in pkl_files:
        ok = convert_pkl(pkl_path,
                         dry_run=args.dry_run,
                         backup=not args.no_backup)
        if ok:
            converted += 1
        else:
            failed += 1

    print(f"\n[SUMMARY] {'Would convert' if args.dry_run else 'Converted'}: "
          f"{converted}  |  Failed/skipped: {failed}")

    if not args.dry_run and not args.no_backup:
        print(f"[INFO] Original files backed up as *.pkl.bak alongside each file.")
        print(f"       Delete backups once you've verified training works:\n"
              f"       find {args.dataset_root or args.traj_dir} -name '*.pkl.bak' -delete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert traj_data.pkl files from NumPy 2.x to NumPy 1.x "
                    "pickle format."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_root", default=None,
                       help="Root of the ViNT-format dataset. Converts all "
                            "traj_data.pkl files found one level deep.")
    group.add_argument("--traj_dir", default=None,
                       help="Single trajectory folder containing traj_data.pkl.")

    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be converted without writing "
                             "any files.")
    parser.add_argument("--no_backup", action="store_true",
                        help="Skip creating .pkl.bak backup files before "
                             "overwriting. Not recommended for first run.")

    main(parser.parse_args())
