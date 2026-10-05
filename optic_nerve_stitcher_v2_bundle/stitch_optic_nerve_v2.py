#!/usr/bin/env python3
"""
stitch_optic_nerve_v2.py

Stable translation-only stitcher for overlapping 100x optic-nerve TIFF tiles.
Designed for the Zeiss/AxoNet workflow used in this project.

Main safeguards
---------------
* Reads ZIP archives or directories of TIFF/TIF tiles.
* Rejects unreadable and near-blank tiles.
* Detects exact duplicate image content.
* Registers at reduced resolution with SIFT.
* Solves a global weighted translation-only mosaic.
* Refuses to continue if the overlap graph is disconnected.
* Writes the mosaic DIRECTLY as an uncompressed classic TIFF using
  tifffile.memmap (no BigTIFF, no slow final LZW compression step).
* Preserves native source pixels: overlap pixels come from the tile whose
  pixel lies farther from that tile's edge.
* Produces full-resolution JPEG, preview JPEG, tile-position CSV, QC CSV,
  and registration-edge CSV.
* Optional --axon-scale 0.542 output for AxoNet 2.0 scale matching.

Important assumptions
---------------------
This is intentionally translation-only. It does not rotate, warp, stretch,
or perspective-correct individual fields. That is desirable for this
microscopy workflow because it avoids geometric distortion of axons.

Always visually inspect the preview before using a mosaic for analysis.

Typical use
-----------
python stitch_optic_nerve_v2.py "R:/path/100x All.zip" --sid 2001_OS \
    --output-dir "R:/path/stitched"

Optional AxoNet-scale copy:
python stitch_optic_nerve_v2.py "R:/path/100x All.zip" --sid 2001_OS \
    --output-dir "R:/path/stitched" --axon-scale 0.542
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

TIFF_EXTENSIONS = {".tif", ".tiff"}
CLASSIC_TIFF_SAFE_BYTES = int(3.8 * 1024**3)  # conservative margin below 4 GiB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stitch overlapping 100x optic-nerve TIFF images."
    )
    p.add_argument("input", help="ZIP archive or folder containing TIFF tiles")
    p.add_argument("--sid", required=True, help='Sample ID, e.g. "2001_OS"')
    p.add_argument("--output-dir", default="stitched_output")
    p.add_argument("--registration-scale", type=float, default=0.15)
    p.add_argument("--ratio-test", type=float, default=0.72)
    p.add_argument("--min-good-matches", type=int, default=12)
    p.add_argument("--min-inliers", type=int, default=10)
    p.add_argument("--inlier-radius", type=float, default=3.0)
    p.add_argument("--sift-features", type=int, default=4000)
    p.add_argument("--preview-max", type=int, default=2600)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--keep-duplicates", action="store_true")
    p.add_argument(
        "--axon-scale",
        type=float,
        default=None,
        help="Optional extra downsample (for current AxoNet protocol use 0.542)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files for this SID to be overwritten",
    )
    return p.parse_args()


def collect_files(input_path: Path, work_dir: Path) -> list[Path]:
    if input_path.is_dir():
        root = input_path
    elif input_path.is_file() and input_path.suffix.lower() == ".zip":
        root = work_dir / "extracted"
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(root)
    else:
        raise ValueError("Input must be an existing ZIP archive or directory.")

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS
    )
    if not files:
        raise RuntimeError("No .tif/.tiff files were found.")
    return files


def qc_tiles(files: list[Path], keep_duplicates: bool):
    seen_hashes: dict[str, Path] = {}
    usable: list[Path] = []
    rows: list[tuple[str, str, str]] = []
    reference_shape = None

    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            rows.append((p.name, "unreadable", ""))
            continue

        if reference_shape is None:
            reference_shape = im.shape
        elif im.shape != reference_shape:
            rows.append((p.name, "shape_mismatch", str(im.shape)))
            continue

        std = float(im.std())
        if std < 1.0:
            rows.append((p.name, "blank", f"std={std:.6f}"))
            continue

        digest = hashlib.md5(im.tobytes()).hexdigest()
        if digest in seen_hashes and not keep_duplicates:
            rows.append(
                (p.name, "exact_duplicate", f"duplicate_of={seen_hashes[digest].name}")
            )
            continue

        seen_hashes.setdefault(digest, p)
        usable.append(p)
        rows.append((p.name, "usable", f"std={std:.6f}"))

    return usable, rows


def compute_features(files: list[Path], scale: float, nfeatures: int):
    if not (0.05 <= scale <= 1.0):
        raise ValueError("--registration-scale must be between 0.05 and 1.0")

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=0.012,
        edgeThreshold=10,
    )
    features = []
    shapes = []

    for idx, p in enumerate(files, 1):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        kp, des = sift.detectAndCompute(small, None)
        features.append((kp, des))
        shapes.append(small.shape)
        print(f"Features {idx}/{len(files)}: {p.name} ({len(kp)} keypoints)")

    return features, shapes


def build_translation_edges(
    features,
    shapes,
    ratio_test: float,
    min_good_matches: int,
    min_inliers: int,
    inlier_radius: float,
):
    """
    Feature relation: p_j ~= p_i + t
    Tile-origin relation: O_j - O_i ~= -t
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    edges = []

    for i, j in itertools.combinations(range(len(features)), 2):
        kp1, d1 = features[i]
        kp2, d2 = features[j]
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
            continue

        knn = matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in knn if m.distance < ratio_test * n.distance]
        if len(good) < min_good_matches:
            continue

        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        displacement = p2 - p1
        median_disp = np.median(displacement, axis=0)
        error = np.linalg.norm(displacement - median_disp, axis=1)
        inlier_mask = error < inlier_radius
        n_inliers = int(inlier_mask.sum())
        if n_inliers < min_inliers:
            continue

        robust_t = np.median(displacement[inlier_mask], axis=0)
        delta = -robust_t
        h, w = shapes[i]

        # Genuine overlapping neighbors should be less than about one field
        # apart in both axes. This filters many false long-range matches.
        if abs(delta[0]) > 0.97 * w or abs(delta[1]) > 0.97 * h:
            continue

        edges.append(
            {
                "i": i,
                "j": j,
                "dx": float(delta[0]),
                "dy": float(delta[1]),
                "inliers": n_inliers,
                "pair_residual": float(np.median(error[inlier_mask])),
            }
        )

    return edges


def graph_component(n: int, edges) -> set[int]:
    adjacency = {i: set() for i in range(n)}
    for e in edges:
        adjacency[e["i"]].add(e["j"])
        adjacency[e["j"]].add(e["i"])

    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def solve_positions(n: int, edges):
    m = len(edges)
    A = np.zeros((m + 1, n), dtype=np.float64)
    bx = np.zeros(m + 1, dtype=np.float64)
    by = np.zeros(m + 1, dtype=np.float64)
    weights = np.ones(m + 1, dtype=np.float64)

    for k, e in enumerate(edges):
        i, j = e["i"], e["j"]
        A[k, i] = -1.0
        A[k, j] = 1.0
        bx[k] = e["dx"]
        by[k] = e["dy"]
        weights[k] = max(e["inliers"], 1)

    # Anchor first tile at the origin.
    A[-1, 0] = 1.0
    weights[-1] = 1000.0

    sw = np.sqrt(weights)
    Aw = A * sw[:, None]
    px = np.linalg.lstsq(Aw, bx * sw, rcond=None)[0]
    py = np.linalg.lstsq(Aw, by * sw, rcond=None)[0]
    positions = np.c_[px, py]

    residuals = []
    for e in edges:
        fitted = positions[e["j"]] - positions[e["i"]]
        expected = np.array([e["dx"], e["dy"]], dtype=np.float64)
        residuals.append(float(np.linalg.norm(fitted - expected)))

    return positions, residuals


def estimate_background(files: list[Path], corner_size: int = 140) -> np.ndarray:
    samples = []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        s = min(corner_size, max(16, h // 5), max(16, w // 5))
        samples.extend(
            [
                im[:s, :s].reshape(-1, 3),
                im[:s, -s:].reshape(-1, 3),
                im[-s:, :s].reshape(-1, 3),
                im[-s:, -s:].reshape(-1, 3),
            ]
        )

    x = np.concatenate(samples, axis=0)
    brightness = x.mean(axis=1)
    bright_half = x[brightness >= np.percentile(brightness, 50)]
    return np.median(bright_half, axis=0).astype(np.uint8)  # BGR


def output_paths(output_dir: Path, sid: str, axon_scale: float | None):
    paths = {
        "tif": output_dir / f"{sid}_100x_stitched_100pct_CLASSIC.tif",
        "jpg": output_dir / f"{sid}_100x_stitched_100pct.jpg",
        "preview": output_dir / f"{sid}_100x_preview.jpg",
        "positions": output_dir / f"{sid}_tile_positions.csv",
        "qc": output_dir / f"{sid}_tile_QC.csv",
        "edges": output_dir / f"{sid}_registration_edges.csv",
    }
    if axon_scale is not None:
        paths["axon_tif"] = output_dir / f"{sid}_100x_{axon_scale:.3f}x_AXONET_CLASSIC.tif"
        paths["axon_jpg"] = output_dir / f"{sid}_100x_{axon_scale:.3f}x_AXONET.jpg"
    return paths


def check_overwrite(paths: dict[str, Path], overwrite: bool):
    existing = [p for p in paths.values() if p.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(p) for p in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n  " + joined
        )


def compose_direct_to_classic_tiff(
    files: list[Path],
    positions_reg: np.ndarray,
    registration_scale: float,
    background_bgr: np.ndarray,
    tif_path: Path,
    work_dir: Path,
):
    full_positions = positions_reg / registration_scale
    full_positions -= np.floor(full_positions.min(axis=0))

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    tile_h, tile_w = first.shape[:2]
    canvas_w = int(math.ceil(full_positions[:, 0].max() + tile_w))
    canvas_h = int(math.ceil(full_positions[:, 1].max() + tile_h))

    raw_bytes = canvas_w * canvas_h * 3
    if raw_bytes >= CLASSIC_TIFF_SAFE_BYTES:
        raise RuntimeError(
            f"Uncompressed mosaic would require {raw_bytes / 1024**3:.2f} GiB, "
            "too close to the classic-TIFF 4 GiB limit. "
            "This script intentionally refuses to create BigTIFF."
        )

    print(f"Final canvas: {canvas_w} x {canvas_h} px")
    print(f"Uncompressed RGB size: {raw_bytes / 1024**2:.1f} MiB")

    # Create an uncompressed CLASSIC TIFF as a memory-mapped RGB image.
    # Because it is the actual output file, there is no giant final TIFF-write step.
    mosaic_rgb = tifffile.memmap(
        tif_path,
        shape=(canvas_h, canvas_w, 3),
        dtype=np.uint8,
        photometric="rgb",
        bigtiff=False,
    )
    mosaic_rgb[:] = background_bgr[::-1]  # BGR -> RGB

    best_path = work_dir / "best_overlap_weight.dat"
    best = np.memmap(
        best_path,
        dtype=np.uint16,
        mode="w+",
        shape=(canvas_h, canvas_w),
    )
    best[:] = 0

    yy = np.arange(tile_h)[:, None]
    xx = np.arange(tile_w)[None, :]
    dist_to_edge = np.minimum(
        np.minimum(xx + 1, tile_w - xx),
        np.minimum(yy + 1, tile_h - yy),
    ).astype(np.uint16)

    for idx, p in enumerate(files):
        tile_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        tile_rgb = tile_bgr[..., ::-1]
        x0 = int(round(full_positions[idx, 0]))
        y0 = int(round(full_positions[idx, 1]))

        weight_roi = best[y0:y0 + tile_h, x0:x0 + tile_w]
        mosaic_roi = mosaic_rgb[y0:y0 + tile_h, x0:x0 + tile_w]
        mask = dist_to_edge > weight_roi
        weight_roi[mask] = dist_to_edge[mask]
        mosaic_roi[mask] = tile_rgb[mask]
        print(f"Placed {idx + 1}/{len(files)}: {p.name}")

    mosaic_rgb.flush()
    best.flush()

    return mosaic_rgb, best, full_positions, canvas_w, canvas_h


def verify_classic_tiff(path: Path, expected_w: int, expected_h: int):
    with path.open("rb") as f:
        header = f.read(4)
    if header in (b"II+\x00", b"MM\x00+"):
        raise RuntimeError("Output is BigTIFF; classic TIFF was required.")
    if header not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"Unexpected TIFF header: {header!r}")

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        if page.imagewidth != expected_w or page.imagelength != expected_h:
            raise RuntimeError("TIFF dimensions do not match expected mosaic dimensions.")
        if page.samplesperpixel != 3:
            raise RuntimeError("TIFF is not 3-channel RGB.")


def write_jpegs_from_mosaic(
    mosaic_rgb: np.ndarray,
    jpg_path: Path,
    preview_path: Path,
    jpeg_quality: int,
    preview_max: int,
):
    h, w = mosaic_rgb.shape[:2]

    # OpenCV expects BGR. Negative-stride RGB->BGR views are not always
    # accepted by writers, so make a contiguous array for the JPEG step.
    mosaic_bgr = np.ascontiguousarray(mosaic_rgb[..., ::-1])
    ok = cv2.imwrite(
        str(jpg_path),
        mosaic_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to write full-resolution JPEG.")

    scale = min(preview_max / w, preview_max / h, 1.0)
    pw = max(1, int(round(w * scale)))
    ph = max(1, int(round(h * scale)))
    preview = cv2.resize(mosaic_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(
        str(preview_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
    )
    if not ok:
        raise RuntimeError("Failed to write preview JPEG.")


def write_axon_scaled_outputs(
    mosaic_rgb: np.ndarray,
    scale: float,
    tif_path: Path,
    jpg_path: Path,
    jpeg_quality: int,
):
    if not (0.0 < scale < 1.0):
        raise ValueError("--axon-scale must be > 0 and < 1")

    h, w = mosaic_rgb.shape[:2]
    target = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    src_bgr = np.ascontiguousarray(mosaic_rgb[..., ::-1])
    scaled_bgr = cv2.resize(src_bgr, target, interpolation=cv2.INTER_AREA)
    scaled_rgb = np.ascontiguousarray(scaled_bgr[..., ::-1])

    raw_bytes = scaled_rgb.size
    if raw_bytes >= CLASSIC_TIFF_SAFE_BYTES:
        raise RuntimeError("Scaled TIFF would exceed safe classic-TIFF size.")

    tifffile.imwrite(
        tif_path,
        scaled_rgb,
        photometric="rgb",
        compression=None,
        bigtiff=False,
    )
    cv2.imwrite(
        str(jpg_path),
        scaled_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    verify_classic_tiff(tif_path, target[0], target[1])


def write_csvs(paths, files, qc_rows, edges, positions_reg, positions_full, scale):
    with paths["qc"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "status", "detail"])
        w.writerows(qc_rows)

    with paths["positions"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filename",
                "registration_x_px",
                "registration_y_px",
                "fullres_x_px",
                "fullres_y_px",
                "registration_scale",
            ]
        )
        for p, pr, pf in zip(files, positions_reg, positions_full):
            w.writerow(
                [
                    p.name,
                    f"{pr[0]:.6f}",
                    f"{pr[1]:.6f}",
                    f"{pf[0]:.3f}",
                    f"{pf[1]:.3f}",
                    scale,
                ]
            )

    with paths["edges"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "tile_i",
                "tile_j",
                "dx_reg_px",
                "dy_reg_px",
                "inliers",
                "pair_residual_px",
            ]
        )
        for e in edges:
            w.writerow(
                [
                    files[e["i"]].name,
                    files[e["j"]].name,
                    f'{e["dx"]:.6f}',
                    f'{e["dy"]:.6f}',
                    e["inliers"],
                    f'{e["pair_residual"]:.6f}',
                ]
            )


def main():
    a = parse_args()
    input_path = Path(a.input).expanduser().resolve()
    output_dir = Path(a.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = output_paths(output_dir, a.sid, a.axon_scale)
    check_overwrite(paths, a.overwrite)

    with tempfile.TemporaryDirectory(prefix="optic_nerve_stitch_") as td:
        work_dir = Path(td)
        raw_files = collect_files(input_path, work_dir)
        files, qc_rows = qc_tiles(raw_files, a.keep_duplicates)

        print(f"Input TIFFs: {len(raw_files)}")
        print(f"Usable TIFFs: {len(files)}")
        for name, status, detail in qc_rows:
            if status != "usable":
                print(f"QC: {name}: {status} {detail}")

        if len(files) < 2:
            raise RuntimeError("Need at least two usable tiles.")

        feats, shapes = compute_features(files, a.registration_scale, a.sift_features)
        edges = build_translation_edges(
            feats,
            shapes,
            a.ratio_test,
            a.min_good_matches,
            a.min_inliers,
            a.inlier_radius,
        )
        if not edges:
            raise RuntimeError("No valid overlap relationships were detected.")

        component = graph_component(len(files), edges)
        print(f"Connected: {len(component)}/{len(files)} tiles")
        print(f"Accepted overlap constraints: {len(edges)}")
        if len(component) != len(files):
            missing = [files[i].name for i in range(len(files)) if i not in component]
            raise RuntimeError(
                "Registration graph is disconnected. Mosaic was NOT written. "
                f"Disconnected tiles: {missing}"
            )

        positions_reg, residuals = solve_positions(len(files), edges)
        med = float(np.median(residuals))
        p95 = float(np.percentile(residuals, 95))
        mx = float(np.max(residuals))
        print(
            f"Global residual @ {a.registration_scale:.3f} scale: "
            f"median={med:.4f}px, 95th={p95:.4f}px, max={mx:.4f}px"
        )

        # A very high graph residual usually means contradictory/false matches.
        # Do not silently make a scientific image in that case.
        if med > 1.0 or p95 > 3.0:
            raise RuntimeError(
                "Registration residuals are too large for an automatic scientific mosaic. "
                "Inspect acquisition overlap / matching parameters instead of forcing output."
            )

        bg_bgr = estimate_background(files)
        print(f"Estimated background BGR: {bg_bgr.tolist()}")

        mosaic_rgb, best, positions_full, cw, ch = compose_direct_to_classic_tiff(
            files,
            positions_reg,
            a.registration_scale,
            bg_bgr,
            paths["tif"],
            work_dir,
        )

        verify_classic_tiff(paths["tif"], cw, ch)
        print("Classic TIFF verification: PASS")

        write_jpegs_from_mosaic(
            mosaic_rgb,
            paths["jpg"],
            paths["preview"],
            a.jpeg_quality,
            a.preview_max,
        )

        if a.axon_scale is not None:
            write_axon_scaled_outputs(
                mosaic_rgb,
                a.axon_scale,
                paths["axon_tif"],
                paths["axon_jpg"],
                a.jpeg_quality,
            )
            print(f"AxoNet-scaled output ({a.axon_scale:.3f}x): PASS")

        write_csvs(
            paths,
            files,
            qc_rows,
            edges,
            positions_reg,
            positions_full,
            a.registration_scale,
        )

        mosaic_rgb.flush()
        best.flush()
        del mosaic_rgb
        del best

    print("\nDONE")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
