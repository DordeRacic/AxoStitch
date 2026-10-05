#!/usr/bin/env python3

"""
stitch_optic_nerve_v3.py

Translation-only stitcher for overlapping 100x optic-nerve TIFF fields.

Features
--------
- Accepts a ZIP archive or directory containing TIFF/TIF tiles.
- Detects blank and exact-duplicate tiles.
- Registers tiles at reduced resolution using SIFT.
- Solves globally consistent tile translations with weighted least squares.
- Refuses disconnected mosaics by default.
- Optional safe exclusion of a very small number of disconnected files.
- Estimates conservative overlap-based additive brightness corrections.
- Feather-blends overlapping tiles to reduce visible seams.
- Writes a classic TIFF (not BigTIFF), full-resolution JPEG, preview JPEG,
  QC CSVs, and an optional AxoNet-scale copy.

Scientific intent
-----------------
The default photometric correction is conservative:
one additive brightness offset is applied equally to B/G/R channels per tile.

It does NOT:
- use CLAHE
- histogram equalize
- sharpen
- denoise
- stretch each tile independently

That makes it preferable for the AxoNet workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile


TIFF_EXTENSIONS = {".tif", ".tiff"}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Stitch overlapping 100x optic-nerve TIFF images."
    )

    p.add_argument(
        "input",
        help="ZIP archive or directory containing TIFF tiles"
    )

    p.add_argument(
        "--sid",
        required=True,
        help='Sample identifier, e.g. "2003 OS"'
    )

    p.add_argument(
        "--output-dir",
        default="stitched_output",
        help="Output directory"
    )

    p.add_argument(
        "--registration-scale",
        type=float,
        default=0.15,
        help="Linear downscale used for SIFT registration"
    )

    p.add_argument(
        "--ratio-test",
        type=float,
        default=0.72,
        help="Lowe ratio threshold for SIFT matches"
    )

    p.add_argument(
        "--min-good-matches",
        type=int,
        default=12,
        help="Minimum ratio-test matches before evaluating a pair"
    )

    p.add_argument(
        "--min-inliers",
        type=int,
        default=10,
        help="Minimum translation-consistent matches for an edge"
    )

    p.add_argument(
        "--inlier-radius",
        type=float,
        default=3.0,
        help="Translation inlier radius at registration scale"
    )

    p.add_argument(
        "--sift-features",
        type=int,
        default=4000,
        help="Maximum SIFT features per tile"
    )

    p.add_argument(
        "--preview-max",
        type=int,
        default=2600,
        help="Maximum preview width/height"
    )

    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="Full-resolution JPEG quality"
    )

    p.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact duplicate images"
    )

    p.add_argument(
        "--axon-scale",
        type=float,
        default=None,
        help="Optional AxoNet downsample, e.g. 0.542"
    )

    p.add_argument(
        "--photometric-correction",
        choices=["none", "brightness", "rgb"],
        default="brightness",
        help=(
            "Overlap-based illumination correction. "
            "'brightness' applies one offset equally to B/G/R "
            "and is recommended for AxoNet."
        ),
    )

    p.add_argument(
        "--max-brightness-offset",
        type=float,
        default=35.0,
        help="Maximum additive correction on the 0-255 scale"
    )

    p.add_argument(
        "--no-feather",
        action="store_true",
        help="Disable feather blending"
    )

    p.add_argument(
        "--allow-small-disconnected",
        action="store_true",
        help=(
            "If <=2 files are outside the largest connected component "
            "and >=90%% of tiles are connected, exclude them and continue. "
            "Useful when overview/reference TIFFs are in the same folder."
        ),
    )

    return p.parse_args()


# ============================================================
# INPUT / QC
# ============================================================

def prepare_input(input_path: Path, work_dir: Path):

    if input_path.is_dir():
        root = input_path

    elif input_path.suffix.lower() == ".zip":

        root = work_dir / "extracted"
        root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(root)

    else:
        raise ValueError(
            "Input must be a ZIP archive or a directory."
        )

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in TIFF_EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            "No TIFF/TIF files were found."
        )

    return files


def image_md5(im):

    return hashlib.md5(
        im.tobytes()
    ).hexdigest()


def qc_tiles(files, keep_duplicates):

    seen_hashes = {}

    usable = []

    qc_rows = []

    expected_shape = None


    for p in files:

        im = cv2.imread(
            str(p),
            cv2.IMREAD_COLOR
        )


        if im is None:

            qc_rows.append(
                (
                    p.name,
                    "unreadable",
                    ""
                )
            )

            continue


        if expected_shape is None:

            expected_shape = im.shape

        elif im.shape != expected_shape:

            qc_rows.append(
                (
                    p.name,
                    "dimension_mismatch",
                    f"shape={im.shape}; expected={expected_shape}"
                )
            )

            continue


        std = float(
            im.std()
        )


        if std < 1.0:

            qc_rows.append(
                (
                    p.name,
                    "blank",
                    f"std={std:.4f}"
                )
            )

            continue


        h = image_md5(im)


        if h in seen_hashes and not keep_duplicates:

            qc_rows.append(
                (
                    p.name,
                    "exact_duplicate",
                    f"duplicate_of={seen_hashes[h].name}"
                )
            )

            continue


        seen_hashes.setdefault(
            h,
            p
        )


        usable.append(p)


        qc_rows.append(
            (
                p.name,
                "usable",
                f"std={std:.4f}"
            )
        )


    return usable, qc_rows


# ============================================================
# SIFT FEATURE EXTRACTION
# ============================================================

def compute_features(
    files,
    scale,
    nfeatures
):

    sift = cv2.SIFT_create(

        nfeatures=nfeatures,

        contrastThreshold=0.012,

        edgeThreshold=10

    )


    features = []

    shapes = []

    keypoint_counts = []


    for idx, p in enumerate(files):

        gray = cv2.imread(
            str(p),
            cv2.IMREAD_GRAYSCALE
        )


        small = cv2.resize(

            gray,

            None,

            fx=scale,

            fy=scale,

            interpolation=cv2.INTER_AREA

        )


        kp, des = sift.detectAndCompute(

            small,

            None

        )


        features.append(
            (kp, des)
        )


        shapes.append(
            small.shape
        )


        n_kp = (
            len(kp)
            if kp is not None
            else 0
        )


        keypoint_counts.append(
            n_kp
        )


        print(

            f"Features {idx + 1}/{len(files)}: "
            f"{p.name} "
            f"({n_kp} keypoints)"

        )


    return (
        features,
        shapes,
        keypoint_counts
    )


# ============================================================
# PAIRWISE REGISTRATION
# ============================================================

def build_translation_edges(

    features,

    shapes,

    ratio_test,

    min_good,

    min_inliers,

    inlier_radius

):

    bf = cv2.BFMatcher()


    edges = []


    for i, j in itertools.combinations(

        range(len(features)),

        2

    ):


        kp1, des1 = features[i]

        kp2, des2 = features[j]


        if des1 is None or des2 is None:

            continue


        knn = bf.knnMatch(

            des1,

            des2,

            k=2

        )


        good = [

            a

            for a, b in knn

            if a.distance
            < ratio_test * b.distance

        ]


        if len(good) < min_good:

            continue


        p1 = np.float32(

            [
                kp1[m.queryIdx].pt

                for m in good
            ]

        )


        p2 = np.float32(

            [
                kp2[m.trainIdx].pt

                for m in good
            ]

        )


        d = p2 - p1


        med = np.median(

            d,

            axis=0

        )


        err = np.linalg.norm(

            d - med,

            axis=1

        )


        inliers = (

            err
            < inlier_radius

        )


        nin = int(

            inliers.sum()

        )


        if nin < min_inliers:

            continue


        t = np.median(

            d[inliers],

            axis=0

        )


        # IMPORTANT:
        # feature motion is opposite
        # tile-origin motion

        delta = -t


        h, w = shapes[i]


        if (

            abs(delta[0])
            > 0.97 * w

            or

            abs(delta[1])
            > 0.97 * h

        ):

            continue


        edges.append(

            {

                "i": i,

                "j": j,

                "dx": float(
                    delta[0]
                ),

                "dy": float(
                    delta[1]
                ),

                "inliers": nin,

                "pair_residual": float(

                    np.median(
                        err[inliers]
                    )

                ),

            }

        )


    return edges


# ============================================================
# GRAPH CONNECTIVITY
# ============================================================

def graph_components(
    n,
    edges
):

    adj = {

        i: set()

        for i in range(n)

    }


    for e in edges:

        adj[e["i"]].add(
            e["j"]
        )

        adj[e["j"]].add(
            e["i"]
        )


    remaining = set(

        range(n)

    )


    comps = []


    while remaining:

        start = next(

            iter(remaining)

        )


        seen = {
            start
        }


        stack = [
            start
        ]


        while stack:

            u = stack.pop()


            for v in adj[u]:

                if v not in seen:

                    seen.add(v)

                    stack.append(v)


        comps.append(

            sorted(seen)

        )


        remaining -= seen


    comps.sort(

        key=len,

        reverse=True

    )


    return comps


def subset_graph(

    files,

    edges,

    keep_indices

):

    keep_indices = sorted(

        keep_indices

    )


    remap = {

        old: new

        for new, old
        in enumerate(
            keep_indices
        )

    }


    new_files = [

        files[i]

        for i in keep_indices

    ]


    new_edges = []


    for e in edges:

        if (

            e["i"] in remap

            and

            e["j"] in remap

        ):


            ee = dict(e)


            ee["i"] = remap[
                e["i"]
            ]


            ee["j"] = remap[
                e["j"]
            ]


            new_edges.append(
                ee
            )


    return (
        new_files,
        new_edges
    )


# ============================================================
# GLOBAL POSITION SOLVE
# ============================================================

def solve_positions(

    n,

    edges

):

    m = len(edges)


    A = np.zeros(

        (m + 1, n),

        dtype=np.float64

    )


    bx = np.zeros(

        m + 1,

        dtype=np.float64

    )


    by = np.zeros(

        m + 1,

        dtype=np.float64

    )


    weights = np.ones(

        m + 1,

        dtype=np.float64

    )


    for k, e in enumerate(edges):

        i = e["i"]

        j = e["j"]


        A[k, i] = -1.0

        A[k, j] = 1.0


        bx[k] = e["dx"]

        by[k] = e["dy"]


        weights[k] = max(

            e["inliers"],

            1

        )


    # anchor tile 0

    A[-1, 0] = 1.0

    weights[-1] = 1000.0


    sw = np.sqrt(

        weights

    )


    Aw = (

        A
        * sw[:, None]

    )


    px = np.linalg.lstsq(

        Aw,

        bx * sw,

        rcond=None

    )[0]


    py = np.linalg.lstsq(

        Aw,

        by * sw,

        rcond=None

    )[0]


    pos = np.c_[

        px,

        py

    ]


    residuals = []


    for e in edges:

        actual = (

            pos[e["j"]]
            -
            pos[e["i"]]

        )


        expected = np.array(

            [
                e["dx"],
                e["dy"]
            ]

        )


        residuals.append(

            float(

                np.linalg.norm(

                    actual
                    -
                    expected

                )

            )

        )


    return (
        pos,
        residuals
    )


# ============================================================
# BACKGROUND ESTIMATION
# ============================================================

def estimate_background(

    files,

    corner_size=140

):

    samples = []


    for p in files:

        im = cv2.imread(

            str(p),

            cv2.IMREAD_COLOR

        )


        h, w = im.shape[:2]


        s = min(

            corner_size,

            h // 5,

            w // 5

        )


        samples.extend(

            [

                im[
                    :s,
                    :s
                ].reshape(
                    -1,
                    3
                ),

                im[
                    :s,
                    -s:
                ].reshape(
                    -1,
                    3
                ),

                im[
                    -s:,
                    :s
                ].reshape(
                    -1,
                    3
                ),

                im[
                    -s:,
                    -s:
                ].reshape(
                    -1,
                    3
                ),

            ]

        )


    arr = np.concatenate(

        samples,

        axis=0

    )


    brightness = arr.mean(

        axis=1

    )


    use = arr[

        brightness

        >=

        np.percentile(
            brightness,
            50
        )

    ]


    return np.median(

        use,

        axis=0

    ).astype(
        np.uint8
    )


# ============================================================
# PHOTOMETRIC / LIGHTING CORRECTION
# ============================================================

def estimate_photometric_offsets(

    files,

    positions_reg,

    edges,

    registration_scale,

    mode="brightness",

    max_offset=35.0

):

    n = len(files)


    if mode == "none":

        return np.zeros(

            (n, 3),

            dtype=np.float32

        )


    small = []


    for p in files:

        im = cv2.imread(

            str(p),

            cv2.IMREAD_COLOR

        )


        ds = cv2.resize(

            im,

            None,

            fx=registration_scale,

            fy=registration_scale,

            interpolation=cv2.INTER_AREA

        )


        small.append(ds)


    rows = []

    values = []

    weights = []


    for e in edges:

        i = e["i"]

        j = e["j"]


        im_i = small[i]

        im_j = small[j]


        hi, wi = im_i.shape[:2]

        hj, wj = im_j.shape[:2]


        xi = int(

            round(
                positions_reg[i, 0]
            )

        )


        yi = int(

            round(
                positions_reg[i, 1]
            )

        )


        xj = int(

            round(
                positions_reg[j, 0]
            )

        )


        yj = int(

            round(
                positions_reg[j, 1]
            )

        )


        # Global overlap

        gx0 = max(
            xi,
            xj
        )


        gy0 = max(
            yi,
            yj
        )


        gx1 = min(

            xi + wi,

            xj + wj

        )


        gy1 = min(

            yi + hi,

            yj + hj

        )


        ow = gx1 - gx0

        oh = gy1 - gy0


        if ow < 20 or oh < 20:

            continue


        ai_x0 = gx0 - xi

        ai_y0 = gy0 - yi


        aj_x0 = gx0 - xj

        aj_y0 = gy0 - yj


        A = im_i[

            ai_y0:
            ai_y0 + oh,

            ai_x0:
            ai_x0 + ow

        ]


        B = im_j[

            aj_y0:
            aj_y0 + oh,

            aj_x0:
            aj_x0 + ow

        ]


        # Downsample overlap

        A = A[
            ::2,
            ::2
        ].astype(
            np.float32
        )


        B = B[
            ::2,
            ::2
        ].astype(
            np.float32
        )


        # Avoid black / saturated pixels

        valid = (

            (
                A.min(
                    axis=2
                )
                > 15
            )

            &

            (
                B.min(
                    axis=2
                )
                > 15
            )

            &

            (
                A.max(
                    axis=2
                )
                < 245
            )

            &

            (
                B.max(
                    axis=2
                )
                < 245
            )

        )


        if int(
            valid.sum()
        ) < 500:

            continue


        if mode == "brightness":

            gray_a = cv2.cvtColor(

                A.astype(
                    np.uint8
                ),

                cv2.COLOR_BGR2GRAY

            ).astype(
                np.float32
            )


            gray_b = cv2.cvtColor(

                B.astype(
                    np.uint8
                ),

                cv2.COLOR_BGR2GRAY

            ).astype(
                np.float32
            )


            difference = float(

                np.median(

                    gray_a[valid]
                    -
                    gray_b[valid]

                )

            )


            d = np.array(

                [
                    difference,
                    difference,
                    difference
                ],

                dtype=np.float64

            )


        else:

            d = np.median(

                A[valid]
                -
                B[valid],

                axis=0

            ).astype(
                np.float64
            )


        row = np.zeros(

            n,

            dtype=np.float64

        )


        # corrected_i = image_i + c_i
        # corrected_j = image_j + c_j
        #
        # therefore
        #
        # c_j - c_i ~= image_i - image_j

        row[i] = -1.0

        row[j] = +1.0


        rows.append(
            row
        )


        values.append(
            d
        )


        weights.append(

            min(

                float(
                    valid.sum()
                ),

                100000.0

            )

        )


    if not rows:

        print(

            "WARNING: no valid overlap regions "
            "for photometric correction. "
            "Using zero offsets."

        )


        return np.zeros(

            (n, 3),

            dtype=np.float32

        )


    M = np.asarray(

        rows,

        dtype=np.float64

    )


    D = np.asarray(

        values,

        dtype=np.float64

    )


    W = np.sqrt(

        np.asarray(

            weights,

            dtype=np.float64

        )

    )


    # Anchor tile 0

    anchor = np.zeros(

        (1, n),

        dtype=np.float64

    )


    anchor[
        0,
        0
    ] = 1.0


    M = np.vstack(

        [
            M,
            anchor
        ]

    )


    D = np.vstack(

        [

            D,

            np.zeros(

                (1, 3),

                dtype=np.float64

            )

        ]

    )


    W = np.concatenate(

        [
            W,
            [1000.0]
        ]

    )


    Mw = (

        M
        * W[:, None]

    )


    corrections = np.zeros(

        (n, 3),

        dtype=np.float64

    )


    for ch in range(3):

        corrections[
            :,
            ch
        ] = np.linalg.lstsq(

            Mw,

            D[:, ch] * W,

            rcond=None

        )[0]


    # Keep global image brightness centered
    # near the original acquisition.

    corrections -= np.median(

        corrections,

        axis=0,

        keepdims=True

    )


    corrections = np.clip(

        corrections,

        -max_offset,

        max_offset

    )


    print(
        "\nPhotometric corrections (B, G, R):"
    )


    for p, c in zip(

        files,

        corrections

    ):

        print(

            f"  {p.name}: "
            f"{c[0]:+.1f}, "
            f"{c[1]:+.1f}, "
            f"{c[2]:+.1f}"

        )


    return corrections.astype(
        np.float32
    )


def apply_photometric_offset(

    tile_bgr,

    offset_bgr

):

    corrected = (

        tile_bgr.astype(
            np.float32
        )

        +

        offset_bgr.reshape(
            1,
            1,
            3
        )

    )


    return np.clip(

        corrected,

        0,

        255

    ).astype(
        np.uint8
    )


# ============================================================
# FEATHER WEIGHT
# ============================================================

def make_distance_weight(

    tile_h,

    tile_w

):

    y = np.arange(

        tile_h

    )[:, None]


    x = np.arange(

        tile_w

    )[None, :]


    d = np.minimum(

        np.minimum(

            x + 1,

            tile_w - x

        ),

        np.minimum(

            y + 1,

            tile_h - y

        )

    )


    return np.maximum(

        d,

        1

    ).astype(
        np.uint16
    )


# ============================================================
# CLASSIC TIFF UTILITIES
# ============================================================

def classic_tiff_header_ok(

    path

):

    with open(

        path,

        "rb"

    ) as f:

        hdr = f.read(4)


    return hdr in (

        b"II*\x00",

        b"MM\x00*"

    )


def create_classic_tiff_memmap(

    path,

    shape

):

    estimated_bytes = int(

        np.prod(
            shape
        )

    )


    # Keep comfortably below
    # classic TIFF's ~4 GiB limit.

    if estimated_bytes >= 3_800_000_000:

        raise RuntimeError(

            "Estimated TIFF is too large "
            "for a safe classic TIFF."

        )


    mm = tifffile.memmap(

        path,

        shape=shape,

        dtype=np.uint8,

        photometric="rgb",

        bigtiff=False

    )


    return mm


# ============================================================
# MOSAIC COMPOSITION
# ============================================================

def compose_mosaic(

    files,

    positions_reg,

    registration_scale,

    background_bgr,

    tif_path,

    work_dir,

    photometric_offsets=None,

    feather=True

):

    full_positions = (

        positions_reg
        /
        registration_scale

    )


    mins = np.floor(

        full_positions.min(
            axis=0
        )

    )


    full_positions = (

        full_positions
        -
        mins

    )


    first = cv2.imread(

        str(files[0]),

        cv2.IMREAD_COLOR

    )


    tile_h, tile_w = first.shape[:2]


    canvas_w = int(

        math.ceil(

            full_positions[:, 0].max()
            +
            tile_w

        )

    )


    canvas_h = int(

        math.ceil(

            full_positions[:, 1].max()
            +
            tile_h

        )

    )


    print(

        f"Canvas: "
        f"{canvas_w} x {canvas_h}"

    )


    mosaic_rgb = create_classic_tiff_memmap(

        tif_path,

        (
            canvas_h,
            canvas_w,
            3
        )

    )


    mosaic_rgb[:] = (

        background_bgr[
            ::-1
        ]

    )


    weight_path = (

        work_dir
        /
        "overlap_weight_sum.dat"

    )


    weight_sum = np.memmap(

        weight_path,

        dtype=np.uint16,

        mode="w+",

        shape=(

            canvas_h,

            canvas_w

        )

    )


    weight_sum[:] = 0


    dist_to_edge = make_distance_weight(

        tile_h,

        tile_w

    )


    if photometric_offsets is None:

        photometric_offsets = np.zeros(

            (
                len(files),
                3
            ),

            dtype=np.float32

        )


    for idx, p in enumerate(files):


        tile_bgr = cv2.imread(

            str(p),

            cv2.IMREAD_COLOR

        )


        tile_bgr = apply_photometric_offset(

            tile_bgr,

            photometric_offsets[
                idx
            ]

        )


        tile_rgb = (

            tile_bgr[
                ...,
                ::-1
            ]

        )


        x0 = int(

            round(

                full_positions[
                    idx,
                    0
                ]

            )

        )


        y0 = int(

            round(

                full_positions[
                    idx,
                    1
                ]

            )

        )


        mosaic_roi = mosaic_rgb[

            y0:
            y0 + tile_h,

            x0:
            x0 + tile_w

        ]


        weight_roi = weight_sum[

            y0:
            y0 + tile_h,

            x0:
            x0 + tile_w

        ]


        # ------------------------------------------
        # ORIGINAL HARD OVERLAP METHOD
        # ------------------------------------------

        if not feather:


            mask = (

                dist_to_edge
                >
                weight_roi

            )


            weight_roi[
                mask
            ] = dist_to_edge[
                mask
            ]


            mosaic_roi[
                mask
            ] = tile_rgb[
                mask
            ]


        # ------------------------------------------
        # FEATHER BLENDING
        # ------------------------------------------

        else:


            new_weight = (

                dist_to_edge.astype(
                    np.float32
                )

            )


            old_weight = (

                weight_roi.astype(
                    np.float32
                )

            )


            empty = (

                weight_roi
                ==
                0

            )


            # First tile touching a pixel
            # copies directly.

            mosaic_roi[
                empty
            ] = tile_rgb[
                empty
            ]


            overlap = ~empty


            if np.any(
                overlap
            ):


                ow = old_weight[
                    overlap
                ]


                nw = new_weight[
                    overlap
                ]


                denom = (

                    ow
                    +
                    nw

                )


                old_pixels = mosaic_roi[

                    overlap

                ].astype(
                    np.float32
                )


                new_pixels = tile_rgb[

                    overlap

                ].astype(
                    np.float32
                )


                blended = (

                    old_pixels
                    *
                    ow[:, None]

                    +

                    new_pixels
                    *
                    nw[:, None]

                ) / denom[:, None]


                mosaic_roi[
                    overlap
                ] = np.clip(

                    np.rint(
                        blended
                    ),

                    0,

                    255

                ).astype(
                    np.uint8
                )


            summed = (

                weight_roi.astype(
                    np.uint32
                )

                +

                dist_to_edge.astype(
                    np.uint32
                )

            )


            weight_roi[:] = np.minimum(

                summed,

                65535

            ).astype(
                np.uint16
            )


        print(

            f"Placed "
            f"{idx + 1}/"
            f"{len(files)}: "
            f"{p.name}"

        )


    mosaic_rgb.flush()

    weight_sum.flush()


    return (

        mosaic_rgb,

        weight_sum,

        full_positions,

        canvas_w,

        canvas_h

    )


# ============================================================
# JPEG / PREVIEW
# ============================================================

def rgb_memmap_to_bgr_array(

    mosaic_rgb

):

    return mosaic_rgb[
        ...,
        ::-1
    ]


def save_jpeg_and_preview(

    mosaic_rgb,

    jpg_path,

    preview_path,

    preview_max,

    jpeg_quality

):

    bgr = rgb_memmap_to_bgr_array(

        mosaic_rgb

    )


    h, w = bgr.shape[:2]


    ok = cv2.imwrite(

        str(jpg_path),

        bgr,

        [

            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),

            jpeg_quality

        ]

    )


    if not ok:

        raise RuntimeError(

            "Failed to write "
            "full-resolution JPEG."

        )


    scale = min(

        preview_max / w,

        preview_max / h,

        1.0

    )


    preview = cv2.resize(

        bgr,

        (

            max(

                1,

                int(
                    round(
                        w * scale
                    )
                )

            ),

            max(

                1,

                int(
                    round(
                        h * scale
                    )
                )

            )

        ),

        interpolation=cv2.INTER_AREA

    )


    ok = cv2.imwrite(

        str(
            preview_path
        ),

        preview,

        [

            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),

            88

        ]

    )


    if not ok:

        raise RuntimeError(

            "Failed to write "
            "preview JPEG."

        )


# ============================================================
# AXONET-SCALE COPY
# ============================================================

def save_axon_copy(

    mosaic_rgb,

    out_tif,

    out_jpg,

    scale,

    jpeg_quality

):

    if not (

        0
        <
        scale
        <
        1

    ):

        raise ValueError(

            "--axon-scale must "
            "be between 0 and 1."

        )


    bgr = rgb_memmap_to_bgr_array(

        mosaic_rgb

    )


    h, w = bgr.shape[:2]


    new_w = max(

        1,

        int(
            round(
                w * scale
            )
        )

    )


    new_h = max(

        1,

        int(
            round(
                h * scale
            )
        )

    )


    ax_bgr = cv2.resize(

        bgr,

        (
            new_w,
            new_h
        ),

        interpolation=cv2.INTER_AREA

    )


    ax_rgb = ax_bgr[
        ...,
        ::-1
    ]


    tifffile.imwrite(

        out_tif,

        ax_rgb,

        photometric="rgb",

        compression=None,

        bigtiff=False

    )


    if not classic_tiff_header_ok(

        out_tif

    ):

        raise RuntimeError(

            "AxoNet TIFF was not "
            "written as classic TIFF."

        )


    cv2.imwrite(

        str(out_jpg),

        ax_bgr,

        [

            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),

            jpeg_quality

        ]

    )


# ============================================================
# CSV OUTPUTS
# ============================================================

def write_positions_csv(

    path,

    files,

    positions_full

):

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:


        writer = csv.writer(
            f
        )


        writer.writerow(

            [

                "filename",

                "x_px",

                "y_px"

            ]

        )


        for p, xy in zip(

            files,

            positions_full

        ):


            writer.writerow(

                [

                    p.name,

                    f"{xy[0]:.3f}",

                    f"{xy[1]:.3f}"

                ]

            )


def write_qc_csv(

    path,

    rows

):

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:


        writer = csv.writer(
            f
        )


        writer.writerow(

            [

                "filename",

                "status",

                "detail"

            ]

        )


        writer.writerows(
            rows
        )


def write_edges_csv(

    path,

    files,

    edges,

    positions_reg

):

    with open(

        path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:


        writer = csv.writer(
            f
        )


        writer.writerow(

            [

                "tile_i",

                "tile_j",

                "dx_reg_px",

                "dy_reg_px",

                "inliers",

                "pair_residual_px",

                "global_residual_px"

            ]

        )


        for e in edges:


            i = e["i"]

            j = e["j"]


            actual = (

                positions_reg[j]
                -
                positions_reg[i]

            )


            expected = np.array(

                [

                    e["dx"],

                    e["dy"]

                ]

            )


            global_residual = float(

                np.linalg.norm(

                    actual
                    -
                    expected

                )

            )


            writer.writerow(

                [

                    files[i].name,

                    files[j].name,

                    f'{e["dx"]:.6f}',

                    f'{e["dy"]:.6f}',

                    e["inliers"],

                    f'{e["pair_residual"]:.6f}',

                    f"{global_residual:.6f}"

                ]

            )


# ============================================================
# MAIN
# ============================================================

def main():


    a = parse_args()


    input_path = Path(

        a.input

    ).expanduser().resolve()


    output_dir = Path(

        a.output_dir

    ).expanduser().resolve()


    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )


    safe_sid = (

        a.sid.strip()

        .replace(
            "/",
            "_"
        )

        .replace(
            "\\",
            "_"
        )

    )


    tif_path = output_dir / (

        f"{safe_sid}_"
        f"100x_stitched_"
        f"100pct_CLASSIC.tif"

    )


    jpg_path = output_dir / (

        f"{safe_sid}_"
        f"100x_stitched_"
        f"100pct.jpg"

    )


    preview_path = output_dir / (

        f"{safe_sid}_"
        f"100x_preview.jpg"

    )


    qc_path = output_dir / (

        f"{safe_sid}_"
        f"tile_QC.csv"

    )


    pos_path = output_dir / (

        f"{safe_sid}_"
        f"tile_positions.csv"

    )


    edges_path = output_dir / (

        f"{safe_sid}_"
        f"registration_edges.csv"

    )


    # Prevent accidental overwrite

    for p in [

        tif_path,

        jpg_path,

        preview_path,

        qc_path,

        pos_path,

        edges_path

    ]:

        if p.exists():

            raise FileExistsError(

                f"Output already exists:\n"
                f"{p}\n\n"
                f"Delete/move it or use "
                f"a different output directory."

            )


    with tempfile.TemporaryDirectory(

        prefix="optic_nerve_stitch_"

    ) as temp_dir:


        work_dir = Path(

            temp_dir

        )


        raw_files = prepare_input(

            input_path,

            work_dir

        )


        files, qc_rows = qc_tiles(

            raw_files,

            a.keep_duplicates

        )


        print(

            f"Input TIFFs: "
            f"{len(raw_files)}"

        )


        print(

            f"Usable TIFFs: "
            f"{len(files)}"

        )


        if len(files) < 2:

            raise RuntimeError(

                "Need at least "
                "two usable TIFF tiles."

            )


        # -------------------------------
        # FEATURE EXTRACTION
        # -------------------------------

        features, shapes, keypoint_counts = compute_features(

            files,

            a.registration_scale,

            a.sift_features

        )


        # -------------------------------
        # PAIRWISE REGISTRATION
        # -------------------------------

        edges = build_translation_edges(

            features,

            shapes,

            a.ratio_test,

            a.min_good_matches,

            a.min_inliers,

            a.inlier_radius

        )


        if not edges:

            raise RuntimeError(

                "No valid overlapping "
                "tile pairs were found."

            )


        # -------------------------------
        # CONNECTIVITY CHECK
        # -------------------------------

        comps = graph_components(

            len(files),

            edges

        )


        largest = comps[0]


        disconnected = sorted(

            set(
                range(
                    len(files)
                )
            )

            -

            set(
                largest
            )

        )


        print(

            f"Connected: "
            f"{len(largest)}/"
            f"{len(files)} tiles"

        )


        print(

            f"Accepted overlap constraints: "
            f"{len(edges)}"

        )


        if disconnected:


            disconnected_names = [

                files[i].name

                for i
                in disconnected

            ]


            if (

                a.allow_small_disconnected

                and

                len(disconnected)
                <=
                2

                and

                len(largest)
                /
                len(files)
                >=
                0.90

            ):


                print(

                    "\nWARNING: excluding "
                    "small disconnected component(s):"

                )


                for name in disconnected_names:

                    print(
                        f"  {name}"
                    )


                    qc_rows.append(

                        (

                            name,

                            "excluded_disconnected",

                            "not in largest "
                            "registration component"

                        )

                    )


                files, edges = subset_graph(

                    files,

                    edges,

                    largest

                )


            else:

                raise RuntimeError(

                    "Registration graph is disconnected. "
                    "Mosaic was NOT written. "
                    f"Disconnected tiles: "
                    f"{disconnected_names}"

                )


        # -------------------------------
        # GLOBAL POSITION SOLVE
        # -------------------------------

        positions_reg, residuals = solve_positions(

            len(files),

            edges

        )


        med_res = float(

            np.median(
                residuals
            )

        )


        p95_res = float(

            np.percentile(

                residuals,

                95

            )

        )


        print(

            f"Global residual median: "
            f"{med_res:.4f} px "
            f"@ registration scale"

        )


        print(

            f"Global residual 95th:   "
            f"{p95_res:.4f} px "
            f"@ registration scale"

        )


        if (

            med_res > 1.0

            or

            p95_res > 3.0

        ):

            raise RuntimeError(

                "Registration residuals "
                "are suspiciously large. "
                "Mosaic was NOT written."

            )


        # -------------------------------
        # BACKGROUND
        # -------------------------------

        background_bgr = estimate_background(

            files

        )


        print(

            f"Estimated BGR background: "
            f"{background_bgr.tolist()}"

        )


        # -------------------------------
        # LIGHTING CORRECTION
        # -------------------------------

        photometric_offsets = estimate_photometric_offsets(

            files=files,

            positions_reg=positions_reg,

            edges=edges,

            registration_scale=a.registration_scale,

            mode=a.photometric_correction,

            max_offset=a.max_brightness_offset

        )


        # -------------------------------
        # COMPOSE MOSAIC
        # -------------------------------

        mosaic_rgb, weight_sum, positions_full, cw, ch = compose_mosaic(

            files=files,

            positions_reg=positions_reg,

            registration_scale=a.registration_scale,

            background_bgr=background_bgr,

            tif_path=tif_path,

            work_dir=work_dir,

            photometric_offsets=photometric_offsets,

            feather=not a.no_feather

        )


        # -------------------------------
        # VERIFY TIFF
        # -------------------------------

        if not classic_tiff_header_ok(

            tif_path

        ):

            raise RuntimeError(

                "Main TIFF was not "
                "written as classic TIFF."

            )


        # -------------------------------
        # JPEG + PREVIEW
        # -------------------------------

        save_jpeg_and_preview(

            mosaic_rgb,

            jpg_path,

            preview_path,

            a.preview_max,

            a.jpeg_quality

        )


        # -------------------------------
        # QC CSVs
        # -------------------------------

        write_positions_csv(

            pos_path,

            files,

            positions_full

        )


        write_qc_csv(

            qc_path,

            qc_rows

        )


        write_edges_csv(

            edges_path,

            files,

            edges,

            positions_reg

        )


        # -------------------------------
        # AXONET COPY
        # -------------------------------

        axon_outputs = []


        if a.axon_scale is not None:


            ax_tif = output_dir / (

                f"{safe_sid}_"
                f"100x_"
                f"{a.axon_scale:.3f}x_"
                f"AXONET_CLASSIC.tif"

            )


            ax_jpg = output_dir / (

                f"{safe_sid}_"
                f"100x_"
                f"{a.axon_scale:.3f}x_"
                f"AXONET.jpg"

            )


            if (

                ax_tif.exists()

                or

                ax_jpg.exists()

            ):

                raise FileExistsError(

                    "AxoNet output already exists. "
                    "Use a clean output directory."

                )


            save_axon_copy(

                mosaic_rgb,

                ax_tif,

                ax_jpg,

                a.axon_scale,

                a.jpeg_quality

            )


            axon_outputs = [

                ax_tif,

                ax_jpg

            ]


        mosaic_rgb.flush()

        weight_sum.flush()


        del mosaic_rgb

        del weight_sum


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\nDONE"
    )


    print(

        f"Classic TIFF: "
        f"{tif_path}"

    )


    print(

        f"Full JPEG:    "
        f"{jpg_path}"

    )


    print(

        f"Preview:      "
        f"{preview_path}"

    )


    print(

        f"Positions:    "
        f"{pos_path}"

    )


    print(

        f"Tile QC:      "
        f"{qc_path}"

    )


    print(

        f"Edges QC:     "
        f"{edges_path}"

    )


    for p in axon_outputs:

        print(

            f"AxoNet copy:  "
            f"{p}"

        )


if __name__ == "__main__":

    main()