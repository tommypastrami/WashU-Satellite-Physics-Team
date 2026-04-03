#Two Goals: 1 -- Predict what the 
# point-spread function convolution of the camera lens is based 
# on the light sources within the photos given.
# 2 -- is a file that takes in our video and outputs
# objects and their directionality

#imports
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import cv2
from glob import glob
import rawpy
import os

#1. Sorting, cr2_files is array of strings
cr2_files = sorted(glob("/Users/dunnmattye/Downloads/2-21-25 Laser Photos/*.CR2"))
if not cr2_files:
    raise FileNotFoundError("No CR2 files found in the specified directory.")

def load_cr2_to_bgr(path):
    """
    Load a CR2 file and return a BGR image for OpenCV.
    """
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# NOTE: In batch mode, bgr/rgb are created inside the runtime loop.
# If you want a single-image debug preview, set DEBUG=True below.
DEBUG = False
if DEBUG:
    debug_path = cr2_files[0]
    bgr = load_cr2_to_bgr(debug_path)

#2. preprocessing
def preprocessing(image):
    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sat_thresh = 245
    frac_thresh = 0.01
    bright_thresh = 50
    area_frac_thresh = 0.01
    median_ksize = 21
    min_blob_area_px = 20000

    bright_mask = (grayscale >= bright_thresh).astype(np.uint8)*255
    blobKernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, blobKernal, iterations=2)

    #big blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    H, W = grayscale.shape
    frame_area = H * W
    if num_labels > 1:
        max_blob_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        if max_blob_area > max(min_blob_area_px, area_frac_thresh * frame_area):
            return None

    #shitty background
    bright_frac = (grayscale >= sat_thresh).mean()
    if bright_frac > frac_thresh:
        return None

    # Subtract background with median filter (float pipeline)
    g = grayscale.astype(np.float32)
    bg = cv2.medianBlur(grayscale, median_ksize).astype(np.float32)
    subtracted = g - bg
    subtracted = np.clip(subtracted, 0, None)
    return subtracted.astype(np.float32)

# OPTIONAL debug preview (single image)
if DEBUG:
    preprocessed_image = preprocessing(bgr)
    if preprocessed_image is None:
        print("Image rejected due to high brightness fraction.")
    else:
        print(preprocessed_image.shape, preprocessed_image.dtype)
        plt.imshow(preprocessed_image, cmap='gray')
        plt.colorbar()
        plt.show()

#3. Detect candidate point sources
def renyi_entropy(patch, alpha=2.0, eps=1e-12):
    """
    Rényi entropy of order alpha on a patch interpreted as a discrete distribution.
    Lower => more concentrated energy (more point-like).
    """
    x = patch.astype(np.float64)
    x = x - x.min()
    s = x.sum()
    if s <= eps:
        return np.inf
    p = x / (s + eps)

    if abs(alpha - 1.0) < 1e-6:
        return -np.sum(p * np.log(p + eps))
    return (1.0 / (1.0 - alpha)) * np.log(np.sum((p + eps) ** alpha))

def extract_patch(img, x, y, r):
    """r is half-size; patch is (2r+1)x(2r+1)."""
    return img[y-r:y+r+1, x-r:x+r+1]

def local_maxima(binary_strength_img, ksize=9):
    """
    Find local maxima positions in an image using dilation.
    Returns boolean mask of maxima.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil = cv2.dilate(binary_strength_img, kernel)
    return (binary_strength_img == dil)

def detect_candidate_point_sources(preprocessed,
                                   patch_size=31,
                                   dog_sigma_small=1.2,
                                   dog_sigma_large=2.6,
                                   peak_ksize=7,
                                   max_candidates=5000,
                                   snr_thresh=6.0,
                                   sat_thresh=None,
                                   renyi_alpha=2.0,
                                   renyi_max=2.2,
                                   ecc_max=0.75,
                                   min_peak=5.0):
    """
    Returns a list of accepted candidates with features.
    """
    if preprocessed is None:
        return [], None

    img = preprocessed.astype(np.float32)
    H, W = img.shape
    r = patch_size // 2

    if sat_thresh is None:
        sat_thresh = np.percentile(img, 99.999)

    # 1) Difference of Gaussians (band-pass)
    g1 = cv2.GaussianBlur(img, (0, 0), dog_sigma_small)
    g2 = cv2.GaussianBlur(img, (0, 0), dog_sigma_large)
    dog = g1 - g2
    dog = np.maximum(dog, 0)

    # Robust threshold for peaks (median + k*MAD style)
    med = np.median(dog)
    mad = np.median(np.abs(dog - med)) + 1e-6
    thresh = med + 3.0 * mad
    thresh = max(thresh, min_peak)

    maxima_mask = local_maxima(dog, ksize=peak_ksize)
    peak_mask = (dog > thresh) & maxima_mask

    ys, xs = np.where(peak_mask)
    if len(xs) == 0:
        return [], dog

    responses = dog[ys, xs]
    order = np.argsort(-responses)[:max_candidates]
    xs, ys, responses = xs[order], ys[order], responses[order]

    candidates = []

    for x, y, resp in zip(xs, ys, responses):
        if x < r or x >= W - r or y < r or y >= H - r:
            continue

        patch = extract_patch(img, x, y, r)

        peak = patch[r, r]
        if peak >= sat_thresh:
            continue

        border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
        mu = float(np.mean(border))
        sigma = float(np.std(border) + 1e-6)

        snr = (float(peak) - mu) / sigma
        if snr < snr_thresh:
            continue

        p = patch.astype(np.float64)
        p = p - p.min()
        s = p.sum()
        if s <= 1e-12:
            continue
        p /= s

        yy, xx = np.mgrid[0:patch_size, 0:patch_size]
        x0 = (p * xx).sum()
        y0 = (p * yy).sum()
        dx = xx - x0
        dy = yy - y0
        cov_xx = (p * dx * dx).sum()
        cov_yy = (p * dy * dy).sum()
        cov_xy = (p * dx * dy).sum()
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        evals = np.linalg.eigvalsh(cov)
        l1, l2 = float(evals[1]), float(evals[0])
        if l1 <= 1e-12:
            continue
        ecc = 1.0 - (l2 / l1)
        if ecc > ecc_max:
            continue

        Hren = renyi_entropy(patch, alpha=renyi_alpha)
        if Hren > renyi_max:
            continue

        candidates.append({
            "x": int(x),
            "y": int(y),
            "dog_resp": float(resp),
            "peak": float(peak),
            "snr": float(snr),
            "ecc": float(ecc),
            "Hren": float(Hren),
        })

    return candidates, dog

#Step 4: Extract patches + subpixel center + normalize
def extract_patches(img, candidates, patch_size=31):
    r = patch_size // 2
    H, W = img.shape
    patches = []
    meta = []
    for d in candidates:
        x, y = d["x"], d["y"]
        if x < r or x >= W - r or y < r or y >= H - r:
            continue
        patch = img[y-r:y+r+1, x-r:x+r+1].astype(np.float32)
        patches.append(patch)
        meta.append(d)
    return patches, meta

def subpixel_center_and_shift(patch, eps=1e-6):
    p = patch.astype(np.float64)
    p = p - p.min()
    s = p.sum()
    if s <= eps:
        return patch, (0.0, 0.0)

    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (p * xx).sum() / s
    cy = (p * yy).sum() / s

    target_x = (w - 1) / 2.0
    target_y = (h - 1) / 2.0
    dx = target_x - cx
    dy = target_y - cy

    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)

    shifted = cv2.warpAffine(
        patch.astype(np.float32), M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return shifted, (dx, dy)

def normalize_patch(patch, eps=1e-6):
    p = patch.astype(np.float32)
    p = p - np.min(p)
    s = float(p.sum())
    if s <= eps:
        return None
    return p / s

#5 -- Estimate PSF from aligned patches. Filter
def patch_metrics(patch, eps=1e-12):
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    s = float(patch.sum()) + eps
    cx = float((patch * xx).sum() / s)
    cy = float((patch * yy).sum() / s)

    peak = float(patch[h//2, w//2])
    peak_any = float(patch.max())

    dx = xx - cx
    dy = yy - cy
    cov_xx = float((patch * dx * dx).sum() / s)
    cov_yy = float((patch * dy * dy).sum() / s)
    cov_xy = float((patch * dx * dy).sum() / s)

    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    evals = np.linalg.eigvalsh(cov)
    l1, l2 = float(evals[1]), float(evals[0])
    ecc = 0.0 if l1 < eps else 1.0 - (l2 / l1)

    return {
        "cx": cx, "cy": cy,
        "peak_center": peak,
        "peak_any": peak_any,
        "l1": l1, "l2": l2,
        "ecc": ecc
    }

def filter_aligned_patches(aligned, max_center_offset_px=0.60,
                           peak_center_min=1e-7,
                           ecc_max=0.95,
                           size_l1_max=200.0):
    kept = []
    metrics = []
    h, w = aligned[0].shape if aligned else (0, 0)
    tx, ty = (w - 1) / 2.0, (h - 1) / 2.0

    for p in aligned:
        m = patch_metrics(p)
        center_offset = np.hypot(m["cx"] - tx, m["cy"] - ty)

        if center_offset > max_center_offset_px:
            continue
        if m["peak_center"] < peak_center_min:
            continue
        if m["ecc"] > ecc_max:
            continue
        if m["l1"] > size_l1_max:
            continue

        kept.append(p)
        metrics.append(m)

    return kept, metrics

def psf_median_stack(patches):
    P = np.stack(patches, axis=0).astype(np.float32)
    psf = np.median(P, axis=0)
    psf = np.maximum(psf, 0)
    psf /= (psf.sum() + 1e-12)
    return psf

def crop_center(img, new_size):
    r = new_size // 2
    h, w = img.shape
    cy, cx = h//2, w//2
    cropped = img[cy-r:cy+r+1, cx-r:cx+r+1]
    cropped = np.maximum(cropped, 0)
    return cropped / (cropped.sum() + 1e-12)

def bin_candidates_by_region(kept_patches, kept_meta, H, W, nx=4, ny=4):
    bins = {(ix, iy): [] for ix in range(nx) for iy in range(ny)}
    for patch, d in zip(kept_patches, kept_meta):
        ix = min(nx - 1, int(nx * d["x"] / W))
        iy = min(ny - 1, int(ny * d["y"] / H))
        bins[(ix, iy)].append(patch)
    return bins

def estimate_psf_per_bin(bins, method="trimmed", trim_frac=0.1, min_patches=30):
    psfs = {}
    for key, plist in bins.items():
        if len(plist) < min_patches:
            psfs[key] = None
            continue
        if method == "median":
            psfs[key] = psf_median_stack(plist)
    return psfs

#6. Results and visualizations
def radial_profile(psf, nbins=50):
    h, w = psf.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    rmax = rr.max()
    bins = np.linspace(0, rmax, nbins + 1)
    rmid = 0.5 * (bins[:-1] + bins[1:])

    prof = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.float64)

    for i in range(nbins):
        m = (rr >= bins[i]) & (rr < bins[i+1])
        counts[i] = m.sum()
        prof[i] = psf[m].mean() if counts[i] > 0 else np.nan

    return rmid, prof

def fwhm_proxy_from_moments(psf, eps=1e-12):
    p = np.maximum(psf.astype(np.float64), 0)
    p /= (p.sum() + eps)
    h, w = p.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (p * xx).sum()
    cy = (p * yy).sum()
    dx = xx - cx
    dy = yy - cy
    varx = (p * dx * dx).sum()
    vary = (p * dy * dy).sum()
    sigma_eq = np.sqrt(0.5 * (varx + vary))
    return 2.355 * sigma_eq, float(varx), float(vary)

def show_global_psf(psf):
    fwhm, varx, vary = fwhm_proxy_from_moments(psf)
    r, prof = radial_profile(psf, nbins=60)

    plt.figure()
    plt.title(f"Global PSF (sum=1), FWHM~{fwhm:.2f}px")
    plt.imshow(psf, cmap="gray")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title("Radial profile (mean intensity vs radius)")
    plt.plot(r, prof)
    plt.xlabel("radius (px)")
    plt.ylabel("mean intensity")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.show()

def show_psf_grid_and_heatmap(psfs_by_bin, nx=4, ny=4):
    fwhm_map = np.full((ny, nx), np.nan, dtype=np.float64)

    for (ix, iy), psf_bin in psfs_by_bin.items():
        if psf_bin is None:
            continue
        fwhm, _, _ = fwhm_proxy_from_moments(psf_bin)
        fwhm_map[iy, ix] = fwhm

    plt.figure(figsize=(10, 10))
    for iy in range(ny):
        for ix in range(nx):
            ax = plt.subplot(ny, nx, iy*nx + ix + 1)
            psf_bin = psfs_by_bin.get((ix, iy), None)
            if psf_bin is None:
                ax.text(0.5, 0.5, "None", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.imshow(psf_bin, cmap="gray")
            fwhm = fwhm_map[iy, ix]
            ax.set_title(f"({ix},{iy})\nFWHM~{fwhm:.2f}")
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("PSF width heatmap (FWHM proxy, px)")
    plt.imshow(fwhm_map, cmap="gray")
    plt.colorbar(label="FWHM (px)")
    plt.xticks(range(nx), [str(i) for i in range(nx)])
    plt.yticks(range(ny), [str(i) for i in range(ny)])
    plt.xlabel("bin ix")
    plt.ylabel("bin iy")
    plt.show()

def summarize_counts(n_detected, n_extracted, n_aligned, n_kept):
    unused = n_detected - n_kept
    frac_used = 0 if n_detected == 0 else (n_kept / n_detected)
    print(f"Detected candidates: {n_detected}")
    print(f"Extracted patches:   {n_extracted}")
    print(f"Aligned patches:     {n_aligned}")
    print(f"Kept for PSF:        {n_kept}")
    print(f"Unused point sources:{unused}  (used fraction={frac_used:.3f})")


def make_point_source_mask(shape_hw, points_xy, radius_px=12):
    """
    shape_hw: (H, W)
    points_xy: list of (x, y)
    Returns uint8 mask in {0,255}.
    """
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x, y) in points_xy:
        cv2.circle(mask, (int(x), int(y)), int(radius_px), 255, thickness=-1)
    return mask

def save_verification_images(out_dir, base_name, raw_gray, points_xy, radius_px=12):
    """
    Saves:
      - raw overlay (circles)
      - masked raw (raw * mask)
      - mask itself
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) overlay
    overlay = cv2.cvtColor(raw_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (x, y) in points_xy:
        cv2.circle(overlay, (int(x), int(y)), int(radius_px), (0, 0, 255), thickness=2)

    # 2) mask + masked image
    mask = make_point_source_mask(raw_gray.shape, points_xy, radius_px=radius_px)
    masked = cv2.bitwise_and(raw_gray.astype(np.uint8), raw_gray.astype(np.uint8), mask=mask)

    cv2.imwrite(os.path.join(out_dir, f"{base_name}_raw_overlay.png"), overlay)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_mask.png"), mask)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_masked.png"), masked)

#=======================
#Runtime Loop
PATCH_SIZE = 31

# Optional: cap how many per-image verification plots to show (None = unlimited)
MAX_VERIFY_SHOW = None
verify_shown = 0

all_aligned = []
all_meta = []

counts = {
    "images_total": len(cr2_files),
    "images_loaded": 0,
    "images_rejected": 0,
    "candidates": 0,
    "extracted": 0,
    "aligned": 0,
    "kept": 0,
    "images_with_candidates": 0,
    "images_with_kept": 0,
}

for i, path in enumerate(cr2_files):
    print(f"\n[{i+1}/{len(cr2_files)}] Processing {path}")

    # ---- load ----
    try:
        bgr = load_cr2_to_bgr(path)
        counts["images_loaded"] += 1
    except Exception as e:
        print("  LOAD FAILED:", e)
        counts["images_rejected"] += 1
        continue

    # ---- preprocess ----
    pre = preprocessing(bgr)
    orig_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if pre is None:
        print("  Rejected at preprocessing")
        counts["images_rejected"] += 1
        continue

    # ---- detect candidates ----
    cands, dog = detect_candidate_point_sources(
        pre,
        patch_size=PATCH_SIZE,
        snr_thresh=3.0,
        renyi_max=2.8,
        ecc_max=0.90,
        min_peak=1.0
    )

    # ---- debug: why 0 candidates? ----
    if len(cands) == 0:
        dog_max = float(np.max(dog))
        dog_p999 = float(np.percentile(dog, 99.9))
        dog_p9999 = float(np.percentile(dog, 99.99))
        dog_med = float(np.median(dog))
        dog_mad = float(np.median(np.abs(dog - dog_med)) + 1e-6)
        est_thresh = dog_med + 3.0 * dog_mad

        pre_max = float(np.max(pre))
        pre_p999 = float(np.percentile(pre, 99.9))

        print(
            f"  0-cand stats: "
            f"pre_max={pre_max:.1f} pre_p99.9={pre_p999:.1f} | "
            f"dog_max={dog_max:.3f} dog_p99.9={dog_p999:.3f} dog_p99.99={dog_p9999:.3f} | "
            f"est_thresh~{est_thresh:.3f}"
        )
        continue

    counts["candidates"] += len(cands)
    counts["images_with_candidates"] += 1

    # ---- extract patches ----
    patches, meta = extract_patches(pre, cands, patch_size=PATCH_SIZE)
    counts["extracted"] += len(patches)

    # ---- center + normalize ----
    aligned = []
    aligned_meta = []

    for patch, d in zip(patches, meta):
        shifted, (dx, dy) = subpixel_center_and_shift(patch)
        norm = normalize_patch(shifted)
        if norm is None:
            continue

        aligned.append(norm)
        aligned_meta.append({**d, "file": path, "dx": float(dx), "dy": float(dy)})

    counts["aligned"] += len(aligned)

    if len(aligned) == 0:
        print("  No usable aligned patches")
        continue

    # ---- debug: why are we keeping 0? ----
    fail = {"center": 0, "peak": 0, "ecc": 0, "size": 0, "pass": 0}
    h, w = aligned[0].shape
    tx, ty = (w - 1) / 2.0, (h - 1) / 2.0

    for p in aligned:
        m = patch_metrics(p)
        center_offset = np.hypot(m["cx"] - tx, m["cy"] - ty)

        if center_offset > 0.60:
            fail["center"] += 1
            continue
        if m["peak_center"] < 1e-7:
            fail["peak"] += 1
            continue
        if m["ecc"] > 0.95:
            fail["ecc"] += 1
            continue
        if m["l1"] > 200.0:
            fail["size"] += 1
            continue
        fail["pass"] += 1

    # ---- filter aligned patches ----
    kept, _ = filter_aligned_patches(aligned)
    print(" Filter Fails:", fail)
    counts["kept"] += len(kept)

    if len(kept) > 0:
        counts["images_with_kept"] += 1

    # rebuild kept_meta to stay aligned
    kept_meta = []
    h, w = aligned[0].shape
    tx, ty = (w - 1) / 2.0, (h - 1) / 2.0

    for p, md in zip(aligned, aligned_meta):
        m = patch_metrics(p)
        if np.hypot(m["cx"] - tx, m["cy"] - ty) > 0.60:
            continue
        if m["peak_center"] < 1e-7:
            continue
        if m["ecc"] > 0.95:
            continue
        if m["l1"] > 200.0:
            continue
        kept_meta.append(md)

    # ============================
    # MANUAL VERIFICATION (PLOTS) + (SAVED IMAGES)
    # Good image = has kept sources
    # ============================
    if len(kept_meta) > 0:
        # coords used-for-PSF in THIS image
        kept_xy = [(d["x"], d["y"]) for d in kept_meta]
        xs = [x for x, y in kept_xy]
        ys = [y for x, y in kept_xy]

        # Display-scaled RAW and preprocessed
        og = orig_gray.copy()
        og = np.clip(og, 0, np.percentile(og, 99.9))

        vis = pre.copy()
        vis = np.clip(vis, 0, np.percentile(vis, 99.9))

        # Build neighborhood mask around kept sources (same radius as patch extraction)
        radius = PATCH_SIZE // 2
        mask = np.zeros(orig_gray.shape, dtype=np.uint8)
        for x, y in kept_xy:
            cv2.circle(mask, (int(x), int(y)), int(radius), 255, thickness=-1)

        # Masked RAW (uint8)
        og8 = (255 * (og / (og.max() + 1e-6))).astype(np.uint8)
        masked = cv2.bitwise_and(og8, og8, mask=mask)

        # ---- show plots for every good image (optionally capped) ----
        if (MAX_VERIFY_SHOW is None) or (verify_shown < MAX_VERIFY_SHOW):
            verify_shown += 1
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].imshow(og, cmap="gray")
            axs[0].scatter(xs, ys, s=60, facecolors="none", edgecolors="r")
            axs[0].set_title(f"RAW + kept ({len(kept_meta)})")
            axs[0].set_axis_off()

            axs[1].imshow(vis, cmap="gray")
            axs[1].scatter(xs, ys, s=60, facecolors="none", edgecolors="r")
            axs[1].set_title("Preprocessed + kept")
            axs[1].set_axis_off()

            axs[2].imshow(masked, cmap="gray", vmin=0, vmax=np.percentile(masked[masked > 0], 99) if np.any(masked > 0) else 1)
            axs[2].set_title("RAW masked (kept neighborhoods)")
            axs[2].set_axis_off()

            plt.tight_layout()
            plt.show()
            # If running as a script and you want it to auto-advance, replace plt.show() with:
            # plt.show(block=False); plt.pause(0.5); plt.close(fig)

        # ---- also save raw_overlay/mask/masked for offline review ----
        base = os.path.splitext(os.path.basename(path))[0]
        save_verification_images(
            out_dir="verify_point_sources",
            base_name=base,
            raw_gray=og8,
            points_xy=kept_xy,
            radius_px=radius
        )

    all_aligned.extend(kept)
    all_meta.extend(kept_meta)

    print(f"  candidates={len(cands)}, extracted={len(patches)}, aligned={len(aligned)}, kept={len(kept)}")

# ===============================
# SUMMARY
print("\n=== SUMMARY ===")
summarize_counts(counts["candidates"], counts["extracted"], counts["aligned"], len(all_aligned))

rejected_frac = 0.0 if counts["images_total"] == 0 else counts["images_rejected"] / counts["images_total"]
kept_image_frac = 0.0 if counts["images_loaded"] == 0 else counts["images_with_kept"] / counts["images_loaded"]
cand_image_frac = 0.0 if counts["images_loaded"] == 0 else counts["images_with_candidates"] / counts["images_loaded"]

print(f"Images total:         {counts['images_total']}")
print(f"Images loaded:        {counts['images_loaded']}")
print(f"Images rejected:      {counts['images_rejected']}  (rejected fraction={rejected_frac:.3f})")
print(f"Images w/ candidates: {counts['images_with_candidates']}  (candidate fraction={cand_image_frac:.3f})")
print(f"Images w/ kept:       {counts['images_with_kept']}  (kept-image fraction={kept_image_frac:.3f})")

print("Total kept patches across all images:", len(all_aligned))
# ===============================


# ===============================
#PSF ESTIMATION VISUALIZATION
if len(all_aligned) == 0:
    print("No kept patches -> cannot estimate PSF.")
else:
    # 1) Global PSF via median stack
    global_psf = psf_median_stack(all_aligned)

    # 2) Optional: crop to a smaller core for easier viewing (e.g., 21x21)
    core_size = 21  # must be odd and <= PATCH_SIZE
    psf_core = crop_center(global_psf, core_size)

    # 3) Plot global PSF and radial profile
    print(f"\nPSF estimated from {len(all_aligned)} patches.")
    show_global_psf(psf_core)

    # 4) Also plot the *raw* global (uncropped) PSF for completeness
    plt.figure()
    plt.title("Global PSF (full patch)")
    plt.imshow(global_psf, cmap="gray")
    plt.colorbar()
    plt.show()

    # 5) Quick sanity: show a few kept patches (random sample)
    k = min(9, len(all_aligned))
    idx = np.random.choice(len(all_aligned), size=k, replace=False)

    plt.figure(figsize=(8, 8))
    for j, ii in enumerate(idx):
        ax = plt.subplot(3, 3, j + 1)
        ax.imshow(all_aligned[ii], cmap="gray")
        ax.set_title(f"patch {ii}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()
#=================================================
