import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d, median_filter


# ============================================================
# USER SETTINGS
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\TLP-001\Documents\GitHub\Project_1_Code\image_proccesing\image.png"
)

# Choose the experimental panel:
# "50", "75", "100", or "full"
PANEL = "100"

# Approximate crop coordinates for the left gray streak in each panel.
# Format: (x_min, x_max, y_min, y_max)
#
# Adjust these values slightly if the crop does not contain the complete
# gray foam region.
PANEL_CROPS = {
    "50": (105, 205, 120, 480),
    "75": (370, 470, 120, 480),
    "100": (635, 735, 120, 480),
    "full": None,
}

# Expand the selected panel crop by these margins (in pixels).
# This keeps one selected place at a time, but makes that crop larger.
CROP_PADDING_X = 3
CROP_PADDING_Y = 40

# Scale-bar measurements from the supplied image.
#
# The horizontal ruler corresponds to 1 mm.
# The vertical ruler corresponds to 1 ns.
#
# Replace these approximate pixel lengths with exact measurements
# for more accurate physical calibration.
HORIZONTAL_SCALE_PIXELS = 126.0
HORIZONTAL_SCALE_MM = 1.0

VERTICAL_SCALE_PIXELS = 201.0
VERTICAL_SCALE_NS = 1.0

# Threshold used to identify the gray/dark foam.
#
# Increase this value to include lighter gray pixels.
# Decrease it to keep only darker pixels.
THRESHOLD_VALUE = 185

# Ignore rows with fewer than this number of detected gray pixels.
MINIMUM_ROW_PIXELS = 10

# For the very beginning of the gray region, allow thinner rows so the
# detected contour starts at the growth onset.
MINIMUM_START_ROW_PIXELS = 3

# Percentiles used instead of the absolute first and last dark pixel.
# This prevents isolated dots from creating a very large false radius.
LEFT_PERCENTILE = 10
RIGHT_PERCENTILE = 90

# Ignore a small strip at the bottom to avoid the ruler/scale bar.
IGNORE_BOTTOM_ROWS = 25

# Image-processing settings.
BLUR_KERNEL_SIZE = 5
MORPHOLOGY_KERNEL_SIZE = 3
MORPHOLOGY_OPEN_ITERATIONS = 1
MORPHOLOGY_CLOSE_ITERATIONS = 2

# Smoothing settings.
MEDIAN_FILTER_SIZE =  None
RADIUS_GAUSSIAN_SIGMA = 2.8
CENTER_GAUSSIAN_SIGMA = 8

# If True, radius is forced to increase from bottom to top.
# Set to False to let the contour follow the measured shape at the ends.
ENFORCE_MONOTONIC_RADIUS = True

# After the radius reaches this fraction of its maximum value
# (in bottom-to-top direction), keep it constant above that point.
PLATEAU_FRACTION_OF_MAX_RADIUS = 0.96

# Growth-onset detection in bottom-to-top order.
# The start is the first row where radius exceeds baseline by this amount
# for a sustained number of rows.
START_RADIUS_DELTA_PIXELS = 2.0
START_SUSTAINED_ROWS = 6

# Reference-like dashed curve styling for position-vs-time plot:
# start at zero radius, then smooth growth toward plateau.
USE_REFERENCE_STYLE_PLOT = True
PLOT_RADIUS_SMOOTH_SIGMA = 2.0

# If True, build final contours directly from left/right edges.
# This avoids shape locking from a symmetric radius model.
USE_DIRECT_EDGE_TRACKING = False

# If True, the contour is symmetric around one fixed center.
# If False, the center is allowed to move slowly with time.
USE_FIXED_CENTER = True

# Save results.
SAVE_CSV = True
SAVE_FIGURES = True


# ============================================================
# CORRECT MEDIAN-FILTER VALUE
# ============================================================

# Python requires an integer. The line above is written descriptively;
# this is the actual numerical value used by the program.
MEDIAN_FILTER_SIZE = 7


# ============================================================
# CHECK USER SETTINGS
# ============================================================

if PANEL not in PANEL_CROPS:
    raise ValueError(
        f"Unknown panel '{PANEL}'. Choose '50', '75', '100', or 'full'."
    )

if BLUR_KERNEL_SIZE % 2 == 0:
    raise ValueError("BLUR_KERNEL_SIZE must be an odd integer.")

if MEDIAN_FILTER_SIZE % 2 == 0:
    raise ValueError("MEDIAN_FILTER_SIZE must be an odd integer.")


# ============================================================
# PHYSICAL CALIBRATION
# ============================================================

mm_per_pixel = HORIZONTAL_SCALE_MM / HORIZONTAL_SCALE_PIXELS
ns_per_pixel = VERTICAL_SCALE_NS / VERTICAL_SCALE_PIXELS

print("Calibration")
print("-----------")
print(f"Spatial calibration: {mm_per_pixel:.7f} mm/pixel")
print(f"Temporal calibration: {ns_per_pixel:.7f} ns/pixel")
print()


# ============================================================
# LOAD THE IMAGE
# ============================================================

image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise FileNotFoundError(
        f"Could not open the image:\n{IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

original_height, original_width = image_rgb.shape[:2]

print(f"Original image size: {original_width} × {original_height} pixels")


# ============================================================
# CROP THE SELECTED FOAM STREAK
# ============================================================

crop_spec = PANEL_CROPS[PANEL]

if crop_spec is None:
    crop_rgb = image_rgb.copy()
else:
    x_min, x_max, y_min, y_max = crop_spec

    x_min = max(0, x_min - CROP_PADDING_X)
    x_max = min(original_width, x_max + CROP_PADDING_X)
    y_min = max(0, y_min - CROP_PADDING_Y)
    y_max = min(original_height, y_max + CROP_PADDING_Y)

    if (
        x_min < 0
        or y_min < 0
        or x_max > original_width
        or y_max > original_height
    ):
        raise ValueError(
            "The crop coordinates extend outside the image."
        )

    crop_rgb = image_rgb[y_min:y_max, x_min:x_max]

crop_height, crop_width = crop_rgb.shape[:2]

print(f"Selected panel: {PANEL} μm")
print(f"Crop size: {crop_width} × {crop_height} pixels")
print()


# ============================================================
# CONVERT TO GRAYSCALE
# ============================================================

crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)


# ============================================================
# SMOOTH THE IMAGE BEFORE THRESHOLDING
# ============================================================

blurred_gray = cv2.GaussianBlur(
    crop_gray,
    (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE),
    sigmaX=0,
    sigmaY=0,
)


# ============================================================
# CREATE A BINARY MASK OF THE GRAY/DARK FOAM
# ============================================================

# THRESH_BINARY_INV means:
#
# dark/gray pixels   -> white in the mask
# bright background  -> black in the mask

_, binary_mask = cv2.threshold(
    blurred_gray,
    THRESHOLD_VALUE,
    255,
    cv2.THRESH_BINARY_INV,
)


# ============================================================
# CLEAN THE BINARY MASK
# ============================================================

morphology_kernel = np.ones(
    (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE),
    dtype=np.uint8,
)

# Remove small isolated noise.
binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_OPEN,
    morphology_kernel,
    iterations=MORPHOLOGY_OPEN_ITERATIONS,
)

# Fill small gaps inside the foam region.
binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_CLOSE,
    morphology_kernel,
    iterations=MORPHOLOGY_CLOSE_ITERATIONS,
)

# Keep only the largest connected dark region so side speckles
# and detached artifacts do not distort the contour width.
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    binary_mask,
    connectivity=8,
)

if num_labels > 1:
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component_label = 1 + int(np.argmax(component_areas))
    binary_mask = np.where(
        labels == largest_component_label,
        255,
        0,
    ).astype(np.uint8)

# Fill internal holes so the foam region is one solid white area.
external_contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

if external_contours:
    largest_external_contour = max(
        external_contours,
        key=cv2.contourArea,
    )
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(
        filled_mask,
        [largest_external_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )
    binary_mask = filled_mask


# ============================================================
# EXTRACT THE RAW LEFT AND RIGHT EDGES
# ============================================================

raw_rows = []
raw_left_edges = []
raw_right_edges = []
previous_segment_center = None

row_start = 0
row_end = crop_height - IGNORE_BOTTOM_ROWS

if row_end <= row_start:
    raise ValueError(
        "IGNORE_BOTTOM_ROWS is too large for the selected crop height."
    )

for row_index in range(row_start, row_end):

    detected_columns = np.where(binary_mask[row_index] > 0)[0]

    minimum_pixels_this_row = (
        MINIMUM_START_ROW_PIXELS
        if previous_segment_center is None
        else MINIMUM_ROW_PIXELS
    )

    if len(detected_columns) < minimum_pixels_this_row:
        continue

    split_points = np.where(np.diff(detected_columns) > 1)[0] + 1
    contiguous_segments = np.split(detected_columns, split_points)
    contiguous_segments = [
        segment
        for segment in contiguous_segments
        if len(segment) >= minimum_pixels_this_row
    ]

    if not contiguous_segments:
        continue

    if previous_segment_center is None:
        selected_segment = max(contiguous_segments, key=len)
    else:
        selected_segment = min(
            contiguous_segments,
            key=lambda segment: abs(segment.mean() - previous_segment_center),
        )

    previous_segment_center = float(selected_segment.mean())

    # Percentiles are used rather than min/max because isolated dark
    # pixels can otherwise produce an unrealistically large radius.
    left_edge = np.percentile(
        selected_segment,
        LEFT_PERCENTILE,
    )

    right_edge = np.percentile(
        selected_segment,
        RIGHT_PERCENTILE,
    )

    if right_edge <= left_edge:
        continue

    raw_rows.append(row_index)
    raw_left_edges.append(left_edge)
    raw_right_edges.append(right_edge)


raw_rows = np.asarray(raw_rows, dtype=float)
raw_left_edges = np.asarray(raw_left_edges, dtype=float)
raw_right_edges = np.asarray(raw_right_edges, dtype=float)

if len(raw_rows) == 0:
    raise RuntimeError(
        "No foam region was detected.\n"
        "Try increasing THRESHOLD_VALUE or changing the crop."
    )

print(f"Detected valid rows: {len(raw_rows)}")


# ============================================================
# CALCULATE RAW CENTER AND RAW RADIUS
# ============================================================

raw_center = 0.5 * (
    raw_left_edges + raw_right_edges
)

raw_radius = 0.5 * (
    raw_right_edges - raw_left_edges
)


# ============================================================
# REMOVE INVALID OR VERY SMALL RADII
# ============================================================

valid_radius_mask = np.isfinite(raw_radius) & (raw_radius > 1.0)

raw_rows = raw_rows[valid_radius_mask]
raw_left_edges = raw_left_edges[valid_radius_mask]
raw_right_edges = raw_right_edges[valid_radius_mask]
raw_center = raw_center[valid_radius_mask]
raw_radius = raw_radius[valid_radius_mask]

if len(raw_rows) < 5:
    raise RuntimeError(
        "Too few valid contour rows remain after filtering."
    )


# ============================================================
# REMOVE LARGE WIDTH OUTLIERS
# ============================================================

median_radius = np.median(raw_radius)

radius_absolute_deviation = np.abs(
    raw_radius - median_radius
)

median_absolute_deviation = np.median(
    radius_absolute_deviation
)

if median_absolute_deviation > 0:

    robust_z_score = (
        radius_absolute_deviation
        / median_absolute_deviation
    )

    valid_outlier_mask = robust_z_score < 6.0

    raw_rows = raw_rows[valid_outlier_mask]
    raw_left_edges = raw_left_edges[valid_outlier_mask]
    raw_right_edges = raw_right_edges[valid_outlier_mask]
    raw_center = raw_center[valid_outlier_mask]
    raw_radius = raw_radius[valid_outlier_mask]


# ============================================================
# INTERPOLATE OVER MISSING ROWS
# ============================================================

first_valid_row = int(np.ceil(raw_rows.min()))
last_valid_row = int(np.floor(raw_rows.max()))

all_rows = np.arange(
    first_valid_row,
    last_valid_row + 1,
    dtype=float,
)

left_interpolated = np.interp(
    all_rows,
    raw_rows,
    raw_left_edges,
)

right_interpolated = np.interp(
    all_rows,
    raw_rows,
    raw_right_edges,
)

center_interpolated = 0.5 * (left_interpolated + right_interpolated)
radius_interpolated = 0.5 * (right_interpolated - left_interpolated)


# ============================================================
# SMOOTH CENTER AND RADIUS
# ============================================================

left_median = median_filter(
    left_interpolated,
    size=MEDIAN_FILTER_SIZE,
    mode="nearest",
)

right_median = median_filter(
    right_interpolated,
    size=MEDIAN_FILTER_SIZE,
    mode="nearest",
)

left_smooth = gaussian_filter1d(
    left_median,
    sigma=RADIUS_GAUSSIAN_SIGMA,
    mode="nearest",
)

right_smooth = gaussian_filter1d(
    right_median,
    sigma=RADIUS_GAUSSIAN_SIGMA,
    mode="nearest",
)

center_median = median_filter(
    center_interpolated,
    size=MEDIAN_FILTER_SIZE,
    mode="nearest",
)

center_smooth = gaussian_filter1d(
    center_median,
    sigma=CENTER_GAUSSIAN_SIGMA,
    mode="nearest",
)

radius_smooth = 0.5 * np.maximum(
    right_smooth - left_smooth,
    1.0,
)


# ============================================================
# ENFORCE A RADIUS THAT ONLY INCREASES FROM BOTTOM TO TOP
# ============================================================

# Image row coordinates increase downward:
#
# top of image    -> small row number
# bottom of image -> large row number
#
# The desired behavior is:
#
# moving from bottom to top, the radius can only increase.
#
# Therefore:
# 1. Reverse the radius array, so its order becomes bottom-to-top.
# 2. Apply cumulative maximum.
# 3. Reverse the result back to normal image order.

if ENFORCE_MONOTONIC_RADIUS:

    radius_bottom_to_top = radius_smooth[::-1]

    radius_bottom_to_top_monotonic = np.maximum.accumulate(
        radius_bottom_to_top
    )

    max_radius_bt = float(np.max(radius_bottom_to_top_monotonic))

    if max_radius_bt > 0:
        plateau_mask = (
            radius_bottom_to_top_monotonic
            >= PLATEAU_FRACTION_OF_MAX_RADIUS * max_radius_bt
        )

        if np.any(plateau_mask):
            plateau_start_index = int(np.argmax(plateau_mask))
            plateau_value = radius_bottom_to_top_monotonic[plateau_start_index]
            radius_bottom_to_top_monotonic[plateau_start_index:] = plateau_value

    radius_final = radius_bottom_to_top_monotonic[::-1]

else:

    radius_final = radius_smooth


# ============================================================
# CHOOSE THE CONTOUR CENTER
# ============================================================

if USE_FIXED_CENTER:

    # One constant center gives a clean, symmetric contour.
    center_value = np.median(center_smooth)

    center_final = np.full_like(
        center_smooth,
        center_value,
    )

else:

    # The center may move slowly while the radius remains monotonic.
    center_final = center_smooth


# ============================================================
# CREATE FINAL LEFT AND RIGHT CONTOURS
# ============================================================

if USE_DIRECT_EDGE_TRACKING:

    left_contour_pixels = left_smooth
    right_contour_pixels = right_smooth

    minimum_gap_pixels = 2.0
    right_contour_pixels = np.maximum(
        right_contour_pixels,
        left_contour_pixels + minimum_gap_pixels,
    )

    center_final = 0.5 * (left_contour_pixels + right_contour_pixels)
    radius_final = 0.5 * (right_contour_pixels - left_contour_pixels)

else:

    left_contour_pixels = (
        center_final - radius_final
    )

    right_contour_pixels = (
        center_final + radius_final
    )

# Keep contours inside the crop.
left_contour_pixels = np.clip(
    left_contour_pixels,
    0,
    crop_width - 1,
)

right_contour_pixels = np.clip(
    right_contour_pixels,
    0,
    crop_width - 1,
)


# ============================================================
# CONVERT TO PHYSICAL UNITS
# ============================================================

# Vertical image coordinate becomes time.
time_ns = (
    all_rows - all_rows.min()
) * ns_per_pixel

# To make time increase from the bottom toward the top, use:
time_bottom_to_top_ns = (
    all_rows.max() - all_rows
) * ns_per_pixel

# Convert horizontal image coordinates to millimeters.
left_position_mm = (
    left_contour_pixels * mm_per_pixel
)

right_position_mm = (
    right_contour_pixels * mm_per_pixel
)

center_position_mm = (
    center_final * mm_per_pixel
)

radius_mm = (
    radius_final * mm_per_pixel
)

# Detect growth onset in bottom-to-top order and use it as time zero.
plot_order = np.argsort(time_bottom_to_top_ns)

radius_bottom_to_top_pixels = radius_final[plot_order]
time_bottom_to_top_sorted_ns = time_bottom_to_top_ns[plot_order]

baseline_count = max(8, len(radius_bottom_to_top_pixels) // 12)
baseline_count = min(baseline_count, len(radius_bottom_to_top_pixels))

baseline_radius_pixels = np.median(
    radius_bottom_to_top_pixels[:baseline_count]
)

start_threshold_pixels = baseline_radius_pixels + START_RADIUS_DELTA_PIXELS
is_above_start_threshold = (
    radius_bottom_to_top_pixels >= start_threshold_pixels
)

onset_index_bottom_to_top = 0

if len(radius_bottom_to_top_pixels) >= START_SUSTAINED_ROWS:
    for index in range(
        0,
        len(radius_bottom_to_top_pixels) - START_SUSTAINED_ROWS + 1,
    ):
        if np.all(
            is_above_start_threshold[
                index:index + START_SUSTAINED_ROWS
            ]
        ):
            onset_index_bottom_to_top = index
            break

onset_index_full = int(plot_order[onset_index_bottom_to_top])

# Center the spatial coordinate around y = 0.
reference_center_mm = np.median(
    center_position_mm
)

left_position_centered_mm = (
    left_position_mm - reference_center_mm
)

right_position_centered_mm = (
    right_position_mm - reference_center_mm
)


# ============================================================
# PLOT 1: ORIGINAL CROP
# ============================================================

plt.figure(figsize=(6, 9))

plt.imshow(
    crop_rgb,
    origin="upper",
)

plt.xlabel("Horizontal pixel", fontsize=14)
plt.ylabel("Vertical pixel", fontsize=14)

plt.title(
    f"{PANEL} μm foam: cropped streak image",
    fontsize=15,
)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        f"foam_{PANEL}um_crop.png",
        dpi=300,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# PLOT 2: BINARY MASK
# ============================================================

plt.figure(figsize=(6, 9))

plt.imshow(
    binary_mask,
    cmap="gray",
    origin="upper",
)

plt.xlabel("Horizontal pixel", fontsize=14)
plt.ylabel("Vertical pixel", fontsize=14)

plt.title(
    f"{PANEL} μm foam: threshold mask",
    fontsize=15,
)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        f"foam_{PANEL}um_mask.png",
        dpi=300,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# PLOT 3: FINAL MONOTONIC CONTOUR OVER THE IMAGE
# ============================================================

plt.figure(figsize=(6, 9))

plt.imshow(
    crop_rgb,
    origin="upper",
)

plt.plot(
    left_contour_pixels,
    all_rows,
    "k--",
    linewidth=2.5,
    label="Monotonic contour",
)

plt.plot(
    right_contour_pixels,
    all_rows,
    "k--",
    linewidth=2.5,
)

start_row_pixel = all_rows[onset_index_full]
start_center_pixel = 0.5 * (
    left_contour_pixels[onset_index_full]
    + right_contour_pixels[onset_index_full]
)

plt.scatter(
    [start_center_pixel],
    [start_row_pixel],
    color="red",
    s=40,
    zorder=5,
    label="Growth start",
)

plt.xlim(0, crop_width - 1)
plt.ylim(crop_height - 1, 0)

plt.xlabel("Position [pixel]", fontsize=14)
plt.ylabel("Image row [pixel]", fontsize=14)

plt.title(
    f"{PANEL} μm foam: expanding gray-region contour",
    fontsize=15,
)

plt.legend(fontsize=12)
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        f"foam_{PANEL}um_contour_overlay.png",
        dpi=300,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# PLOT 4: PHYSICAL POSITION VERSUS TIME
# ============================================================

# Sort by increasing bottom-to-top time for plotting and start at onset.
plot_time_ns = (
    time_bottom_to_top_sorted_ns[onset_index_bottom_to_top:]
    - time_bottom_to_top_sorted_ns[onset_index_bottom_to_top]
)

plot_left_raw_mm = left_position_centered_mm[plot_order][onset_index_bottom_to_top:]
plot_right_raw_mm = right_position_centered_mm[plot_order][onset_index_bottom_to_top:]

if USE_REFERENCE_STYLE_PLOT:

    plot_radius_raw_mm = 0.5 * (plot_right_raw_mm - plot_left_raw_mm)
    onset_radius_mm = float(plot_radius_raw_mm[0])

    # Rebase the radius so the dashed boundaries start at y=0 at onset.
    plot_radius_mm = np.maximum(plot_radius_raw_mm - onset_radius_mm, 0.0)

    plot_radius_mm = gaussian_filter1d(
        plot_radius_mm,
        sigma=PLOT_RADIUS_SMOOTH_SIGMA,
        mode="nearest",
    )

    plot_left_mm = -plot_radius_mm
    plot_right_mm = plot_radius_mm

else:

    plot_left_mm = plot_left_raw_mm
    plot_right_mm = plot_right_raw_mm
    plot_radius_mm = 0.5 * (plot_right_mm - plot_left_mm)

plt.figure(figsize=(9, 6))

plt.plot(
    plot_time_ns,
    plot_left_mm,
    "k--",
    linewidth=2.5,
    label="Left contour",
)

plt.plot(
    plot_time_ns,
    plot_right_mm,
    "k--",
    linewidth=2.5,
    label="Right contour",
)

plt.fill_between(
    plot_time_ns,
    plot_left_mm,
    plot_right_mm,
    alpha=0.15,
)

plt.xlabel(r"$t$ [ns]", fontsize=17)
plt.ylabel(r"Position $y$ [mm]", fontsize=17)

plt.title(
    f"{PANEL} μm foam: monotonic expanding radius",
    fontsize=16,
)

plt.tick_params(
    axis="both",
    labelsize=14,
)

plt.grid(alpha=0.25)
plt.legend(fontsize=12)
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        f"foam_{PANEL}um_position_vs_time.png",
        dpi=300,
        bbox_inches="tight",
    )

plt.show()


# ============================================================
# PLOT 5: RADIUS VERSUS TIME
# ============================================================

plot_radius_mm = 0.5 * (plot_right_mm - plot_left_mm)

# plt.figure(figsize=(9, 6))

# plt.plot(
#     plot_time_ns,
#     plot_radius_mm,
#     "k--",
#     linewidth=2.5,
# )

# plt.xlabel(r"$t$ [ns]", fontsize=17)
# plt.ylabel(r"Radius $R$ [mm]", fontsize=17)

# plt.title(
#     f"{PANEL} μm foam: radius increasing from bottom to top",
#     fontsize=16,
# )

# plt.tick_params(
#     axis="both",
#     labelsize=14,
# )

# plt.grid(alpha=0.25)
# plt.tight_layout()

# if SAVE_FIGURES:
#     plt.savefig(
#         f"foam_{PANEL}um_radius_vs_time.png",
#         dpi=300,
#         bbox_inches="tight",
#     )

# plt.show()


# ============================================================
# SAVE CONTOUR DATA TO CSV
# ============================================================

if SAVE_CSV:

    output_data = np.column_stack(
        (
            plot_time_ns,
            plot_left_mm,
            plot_right_mm,
            plot_radius_mm,
        )
    )

    output_filename = f"foam_contour_{PANEL}um.csv"

    np.savetxt(
        output_filename,
        output_data,
        delimiter=",",
        header=(
            "time_ns,"
            "left_position_mm,"
            "right_position_mm,"
            "radius_mm"
        ),
        comments="",
    )

    print()
    print(f"Saved contour data to: {output_filename}")


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("Contour summary")
print("----------------")
print(f"Panel thickness: {PANEL} μm")
print(f"Number of contour points: {len(all_rows)}")
print(f"Minimum radius: {np.min(radius_mm):.4f} mm")
print(f"Maximum radius: {np.max(radius_mm):.4f} mm")
print(
    "The radius has been constrained so that it only increases "
    "when moving from the bottom of the image toward the top."
)