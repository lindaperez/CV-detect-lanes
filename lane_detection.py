import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def build_stage_collage(
    stages: list[tuple[str, str, np.ndarray]],
    panel_width: int = 360,
    panel_height: int = 240,
) -> np.ndarray:
    """Create a tiled image with titles and short descriptions per stage."""
    header_h = 72
    columns = 3
    h_gap = 16
    v_gap = 28
    outer_margin = 24
    panels: list[np.ndarray] = []

    for title, reason, image in stages:
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()

        resized = cv2.resize(image_bgr, (panel_width, panel_height))
        ax_color = (0, 180, 0)
        cv2.line(resized, (24, panel_height - 24), (panel_width - 16, panel_height - 24), ax_color, 1)
        cv2.line(resized, (24, panel_height - 24), (24, 16), ax_color, 1)
        cv2.putText(
            resized,
            "(0,0)",
            (6, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            ax_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            resized,
            f"({panel_width - 1},0)",
            (panel_width - 96, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            ax_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            resized,
            f"(0,{panel_height - 1})",
            (6, panel_height - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            ax_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            resized,
            f"({panel_width - 1},{panel_height - 1})",
            (panel_width - 126, panel_height - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            ax_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(resized, "x", (panel_width - 12, panel_height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ax_color, 1, cv2.LINE_AA)
        cv2.putText(resized, "y", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ax_color, 1, cv2.LINE_AA)

        panel = np.full((panel_height + header_h, panel_width, 3), 255, dtype=np.uint8)
        panel[header_h:, :, :] = resized
        cv2.putText(panel, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, reason, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (60, 60, 60), 1, cv2.LINE_AA)
        panels.append(panel)

    blank_panel = np.full((panel_height + header_h, panel_width, 3), 255, dtype=np.uint8)
    rows: list[np.ndarray] = []
    for start in range(0, len(panels), columns):
        row_panels = panels[start:start + columns]
        if len(row_panels) < columns:
            row_panels.extend([blank_panel] * (columns - len(row_panels)))
        row = row_panels[0]
        for panel in row_panels[1:]:
            spacer = np.full((row.shape[0], h_gap, 3), 255, dtype=np.uint8)
            row = np.hstack((row, spacer, panel))
        rows.append(row)

    collage = rows[0]
    for row in rows[1:]:
        spacer = np.full((v_gap, collage.shape[1], 3), 255, dtype=np.uint8)
        collage = np.vstack((collage, spacer, row))

    return cv2.copyMakeBorder(
        collage,
        outer_margin,
        outer_margin,
        outer_margin,
        outer_margin,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def region_of_interest(img: np.ndarray) -> np.ndarray:
    """Mask image to focus on the road area where lanes usually appear."""
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array(
        [[
            (int(width * 0.1), height),
            (int(width * 0.45), int(height * 0.6)),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.9), height),
        ]],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def average_lane_line(
    image: np.ndarray, lines: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split and average Hough lines into left and right lane lines."""
    if lines is None:
        return None, None

    left_fits: list[tuple[float, float]] = []
    right_fits: list[tuple[float, float]] = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter nearly horizontal lines.
        if abs(slope) < 0.4:
            continue

        if slope < 0:
            left_fits.append((slope, intercept))
        else:
            right_fits.append((slope, intercept))

    left_line = make_line_points(image, np.mean(left_fits, axis=0)) if left_fits else None
    right_line = make_line_points(image, np.mean(right_fits, axis=0)) if right_fits else None

    return left_line, right_line


def make_line_points(image: np.ndarray, line_fit: np.ndarray) -> np.ndarray:
    """Convert slope/intercept line into two endpoints."""
    slope, intercept = line_fit
    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.6)

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2], dtype=np.int32)


def draw_lane_overlay(frame: np.ndarray, left_line: np.ndarray | None, right_line: np.ndarray | None) -> np.ndarray:
    """Draw detected left/right lane edges in red without lane fill."""
    result = frame.copy()
    height, width = frame.shape[:2]
    roi_polygon = np.array(
        [[
            (int(width * 0.1), height),
            (int(width * 0.45), int(height * 0.6)),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.9), height),
        ]],
        dtype=np.int32,
    )
    cv2.polylines(result, roi_polygon, isClosed=True, color=(0, 255, 255), thickness=1)

    if left_line is not None:
        cv2.line(
            result,
            (int(left_line[0]), int(left_line[1])),
            (int(left_line[2]), int(left_line[3])),
            (0, 0, 255),
            2,
        )
    if right_line is not None:
        cv2.line(
            result,
            (int(right_line[0]), int(right_line[1])),
            (int(right_line[2]), int(right_line[3])),
            (0, 0, 255),
            2,
        )

    return result


def evaluate_lane_pair(
    left_line: np.ndarray | None,
    right_line: np.ndarray | None,
    min_lane_width_px: int = 80,
) -> tuple[bool, int | None]:
    """Return geometric validity and estimated lane width (bottom x-distance)."""
    if left_line is None or right_line is None:
        return False, None

    left_bottom_x, _, left_top_x, _ = left_line
    right_bottom_x, _, right_top_x, _ = right_line
    lane_width = int(right_bottom_x - left_bottom_x)
    is_valid = (
        left_bottom_x < right_bottom_x
        and left_top_x < right_top_x
        and lane_width >= min_lane_width_px
    )
    return is_valid, lane_width


def smooth_lane_line(
    current_line: np.ndarray | None,
    previous_line: np.ndarray | None,
    alpha: float,
) -> np.ndarray | None:
    """Exponentially smooth lane endpoints to reduce frame-to-frame jitter."""
    if current_line is None:
        return None
    current = current_line.astype(np.float32)
    if previous_line is None:
        return current
    previous = previous_line.astype(np.float32)
    return alpha * current + (1.0 - alpha) * previous


def tuned_detection_params(
    width: int,
    height: int,
    camera_position: str,
    canny_low_override: int | None,
    canny_high_override: int | None,
    hough_threshold_override: int | None,
    hough_min_line_length_override: int | None,
    hough_max_line_gap_override: int | None,
) -> dict[str, int]:
    """Tune Canny/Hough values by resolution and camera mounting position."""
    base_area = 1280 * 720
    scale = float(np.sqrt(max(1, width * height) / base_area))

    camera_multipliers = {
        "low": {"canny": 0.9, "hough_thr": 0.9, "min_len": 0.9, "max_gap": 1.1},
        "mid": {"canny": 1.0, "hough_thr": 1.0, "min_len": 1.0, "max_gap": 1.0},
        "high": {"canny": 1.1, "hough_thr": 1.1, "min_len": 1.15, "max_gap": 0.9},
    }
    mult = camera_multipliers[camera_position]

    tuned_canny_low = int(np.clip(round(50 * scale * mult["canny"]), 20, 120))
    tuned_canny_high = int(np.clip(round(150 * scale * mult["canny"]), tuned_canny_low + 20, 255))
    tuned_hough_threshold = int(np.clip(round(30 * scale * mult["hough_thr"]), 15, 120))
    tuned_hough_min_line_length = int(np.clip(round(20 * scale * mult["min_len"]), 10, 300))
    tuned_hough_max_line_gap = int(np.clip(round(200 * scale * mult["max_gap"]), 30, 400))

    return {
        "canny_low": canny_low_override if canny_low_override is not None else tuned_canny_low,
        "canny_high": canny_high_override if canny_high_override is not None else tuned_canny_high,
        "hough_threshold": (
            hough_threshold_override if hough_threshold_override is not None else tuned_hough_threshold
        ),
        "hough_min_line_length": (
            hough_min_line_length_override
            if hough_min_line_length_override is not None
            else tuned_hough_min_line_length
        ),
        "hough_max_line_gap": (
            hough_max_line_gap_override if hough_max_line_gap_override is not None else tuned_hough_max_line_gap
        ),
    }


def detect_and_color_lanes(
    input_video: str,
    output_video: str | None = None,
    display: bool = True,
    outputs_dir: str = "outputs",
    use_adaptive_threshold: bool = False,
    frame_check_report: bool = False,
    camera_position: str = "mid",
    canny_low: int | None = None,
    canny_high: int | None = None,
    hough_threshold: int | None = None,
    hough_min_line_length: int | None = None,
    hough_max_line_gap: int | None = None,
) -> None:
    """Run lane detection on each frame and highlight lane area in red."""
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_video}")

    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)
    input_stem = Path(input_video).stem

    if output_video is None:
        output_video = str(outputs_path / f"{input_stem}_lanes.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    params = tuned_detection_params(
        width=width,
        height=height,
        camera_position=camera_position,
        canny_low_override=canny_low,
        canny_high_override=canny_high,
        hough_threshold_override=hough_threshold,
        hough_min_line_length_override=hough_min_line_length,
        hough_max_line_gap_override=hough_max_line_gap,
    )
    print(
        "Using detection params: "
        f"canny=({params['canny_low']},{params['canny_high']}), "
        f"hough_threshold={params['hough_threshold']}, "
        f"minLineLength={params['hough_min_line_length']}, "
        f"maxLineGap={params['hough_max_line_gap']}, "
        f"camera_position={camera_position}, resolution={width}x{height}"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {output_video}")

    snapshots_saved = False
    frame_index = -1
    total_frames = 0
    both_detected_count = 0
    valid_geometry_count = 0
    report_rows: list[dict[str, int | bool | str]] = []
    prev_left_line: np.ndarray | None = None
    prev_right_line: np.ndarray | None = None
    left_missed_frames = 0
    right_missed_frames = 0
    smoothing_alpha = 0.35
    max_hold_frames = 6

    def save_snapshot(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            print(f"Warning: failed to save snapshot: {path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            total_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if use_adaptive_threshold:
                adaptive = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )
                canny_input = adaptive
            else:
                adaptive = None
                canny_input = gray

            blur = cv2.GaussianBlur(canny_input, (5, 5), 0)
            edges = cv2.Canny(blur, params["canny_low"], params["canny_high"])

            roi_edges = region_of_interest(edges)
            lines = cv2.HoughLinesP(
                roi_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=params["hough_threshold"],
                minLineLength=params["hough_min_line_length"],
                maxLineGap=params["hough_max_line_gap"],
            )

            raw_left_line, raw_right_line = average_lane_line(frame, lines)

            smoothed_left = smooth_lane_line(raw_left_line, prev_left_line, smoothing_alpha)
            if smoothed_left is not None:
                left_line = smoothed_left
                prev_left_line = smoothed_left
                left_missed_frames = 0
            elif prev_left_line is not None and left_missed_frames < max_hold_frames:
                left_line = prev_left_line
                left_missed_frames += 1
            else:
                left_line = None
                prev_left_line = None
                left_missed_frames = 0

            smoothed_right = smooth_lane_line(raw_right_line, prev_right_line, smoothing_alpha)
            if smoothed_right is not None:
                right_line = smoothed_right
                prev_right_line = smoothed_right
                right_missed_frames = 0
            elif prev_right_line is not None and right_missed_frames < max_hold_frames:
                right_line = prev_right_line
                right_missed_frames += 1
            else:
                right_line = None
                prev_right_line = None
                right_missed_frames = 0

            result = draw_lane_overlay(frame, left_line, right_line)
            left_detected = left_line is not None
            right_detected = right_line is not None
            both_detected = left_detected and right_detected
            geometry_ok, lane_width = evaluate_lane_pair(left_line, right_line)

            if both_detected:
                both_detected_count += 1
            if geometry_ok:
                valid_geometry_count += 1
            if frame_check_report:
                report_rows.append(
                    {
                        "frame": frame_index,
                        "left_detected": left_detected,
                        "right_detected": right_detected,
                        "both_detected": both_detected,
                        "geometry_ok": geometry_ok,
                        "lane_width_px": "" if lane_width is None else lane_width,
                    }
                )

            if not snapshots_saved:
                save_snapshot(outputs_path / f"{input_stem}_gray.png", gray)
                if adaptive is not None:
                    save_snapshot(outputs_path / f"{input_stem}_adaptive_threshold.png", adaptive)
                save_snapshot(outputs_path / f"{input_stem}_edges.png", edges)
                save_snapshot(outputs_path / f"{input_stem}_roi.png", roi_edges)
                save_snapshot(outputs_path / f"{input_stem}_lane_overlay.png", result)

                stages: list[tuple[str, str, np.ndarray]] = [
                    ("Gray", "Removes color to simplify lane contrast", gray),
                ]
                if adaptive is not None:
                    stages.append(
                        ("Adaptive Threshold", "Handles uneven lighting with local thresholding", adaptive)
                    )
                stages.extend(
                    [
                        ("Edges (Canny)", "Finds strong intensity transitions", edges),
                        ("ROI Mask", "Keeps only road area likely to contain lanes", roi_edges),
                        ("Detected Lane", "Final lane area overlay on road frame", result),
                    ]
                )
                collage = build_stage_collage(stages)
                save_snapshot(outputs_path / f"{input_stem}_pipeline_overview.png", collage)
                snapshots_saved = True

            writer.write(result)

            if display:
                cv2.imshow("Lane Detection", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    if frame_check_report and report_rows:
        report_path = outputs_path / f"{input_stem}_frame_check.csv"
        with report_path.open("w", newline="", encoding="utf-8") as file:
            writer_csv = csv.DictWriter(
                file,
                fieldnames=[
                    "frame",
                    "left_detected",
                    "right_detected",
                    "both_detected",
                    "geometry_ok",
                    "lane_width_px",
                ],
            )
            writer_csv.writeheader()
            writer_csv.writerows(report_rows)
        print(f"Saved frame check report: {report_path}")

    if total_frames > 0:
        both_ratio = (100.0 * both_detected_count) / total_frames
        valid_ratio = (100.0 * valid_geometry_count) / total_frames
        print(
            f"Lane detection summary: total_frames={total_frames}, "
            f"both_detected={both_detected_count} ({both_ratio:.1f}%), "
            f"geometry_ok={valid_geometry_count} ({valid_ratio:.1f}%)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect road lanes from a video and color detected lane area in red."
    )
    parser.add_argument("input_video", help="Path to the road video")
    parser.add_argument(
        "-o",
        "--output-video",
        help="Optional output video path (e.g. output.mp4)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live preview window",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory to save debug images and default output video",
    )
    parser.add_argument(
        "--use-adaptive-threshold",
        action="store_true",
        help="Enable adaptive thresholding before edge detection",
    )
    parser.add_argument(
        "--frame-check-report",
        action="store_true",
        help="Save per-frame lane detection quality report (CSV)",
    )
    parser.add_argument(
        "--camera-position",
        choices=["low", "mid", "high"],
        default="mid",
        help="Camera mounting profile used to tune Canny/Hough parameters",
    )
    parser.add_argument("--canny-low", type=int, help="Override tuned Canny low threshold")
    parser.add_argument("--canny-high", type=int, help="Override tuned Canny high threshold")
    parser.add_argument("--hough-threshold", type=int, help="Override tuned Hough vote threshold")
    parser.add_argument(
        "--hough-min-line-length",
        type=int,
        help="Override tuned Hough minimum line length",
    )
    parser.add_argument(
        "--hough-max-line-gap",
        type=int,
        help="Override tuned Hough maximum line gap",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_and_color_lanes(
        input_video=args.input_video,
        output_video=args.output_video,
        display=not args.no_display,
        outputs_dir=args.outputs_dir,
        use_adaptive_threshold=args.use_adaptive_threshold,
        frame_check_report=args.frame_check_report,
        camera_position=args.camera_position,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_threshold=args.hough_threshold,
        hough_min_line_length=args.hough_min_line_length,
        hough_max_line_gap=args.hough_max_line_gap,
    )
