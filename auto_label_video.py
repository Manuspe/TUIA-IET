"""Automatically label a video and export detections as CVAT video tracks."""

from __future__ import annotations

import argparse
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in a video inside an ROI and export CVAT XML."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--model",
        default="yolo26x.pt",
        help="Ultralytics model or local weights path (default: yolo26x.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cvat_annotations.xml"),
        help="CVAT XML output path",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="ROI in pixels; if omitted, select it in a window",
    )
    parser.add_argument(
        "--roi-mode",
        choices=("center", "intersection"),
        default="center",
        help="How a detection is accepted by the ROI (default: center)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument(
        "--frame-step",
        "--stride",
        dest="frame_step",
        type=int,
        default=1,
        help="Process one frame every N frames (default: 1)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        help="Only keep these class names, for example: --classes person car",
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0")
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional path for a video preview with boxes and the ROI",
    )
    return parser.parse_args()


def choose_roi(frame: Any, roi: list[int] | None) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    if roi is None:
        selected = cv2.selectROI("Select detection ROI and press ENTER", frame, False, False)
        cv2.destroyWindow("Select detection ROI and press ENTER")
        roi = [int(value) for value in selected]
    x, y, roi_width, roi_height = roi
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI must have a positive width and height")
    if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError(f"ROI must fit inside the video frame ({width}x{height})")
    return x, y, roi_width, roi_height


def box_is_in_roi(
    box: tuple[float, float, float, float],
    roi: tuple[int, int, int, int],
    mode: str,
) -> bool:
    x1, y1, x2, y2 = box
    roi_x, roi_y, roi_width, roi_height = roi
    roi_x2, roi_y2 = roi_x + roi_width, roi_y + roi_height
    if mode == "center":
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        return roi_x <= center_x <= roi_x2 and roi_y <= center_y <= roi_y2
    return max(x1, roi_x) < min(x2, roi_x2) and max(y1, roi_y) < min(y2, roi_y2)


def add_box(track: ET.Element, frame_number: int, box: tuple[float, ...]) -> None:
    x1, y1, x2, y2, confidence = box
    ET.SubElement(
        track,
        "box",
        {
            "frame": str(frame_number),
            "outside": "0",
            "occluded": "0",
            "keyframe": "1",
            "xtl": f"{x1:.2f}",
            "ytl": f"{y1:.2f}",
            "xbr": f"{x2:.2f}",
            "ybr": f"{y2:.2f}",
            "conf": f"{confidence:.4f}",
            "z_order": "0",
        },
    )


def export_cvat(
    output: Path,
    video: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    tracks: dict[tuple[int, str], ET.Element],
) -> None:
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = video.stem
    ET.SubElement(task, "size").text = str(frame_count)
    ET.SubElement(task, "mode").text = "interpolation"
    original_size = ET.SubElement(task, "original_size")
    ET.SubElement(original_size, "width").text = str(width)
    ET.SubElement(original_size, "height").text = str(height)
    ET.SubElement(task, "source").text = str(video)
    ET.SubElement(task, "fps").text = f"{fps:.6f}"
    for track in tracks.values():
        annotations.append(track)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(annotations).write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    if not args.video.exists():
        raise FileNotFoundError(args.video)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("The video has no readable frames")
    roi = choose_roi(first_frame, args.roi)

    model = YOLO(args.model)
    allowed_classes = set(args.classes or [])
    tracks: dict[tuple[int, str], ET.Element] = {}
    fallback_id = 0
    preview_writer = None
    if args.preview:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(str(args.preview), codec, fps, (width, height))

    frame_number = 0
    while True:
        if frame_number == 0:
            frame = first_frame
        else:
            ok, frame = capture.read()
            if not ok:
                break
        if frame_number % args.frame_step == 0:
            inference = model.track(
                frame,
                persist=True,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            names = inference.names
            boxes = inference.boxes
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            for box_tensor, confidence_tensor, class_tensor, track_id in zip(
                boxes.xyxy.tolist(),
                boxes.conf.tolist(),
                boxes.cls.int().tolist(),
                ids,
            ):
                label = str(names[class_tensor])
                box = tuple(float(value) for value in box_tensor)
                if allowed_classes and label not in allowed_classes:
                    continue
                if not box_is_in_roi(box, roi, args.roi_mode):
                    continue
                if track_id is None:
                    fallback_id += 1
                    track_id = fallback_id
                track_key = (int(track_id), label)
                track = tracks.get(track_key)
                if track is None:
                    track = ET.Element(
                        "track", {"id": str(track_key[0]), "label": label, "source": "auto"}
                    )
                    tracks[track_key] = track
                confidence = float(confidence_tensor)
                add_box(track, frame_number, (*box, confidence))

                if preview_writer is not None:
                    x1, y1, x2, y2 = (int(value) for value in box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 220, 0),
                        2,
                    )
        if preview_writer is not None:
            x, y, roi_width, roi_height = roi
            cv2.rectangle(frame, (x, y), (x + roi_width, y + roi_height), (255, 180, 0), 2)
            preview_writer.write(frame)
        frame_number += 1

    capture.release()
    if preview_writer is not None:
        preview_writer.release()
    export_cvat(args.output, args.video, width, height, fps, frame_count, tracks)
    print(f"Exported {len(tracks)} tracks to {args.output}")


if __name__ == "__main__":
    main()