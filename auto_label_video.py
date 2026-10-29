"""Extract frames from a video based on frame-step, detect objects,
and export both frames and annotations as a CVAT Images 1.1 ZIP dataset.
"""

from __future__ import annotations

import argparse
import cv2
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames, detect objects inside ROI, and export CVAT Images ZIP dataset."
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
        default=Path("cvat_dataset.zip"),
        help="CVAT ZIP output path (default: cvat_dataset.zip)",
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
        default=30,
        help="Process one frame every N frames (default: 30)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        help="Only keep these class names, for example: --classes person car",
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0")
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
    
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("The video has no readable frames")

    roi = choose_roi(first_frame, args.roi)
    model = YOLO(args.model)
    allowed_classes = set(args.classes or [])

    # Inicializar el árbol XML para CVAT for images 1.1
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = args.video.stem

    used_labels: set[str] = set()
    images_nodes: list[ET.Element] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    extracted_count = 0

    print("Procesando video y creando ZIP...")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zipf:
        while True:
            if frame_number == 0:
                frame = first_frame
                ok = True
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            if frame_number % args.frame_step == 0:
                image_name = f"frame_{frame_number:06d}.jpg"
                
                # Inferencia con Ultralytics
                results = model.predict(
                    frame,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )[0]

                # Nodo XML para la imagen actual
                image_node = ET.Element(
                    "image",
                    {
                        "id": str(extracted_count),
                        "name": image_name,
                        "width": str(width),
                        "height": str(height),
                    },
                )

                names = results.names
                boxes = results.boxes

                for box_tensor, confidence_tensor, class_tensor in zip(
                    boxes.xyxy.tolist(),
                    boxes.conf.tolist(),
                    boxes.cls.int().tolist(),
                ):
                    label = str(names[class_tensor])
                    box = tuple(float(v) for v in box_tensor)

                    if allowed_classes and label not in allowed_classes:
                        continue
                    if not box_is_in_roi(box, roi, args.roi_mode):
                        continue

                    used_labels.add(label)
                    x1, y1, x2, y2 = box
                    confidence = float(confidence_tensor)

                    # Sub-nodo para la caja delimitadora (detección)
                    ET.SubElement(
                        image_node,
                        "box",
                        {
                            "label": label,
                            "source": "auto",
                            "occluded": "0",
                            "xtl": f"{x1:.2f}",
                            "ytl": f"{y1:.2f}",
                            "xbr": f"{x2:.2f}",
                            "ybr": f"{y2:.2f}",
                            "confidence": f"{confidence:.4f}",
                            "z_order": "0",
                        },
                    )

                images_nodes.append(image_node)

                # Comprimir la imagen JPEG directamente dentro del ZIP en memoria
                _, img_encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                zipf.writestr(f"images/{image_name}", img_encoded.tobytes())
                
                extracted_count += 1

            frame_number += 1

        capture.release()

        # Construir la sección de etiquetas en el meta del XML
        labels_elem = ET.SubElement(task, "labels")
        for label_name in sorted(used_labels):
            label_elem = ET.SubElement(labels_elem, "label")
            ET.SubElement(label_elem, "name").text = label_name

        # Adjuntar las imágenes al XML
        for img_node in images_nodes:
            annotations.append(img_node)

        # Escribir el annotations.xml dentro del ZIP
        xml_bytes = ET.tostring(annotations, encoding="utf-8", xml_declaration=True)
        zipf.writestr("annotations.xml", xml_bytes)

    print(f"Proceso finalizado. Se extrajeron {extracted_count} frames.")
    print(f"Archivo listo para importar en CVAT: {args.output}")


if __name__ == "__main__":
    main()
