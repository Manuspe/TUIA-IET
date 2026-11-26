"""Extract video frames, detect objects across the full frame,
and export both frames and annotations as a CVAT Images 1.1 ZIP dataset.
"""

from __future__ import annotations

import argparse
import cv2
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames, detect objects across the full frame, and export a CVAT Images ZIP dataset."
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
    parser.add_argument("--conf", type=float, default=0.22, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
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
        default=["car", "motorcycle", "bicycle", "person"],
        help="Solo conserva las clases indicadas (por defecto: car, motorcycle, bicycle, person)",
    )
    parser.add_argument("--device", default=0, help="Inference device, e.g. cpu or 0")
    return parser.parse_args()


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

                    if label not in allowed_classes:
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
