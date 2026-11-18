"""Extract frames from a video based on frame-step, detect objects,
and export both frames and annotations as a CVAT Images 1.1 ZIP dataset.
"""

from __future__ import annotations

import argparse
import cv2
import io
import zipfile
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ultralytics import YOLO

# Variable global para almacenar los puntos del polígono
puntos_poligono = []

def capturar_puntos(event, x, y, flags, param):
    """Callback para manejar los clics del mouse en la ventana de dibujo."""
    global puntos_poligono
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_poligono.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if puntos_poligono:
            puntos_poligono.pop()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames, detect objects inside a polygonal ROI, and export CVAT Images ZIP dataset."
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
        "--roi-mode",
        choices=("center", "intersection"),
        default="center",
        help="How a detection is accepted by the ROI (default: center)",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
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
        default=["car", "motorcycle", "bicycle", "person"],
        help="Solo conserva las clases indicadas (por defecto: car, motorcycle, bicycle, person)",
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0")
    return parser.parse_args()


def definir_roi_poligonal(frame: Any) -> np.ndarray:
    """Abre una interfaz para trazar un polígono y devuelve sus vértices."""
    global puntos_poligono
    puntos_poligono.clear()
    
    frame_copia = frame.copy()
    # Nombre estricto ASCII para evitar fallos en setMouseCallback
    nombre_ventana = "Seleccionar ROI"
    
    # WINDOW_NORMAL permite redimensionar la ventana sin desfasar los clics
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(nombre_ventana, capturar_puntos)

    while True:
        temp_frame = frame_copia.copy()
        
        # Dibujar polígono en tiempo real
        if len(puntos_poligono) > 0:
            for pt in puntos_poligono:
                cv2.circle(temp_frame, pt, 4, (0, 0, 255), -1)
            for i in range(len(puntos_poligono) - 1):
                cv2.line(temp_frame, puntos_poligono[i], puntos_poligono[i+1], (0, 255, 0), 2)
            if len(puntos_poligono) > 2:
                cv2.line(temp_frame, puntos_poligono[-1], puntos_poligono[0], (255, 0, 0), 2)

        # Imprimir instrucciones sobre el frame en lugar del título de la ventana
        cv2.putText(
            temp_frame, 
            "Click Izq: Agregar | Click Der: Borrar | ENTER o ESPACIO: Confirmar", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2
        )

        cv2.imshow(nombre_ventana, temp_frame)
        
        # Capturar teclado (13 = ENTER, 32 = ESPACIO)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]: 
            break

    cv2.destroyWindow(nombre_ventana)
    
    if len(puntos_poligono) > 2:
        return np.array(puntos_poligono, np.int32)
    else:
        # Si no se dibuja nada válido, asume el frame completo como ROI
        h, w = frame.shape[:2]
        print("No se dibujó un polígono válido. Usando todo el frame como ROI.")
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], np.int32)


def box_is_in_polygon(
    box: tuple[float, float, float, float],
    polygon: np.ndarray,
    mode: str,
) -> bool:
    """Verifica si el bounding box está dentro del polígono utilizando pointPolygonTest."""
    x1, y1, x2, y2 = box
    
    if mode == "center":
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # pointPolygonTest retorna >= 0 si el punto está dentro o en el borde del polígono
        return cv2.pointPolygonTest(polygon, (float(center_x), float(center_y)), False) >= 0
    
    else: # mode == "intersection"
        # Para intersección, comprobamos si el centro o alguna de las esquinas del box cae dentro del polígono
        puntos_a_comprobar = [
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0), # Centro
            (x1, y1), # Esquina superior izquierda
            (x2, y1), # Esquina superior derecha
            (x2, y2), # Esquina inferior derecha
            (x1, y2)  # Esquina inferior izquierda
        ]
        for pt_x, pt_y in puntos_a_comprobar:
            if cv2.pointPolygonTest(polygon, (float(pt_x), float(pt_y)), False) >= 0:
                return True
        return False


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

    # Extraer los vértices del polígono al iniciar
    polygon_roi = definir_roi_poligonal(first_frame)
    
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
                        
                    # Validar si el vehículo está dentro de la ROI Poligonal
                    if not box_is_in_polygon(box, polygon_roi, args.roi_mode):
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