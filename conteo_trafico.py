"""Aforo vehicular y peatonal sobre video con YOLO y supervision.

Flujo de trabajo:
1. Extrae un frame de referencia (~1.5 s) para dibujar interactivamente:
   - Paso A: Poligono ROI (zona de transito valido).
   - Paso B: 4 lineas virtuales de conteo (Ingreso, Egreso, Ingreso/Egreso 1, Ingreso/Egreso 2).
2. Ejecuta deteccion con Ultralytics YOLO, filtra clases de interes y ROI
   (usando BOTTOM_CENTER) y realiza tracking con ByteTrack.
3. Contabiliza cruces unicos por par (tracker_id, linea) y exporta una tabla
   consolidada en formato CSV junto al video de entrada.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


TARGET_CLASSES = ("car", "motorcycle", "bicycle", "person")

LINE_SPECS: tuple[tuple[str, tuple[int, int, int], sv.Color], ...] = (
    ("Ingreso", (0, 220, 0), sv.Color.from_bgr_tuple((0, 220, 0))),
    ("Egreso", (0, 0, 255), sv.Color.from_bgr_tuple((0, 0, 255))),
    ("Ingreso/Egreso 1", (255, 200, 0), sv.Color.from_bgr_tuple((255, 200, 0))),
    ("Ingreso/Egreso 2", (255, 0, 220), sv.Color.from_bgr_tuple((255, 0, 220))),
)

DEFAULT_WEIGHTS = "runs/detect/TUIA-IET_runs/yolo26l_trafico_v1-3/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aforo vehicular y peatonal sobre video con YOLO y supervision."
    )
    parser.add_argument("video", type=Path, help="Ruta al archivo de video de entrada")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(DEFAULT_WEIGHTS),
        help=f"Ruta a los pesos del modelo (por defecto: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Umbral de confianza para detecciones (por defecto: 0.35)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamano de inferencia (por defecto: 640)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Dispositivo de inferencia, ej. 0 o cpu (por defecto: 0)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Procesa el video sin abrir ventana de visualizacion",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Guarda el video anotado como <nombre_video>_aforo.mp4",
    )
    return parser.parse_args()


def normalize_device(device_arg: str | int) -> str | int:
    if isinstance(device_arg, int):
        return device_arg
    text = str(device_arg).strip()
    return int(text) if text.isdigit() else text


def extract_reference_frame(cap: cv2.VideoCapture) -> np.ndarray:
    """Extrae un frame alrededor del segundo 1-2 evitando el frame 0."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_idx = max(1, int(round(fps * 1.5)))
    if total_frames > 1:
        target_idx = min(target_idx, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame

    # Respaldo secuencial si el seek fallo por codec/contenedor
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_good: np.ndarray | None = None
    for idx in range(max(2, target_idx + 1)):
        ok, candidate = cap.read()
        if not ok or candidate is None:
            break
        last_good = candidate
        if idx >= 1 and idx >= target_idx // 2:
            return candidate

    if last_good is not None:
        return last_good
    raise RuntimeError("No se pudo leer ningun frame de referencia del video.")


def draw_hud_banner(
    frame: np.ndarray,
    lines: list[str],
    accent_bgr: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Dibuja un cartel legible en la parte superior del frame."""
    overlay = frame.copy()
    banner_h = 26 + 26 * len(lines)
    cv2.rectangle(overlay, (10, 10), (min(frame.shape[1] - 10, 960), 10 + banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for idx, text in enumerate(lines):
        color = accent_bgr if idx == 0 else (235, 235, 235)
        cv2.putText(
            frame,
            text,
            (22, 34 + idx * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_label_pill(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    bgr_color: tuple[int, int, int],
) -> None:
    """Dibuja una etiqueta con fondo oscuro junto a un punto del frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, min(frame.shape[1] - tw - 16, anchor[0] - tw // 2))
    y = max(th + 12, min(frame.shape[0] - baseline - 8, anchor[1]))
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 4), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 4), bgr_color, 1)
    cv2.putText(frame, text, (x, y), font, scale, bgr_color, thickness, cv2.LINE_AA)


def select_roi_polygon(ref_frame: np.ndarray) -> np.ndarray:
    """Paso A: permite trazar el poligono ROI de transito valido."""
    points: list[tuple[int, int]] = []
    window_name = "Paso A - Poligono ROI (Zona Valida)"

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            canvas = ref_frame.copy()
            if points:
                for pt in points:
                    cv2.circle(canvas, pt, 5, (0, 255, 255), -1)
                for i in range(len(points) - 1):
                    cv2.line(canvas, points[i], points[i + 1], (255, 180, 0), 2)
                if len(points) >= 3:
                    pts_arr = np.array(points, dtype=np.int32)
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [pts_arr], (255, 180, 0))
                    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
                    cv2.polylines(canvas, [pts_arr], True, (255, 180, 0), 2)

            draw_hud_banner(
                canvas,
                [
                    "PASO A: Marcar Poligono ROI (zona de transito valido)",
                    f"Puntos: {len(points)} | Clic Izq: agregar | Clic Der: deshacer | ENTER/ESPACIO: confirmar (<3 pts = todo el frame)",
                ],
                accent_bgr=(255, 200, 0),
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(16) & 0xFF
            if key in (13, 10, 32):
                break
    finally:
        cv2.destroyWindow(window_name)

    h, w = ref_frame.shape[:2]
    if len(points) >= 3:
        return np.array(points, dtype=np.int32)
    # Margen de 1 px para incluir detecciones cuyo BOTTOM_CENTER toca los bordes
    return np.array([[-1, -1], [w + 1, -1], [w + 1, h + 1], [-1, h + 1]], dtype=np.int32)


def draw_confirmed_geometries(
    canvas: np.ndarray,
    roi_polygon: np.ndarray,
    confirmed_lines: list[tuple[str, tuple[int, int, int], tuple[int, int], tuple[int, int]]],
) -> None:
    """Dibuja la ROI y las lineas ya confirmadas sobre el lienzo."""
    cv2.polylines(canvas, [roi_polygon], True, (255, 180, 0), 2)
    for label, bgr, p1, p2 in confirmed_lines:
        cv2.line(canvas, p1, p2, bgr, 3)
        cv2.circle(canvas, p1, 5, bgr, -1)
        cv2.circle(canvas, p2, 5, bgr, -1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
        draw_label_pill(canvas, label, mid, bgr)


def select_counting_lines(
    ref_frame: np.ndarray,
    roi_polygon: np.ndarray,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Paso B: guia secuencialmente el trazado de las 4 lineas de conteo."""
    window_name = "Paso B - Lineas de Conteo"
    confirmed: list[tuple[str, tuple[int, int, int], tuple[int, int], tuple[int, int]]] = []
    current_points: list[tuple[int, int]] = []
    mouse_pos: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[:] = [(x, y)]
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(current_points) < 2:
                current_points.append((x, y))
            else:
                current_points[1] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            current_points.clear()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        for idx, (line_name, bgr, _) in enumerate(LINE_SPECS, start=1):
            current_points.clear()
            warning_msg = ""
            while True:
                canvas = ref_frame.copy()
                draw_confirmed_geometries(canvas, roi_polygon, confirmed)

                if len(current_points) == 1:
                    cv2.circle(canvas, current_points[0], 6, bgr, -1)
                    if mouse_pos:
                        cv2.line(canvas, current_points[0], mouse_pos[0], bgr, 2, cv2.LINE_AA)
                elif len(current_points) == 2:
                    p1, p2 = current_points
                    cv2.line(canvas, p1, p2, bgr, 3, cv2.LINE_AA)
                    cv2.circle(canvas, p1, 6, bgr, -1)
                    cv2.circle(canvas, p2, 6, bgr, -1)
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
                    draw_label_pill(canvas, line_name, mid, bgr)

                hud_lines = [
                    f"PASO B ({idx}/4): Trazar linea '{line_name}'",
                    "Clic Izq (2 veces): inicio y fin | Clic Der: reiniciar linea | ENTER/ESPACIO: confirmar",
                ]
                if warning_msg:
                    hud_lines.append(warning_msg)
                draw_hud_banner(canvas, hud_lines, accent_bgr=bgr)

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(16) & 0xFF
                if key in (13, 10, 32):
                    if len(current_points) == 2 and current_points[0] != current_points[1]:
                        confirmed.append((line_name, bgr, current_points[0], current_points[1]))
                        break
                    warning_msg = "Marca 2 puntos distintos antes de confirmar."
    finally:
        cv2.destroyWindow(window_name)

    return [(p1, p2) for _, _, p1, p2 in confirmed]


def export_csv(
    csv_path: Path,
    counts: dict[str, dict[str, int]],
) -> None:
    """Exporta la tabla consolidada de aforo en UTF-8."""
    header = [
        "Clase",
        "Ingreso",
        "Egreso",
        "Ingreso/Egreso 1",
        "Ingreso/Egreso 2",
        "Total Ingreso/Egreso",
        "Total General",
    ]
    col_totals = [0, 0, 0, 0, 0, 0]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cls_name in TARGET_CLASSES:
            c = counts[cls_name]
            ingreso = c["Ingreso"]
            egreso = c["Egreso"]
            ie1 = c["Ingreso/Egreso 1"]
            ie2 = c["Ingreso/Egreso 2"]
            total_ie = ie1 + ie2
            total_gen = ingreso + egreso + total_ie

            row_vals = [ingreso, egreso, ie1, ie2, total_ie, total_gen]
            for i, v in enumerate(row_vals):
                col_totals[i] += v
            writer.writerow([cls_name, *row_vals])

        writer.writerow(["TOTAL", *col_totals])


def create_tracker(lost_track_buffer: int = 60):
    """Instancia ByteTrack compatible con supervision <0.31 y >=0.31 (paquete trackers)."""
    if hasattr(sv, "ByteTrack"):
        tracker = sv.ByteTrack(lost_track_buffer=lost_track_buffer)
        return tracker, tracker.update_with_detections

    try:
        from trackers import ByteTrackTracker
    except ImportError as exc:
        raise ImportError(
            "En supervision >= 0.31.0 ByteTrack se movio al paquete 'trackers'. "
            "Instalalo con: pip install trackers  (o usa: pip install 'supervision<0.31.0')"
        ) from exc

    tracker = ByteTrackTracker(lost_track_buffer=lost_track_buffer)
    return tracker, tracker.update


def run_counting(args: argparse.Namespace) -> None:
    video_path: Path = args.video
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontro el video de entrada: {video_path}")
    if not args.weights.exists():
        raise FileNotFoundError(f"No se encontraron los pesos del modelo: {args.weights}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    writer: cv2.VideoWriter | None = None
    counts: dict[str, dict[str, int]] = {
        cls_name: {line_name: 0 for line_name, _, _ in LINE_SPECS}
        for cls_name in TARGET_CLASSES
    }
    csv_path = video_path.with_suffix(".csv")
    processing_started = False

    try:
        ref_frame = extract_reference_frame(cap)
        roi_polygon = select_roi_polygon(ref_frame)
        line_segments = select_counting_lines(ref_frame, roi_polygon)

        # Reiniciar video al frame 0 para procesar completo
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        if args.save_video:
            out_video_path = video_path.with_name(f"{video_path.stem}_aforo.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"No se pudo crear el archivo de video: {out_video_path}")

        device = normalize_device(args.device)
        model = YOLO(str(args.weights))
        model_names: dict[int, str] = model.names
        allowed_ids = [
            cid for cid, name in model_names.items() if name in TARGET_CLASSES
        ]
        if not allowed_ids:
            raise ValueError(
                f"El modelo no contiene las clases esperadas {TARGET_CLASSES}. "
                f"Clases encontradas: {model_names}"
            )
        allowed_ids_arr = np.array(allowed_ids, dtype=int)

        polygon_zone = sv.PolygonZone(
            polygon=roi_polygon,
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
        )
        _, update_tracker = create_tracker(lost_track_buffer=60)

        line_zones: list[sv.LineZone] = []
        line_annotators: list[sv.LineZoneAnnotator] = []
        for (p1, p2), (_, _, sv_color) in zip(line_segments, LINE_SPECS):
            line_zones.append(
                sv.LineZone(
                    start=sv.Point(p1[0], p1[1]),
                    end=sv.Point(p2[0], p2[1]),
                    triggering_anchors=[sv.Position.BOTTOM_CENTER],
                )
            )
            line_annotators.append(
                sv.LineZoneAnnotator(
                    color=sv_color,
                    thickness=2,
                    display_in_count=False,
                    display_out_count=False,
                )
            )

        polygon_annotator = sv.PolygonZoneAnnotator(
            zone=polygon_zone,
            color=sv.Color.from_bgr_tuple((255, 180, 0)),
            thickness=2,
            display_in_zone_count=False,
        )
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        counted_pairs: set[tuple[int, int]] = set()
        line_totals = [0, 0, 0, 0]

        display_window = "Aforo de Trafico - Presiona 'q' para salir"
        if not args.no_show:
            cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)

        processing_started = True
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)

            if len(detections) > 0 and detections.class_id is not None:
                class_mask = np.isin(detections.class_id, allowed_ids_arr)
                detections = detections[class_mask]

            if len(detections) > 0:
                roi_mask = polygon_zone.trigger(detections=detections)
                detections = detections[roi_mask]

            detections = update_tracker(detections)

            if len(detections) > 0 and detections.tracker_id is not None:
                for line_idx, line_zone in enumerate(line_zones):
                    crossed_in, crossed_out = line_zone.trigger(detections=detections)
                    crossed_mask = np.asarray(crossed_in, dtype=bool) | np.asarray(
                        crossed_out, dtype=bool
                    )
                    if not np.any(crossed_mask):
                        continue

                    line_name = LINE_SPECS[line_idx][0]
                    for det_idx in np.where(crossed_mask)[0]:
                        tid = detections.tracker_id[det_idx]
                        cid = detections.class_id[det_idx]
                        if tid is None or cid is None:
                            continue
                        pair = (int(tid), line_idx)
                        if pair in counted_pairs:
                            continue
                        counted_pairs.add(pair)
                        cls_name = model_names[int(cid)]
                        if cls_name in counts:
                            counts[cls_name][line_name] += 1
                            line_totals[line_idx] += 1

            annotated = frame.copy()
            annotated = polygon_annotator.annotate(scene=annotated, label="ROI")

            for line_idx, (line_zone, line_ann, (p1, p2), (line_name, bgr, _)) in enumerate(
                zip(line_zones, line_annotators, line_segments, LINE_SPECS)
            ):
                annotated = line_ann.annotate(annotated, line_zone)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 8)
                draw_label_pill(annotated, f"{line_name}: {line_totals[line_idx]}", mid, bgr)

            if len(detections) > 0:
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                labels = [
                    f"#{int(tid) if tid is not None else '?'} {model_names[int(cid)]}"
                    for tid, cid in zip(
                        detections.tracker_id
                        if detections.tracker_id is not None
                        else [None] * len(detections),
                        detections.class_id,
                    )
                ]
                annotated = label_annotator.annotate(
                    scene=annotated,
                    detections=detections,
                    labels=labels,
                )

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow(display_window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if processing_started:
            export_csv(csv_path, counts)
            print(f"CSV exportado en: {csv_path}")


if __name__ == "__main__":
    run_counting(parse_args())
