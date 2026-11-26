"""Construye un dataset YOLO (train/val + data.yaml) combinando múltiples
fuentes comprimidas en ZIP:

- Archivos `*_dataset.zip` generados por `auto_label_video.py` (contienen las
  imágenes y las anotaciones automáticas iniciales).
- Archivos `.zip` exportados desde CVAT (contienen `annotations.xml` con las
  etiquetas corregidas).

Por defecto:
- Empareja automáticamente cada ZIP de CVAT con su `*_dataset.zip` ignorando
  mayúsculas, guiones bajos y el sufijo `_dataset` (p. ej. `Imae_1730.zip` con
  `IMAE1730_dataset.zip`).
- Omite los `*_dataset.zip` que aún no tienen un ZIP de CVAT asociado (puede
  cambiarse con `--include-auto-only`).
- Realiza un split estratificado 80/20 dentro de cada fuente para que todas las
  escenas queden representadas proporcionalmente en `train` y `val`.
- Prefija los nombres de archivo con el nombre de la fuente para evitar
  colisiones entre frames (`frame_000000.jpg`, etc.).
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetSource:
    """Representa una fuente emparejada de imágenes y anotaciones CVAT XML."""

    name: str
    images_zip: Path
    annotations_zip: Path
    using_cvat: bool


@dataclass(frozen=True)
class SampleItem:
    """Una imagen y sus cajas normalizadas en formato YOLO."""

    source_name: str
    zip_member: str
    output_stem: str
    image_ext: str
    yolo_lines: tuple[str, ...]


def normalize_key(stem: str) -> str:
    """Normaliza un nombre de archivo para emparejar ZIPs de CVAT y _dataset."""
    cleaned = re.sub(r"_dataset$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", cleaned.lower())


def clean_source_name(dataset_zip: Path) -> str:
    """Devuelve un prefijo legible y seguro a partir del nombre del ZIP."""
    stem = re.sub(r"_dataset$", "", dataset_zip.stem, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "source"


def zip_has_images(zip_path: Path) -> bool:
    """Indica si el archivo ZIP contiene imágenes."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return any(
            not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
            for info in zf.infolist()
        )


def zip_has_annotations_xml(zip_path: Path) -> bool:
    """Indica si el archivo ZIP contiene annotations.xml."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        return any(
            not info.is_dir() and PurePosixPath(info.filename).name == "annotations.xml"
            for info in zf.infolist()
        )


def find_annotations_member(zf: zipfile.ZipFile, zip_path: Path) -> str:
    """Encuentra el archivo annotations.xml dentro de un ZIP."""
    candidates = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name == "annotations.xml"
    ]
    if not candidates:
        raise FileNotFoundError(f"No se encontró 'annotations.xml' dentro de {zip_path}")
    # Priorizar annotations.xml en la raíz si existiera más de uno
    candidates.sort(key=lambda p: (len(PurePosixPath(p).parts), p))
    return candidates[0]


def build_image_index(zf: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Indexa entradas de imagen de un ZIP por ruta exacta y por nombre base."""
    by_path: dict[str, str] = {}
    by_basename: dict[str, list[str]] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        p = PurePosixPath(info.filename)
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        norm_path = p.as_posix().lstrip("./")
        by_path[norm_path] = info.filename
        by_basename.setdefault(p.name, []).append(info.filename)
    return by_path, by_basename


def resolve_zip_image(
    xml_image_name: str,
    by_path: dict[str, str],
    by_basename: dict[str, list[str]],
    images_zip: Path,
) -> str:
    """Resuelve el miembro del ZIP correspondiente al atributo `name` del XML."""
    norm = PurePosixPath(xml_image_name.replace("\\", "/")).as_posix().lstrip("./")
    if norm in by_path:
        return by_path[norm]

    basename = PurePosixPath(norm).name
    for candidate in (f"images/{basename}", basename):
        if candidate in by_path:
            return by_path[candidate]

    matches = by_basename.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Nombre de imagen ambiguo '{xml_image_name}' en {images_zip}: {matches}"
        )
    raise FileNotFoundError(
        f"No se encontró la imagen '{xml_image_name}' dentro de {images_zip}"
    )


def parse_explicit_pairs(pair_args: list[str], input_dir: Path) -> list[DatasetSource]:
    """Parsea pares explícitos `imagenes.zip=cvat.zip` pasados por CLI."""
    sources: list[DatasetSource] = []
    for raw in pair_args:
        if "=" not in raw:
            raise ValueError(
                f"Formato inválido para --pair '{raw}'. Usa 'dataset.zip=cvat.zip'."
            )
        img_str, ann_str = raw.split("=", 1)
        img_path = Path(img_str)
        ann_path = Path(ann_str)
        if not img_path.is_absolute():
            img_path = input_dir / img_path
        if not ann_path.is_absolute():
            ann_path = input_dir / ann_path
        if not img_path.exists():
            raise FileNotFoundError(f"No existe el ZIP de imágenes: {img_path}")
        if not ann_path.exists():
            raise FileNotFoundError(f"No existe el ZIP de anotaciones: {ann_path}")
        sources.append(
            DatasetSource(
                name=clean_source_name(img_path),
                images_zip=img_path,
                annotations_zip=ann_path,
                using_cvat=img_path.resolve() != ann_path.resolve(),
            )
        )
    return sources


def discover_sources(
    input_dir: Path,
    include_auto_only: bool = False,
) -> tuple[list[DatasetSource], list[Path]]:
    """Descubre y empareja ZIPs `*_dataset.zip` con sus ZIPs de CVAT."""
    all_zips = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip")
    dataset_zips = [p for p in all_zips if p.stem.lower().endswith("_dataset")]
    other_zips = [p for p in all_zips if not p.stem.lower().endswith("_dataset")]

    cvat_by_key: dict[str, Path] = {}
    for zip_path in other_zips:
        if not zip_has_annotations_xml(zip_path):
            print(f"[OMITIDO] {zip_path.name}: no contiene annotations.xml.")
            continue
        key = normalize_key(zip_path.stem)
        if key in cvat_by_key:
            raise ValueError(
                f"Múltiples ZIPs de CVAT coinciden con la clave '{key}': "
                f"{cvat_by_key[key].name} y {zip_path.name}"
            )
        cvat_by_key[key] = zip_path

    dataset_by_key: dict[str, Path] = {}
    for zip_path in dataset_zips:
        key = normalize_key(zip_path.stem)
        if key in dataset_by_key:
            raise ValueError(
                f"Múltiples ZIPs *_dataset coinciden con la clave '{key}': "
                f"{dataset_by_key[key].name} y {zip_path.name}"
            )
        dataset_by_key[key] = zip_path

    sources: list[DatasetSource] = []
    skipped_auto: list[Path] = []
    used_cvat_keys: set[str] = set()

    for key, ds_zip in dataset_by_key.items():
        if key in cvat_by_key:
            cvat_zip = cvat_by_key[key]
            used_cvat_keys.add(key)
            sources.append(
                DatasetSource(
                    name=clean_source_name(ds_zip),
                    images_zip=ds_zip,
                    annotations_zip=cvat_zip,
                    using_cvat=True,
                )
            )
        elif include_auto_only:
            sources.append(
                DatasetSource(
                    name=clean_source_name(ds_zip),
                    images_zip=ds_zip,
                    annotations_zip=ds_zip,
                    using_cvat=False,
                )
            )
        else:
            skipped_auto.append(ds_zip)

    # Si un ZIP de CVAT ya incluye imágenes propias y no tiene *_dataset.zip, también soportarlo
    for key, cvat_zip in cvat_by_key.items():
        if key in used_cvat_keys:
            continue
        if zip_has_images(cvat_zip):
            sources.append(
                DatasetSource(
                    name=clean_source_name(cvat_zip),
                    images_zip=cvat_zip,
                    annotations_zip=cvat_zip,
                    using_cvat=True,
                )
            )
        else:
            raise FileNotFoundError(
                f"El ZIP de CVAT '{cvat_zip.name}' no contiene imágenes y no se encontró "
                f"un '*_dataset.zip' correspondiente en {input_dir}."
            )

    return sources, skipped_auto


def collect_class_names(sources: list[DatasetSource]) -> list[str]:
    """Obtiene la lista ordenada y unificada de clases a partir de los XMLs."""
    labels: set[str] = set()
    for src in sources:
        with zipfile.ZipFile(src.annotations_zip, "r") as zf:
            xml_member = find_annotations_member(zf, src.annotations_zip)
            root = ET.fromstring(zf.read(xml_member))
        for elem in root.findall(".//meta/task/labels/label/name"):
            if elem.text:
                labels.add(elem.text.strip())
        for elem in root.findall(".//meta/project/labels/label/name"):
            if elem.text:
                labels.add(elem.text.strip())
        for box in root.findall(".//box"):
            label = box.get("label")
            if label:
                labels.add(label.strip())

    if not labels:
        raise ValueError("No se encontraron clases en ningún annotations.xml.")
    return sorted(labels)


def load_source_samples(
    source: DatasetSource,
    class_to_id: dict[str, int],
) -> tuple[list[SampleItem], Counter[str], int]:
    """Carga las muestras de una fuente y convierte sus cajas a formato YOLO."""
    with zipfile.ZipFile(source.annotations_zip, "r") as ann_zf:
        xml_member = find_annotations_member(ann_zf, source.annotations_zip)
        root = ET.fromstring(ann_zf.read(xml_member))

    with zipfile.ZipFile(source.images_zip, "r") as img_zf:
        by_path, by_basename = build_image_index(img_zf)

    samples: list[SampleItem] = []
    class_counts: Counter[str] = Counter()
    empty_frames = 0
    seen_stems: set[str] = set()

    for img_elem in root.findall("image"):
        xml_name = img_elem.get("name")
        if not xml_name:
            continue
        width = float(img_elem.get("width", "0"))
        height = float(img_elem.get("height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Dimensiones inválidas para '{xml_name}' en {source.annotations_zip}: "
                f"{width}x{height}"
            )

        zip_member = resolve_zip_image(xml_name, by_path, by_basename, source.images_zip)
        member_path = PurePosixPath(zip_member)
        base_stem = member_path.stem
        output_stem = f"{source.name}__{base_stem}"
        if output_stem in seen_stems:
            # Respaldo ante subdirectorios con mismo nombre dentro de un mismo ZIP
            rel_safe = re.sub(r"[^A-Za-z0-9_-]+", "_", member_path.with_suffix("").as_posix())
            output_stem = f"{source.name}__{rel_safe}"
        seen_stems.add(output_stem)

        yolo_lines: list[str] = []
        for box in img_elem.findall("box"):
            if box.get("outside") == "1":
                continue
            label = (box.get("label") or "").strip()
            if label not in class_to_id:
                continue

            xtl = max(0.0, min(width, float(box.attrib["xtl"])))
            ytl = max(0.0, min(height, float(box.attrib["ytl"])))
            xbr = max(0.0, min(width, float(box.attrib["xbr"])))
            ybr = max(0.0, min(height, float(box.attrib["ybr"])))

            bw = xbr - xtl
            bh = ybr - ytl
            if bw <= 0.0 or bh <= 0.0:
                continue

            xc = ((xtl + xbr) / 2.0) / width
            yc = ((ytl + ybr) / 2.0) / height
            wn = bw / width
            hn = bh / height

            class_id = class_to_id[label]
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            class_counts[label] += 1

        if not yolo_lines:
            empty_frames += 1

        samples.append(
            SampleItem(
                source_name=source.name,
                zip_member=zip_member,
                output_stem=output_stem,
                image_ext=member_path.suffix.lower() or ".jpg",
                yolo_lines=tuple(yolo_lines),
            )
        )

    return samples, class_counts, empty_frames


def write_split_files(
    source: DatasetSource,
    train_items: list[SampleItem],
    val_items: list[SampleItem],
    output_dir: Path,
) -> None:
    """Extrae las imágenes del ZIP y escribe imágenes y labels en train/ y val/."""
    with zipfile.ZipFile(source.images_zip, "r") as img_zf:
        for subset, items in (("train", train_items), ("val", val_items)):
            images_dir = output_dir / subset / "images"
            labels_dir = output_dir / subset / "labels"
            for item in items:
                img_dest = images_dir / f"{item.output_stem}{item.image_ext}"
                lbl_dest = labels_dir / f"{item.output_stem}.txt"

                with img_zf.open(item.zip_member, "r") as src_f, img_dest.open("wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

                content = "\n".join(item.yolo_lines)
                if content:
                    content += "\n"
                lbl_dest.write_text(content, encoding="utf-8")


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Genera el archivo data.yaml compatible con Ultralytics YOLO."""
    yaml_path = output_dir / "data.yaml"
    lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def prepare_output_dirs(output_dir: Path, keep_existing: bool) -> None:
    """Prepara las carpetas train/ y val/ dentro del directorio de salida."""
    for subset in ("train", "val"):
        subset_dir = output_dir / subset
        if not keep_existing and subset_dir.exists():
            shutil.rmtree(subset_dir)
        (subset_dir / "images").mkdir(parents=True, exist_ok=True)
        (subset_dir / "labels").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combina múltiples pares *_dataset.zip + ZIP de CVAT y genera un "
            "dataset YOLO con split estratificado train/val."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Carpeta donde se encuentran los archivos .zip (por defecto: directorio actual)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Carpeta destino donde se crearán train/, val/ y data.yaml (por defecto: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proporción de imágenes para train dentro de cada fuente (por defecto: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad del split (por defecto: 42)",
    )
    parser.add_argument(
        "--include-auto-only",
        action="store_true",
        help="Incluye también los *_dataset.zip que aún no tienen ZIP de CVAT asociado",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="No borra train/ y val/ previos en --output-dir antes de escribir",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help=(
            "Emparejamiento manual en formato 'imagenes_dataset.zip=cvat.zip'. "
            "Puede repetirse varias veces; si se usa, reemplaza el autodescubrimiento."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio debe estar estrictamente entre 0 y 1.")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input_dir}")

    if args.pair:
        sources = parse_explicit_pairs(args.pair, args.input_dir)
        skipped_auto: list[Path] = []
    else:
        sources, skipped_auto = discover_sources(
            args.input_dir, include_auto_only=args.include_auto_only
        )

    if skipped_auto:
        print("Fuentes *_dataset.zip omitidas por no tener ZIP de CVAT:")
        for p in skipped_auto:
            print(f"  - {p.name}")
        print()

    if not sources:
        raise RuntimeError(
            "No se encontraron fuentes válidas para procesar. Verifica los archivos .zip."
        )

    class_names = collect_class_names(sources)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    prepare_output_dirs(args.output_dir, keep_existing=args.keep_existing)
    rng = random.Random(args.seed)

    total_train = 0
    total_val = 0
    total_boxes: Counter[str] = Counter()

    print(f"Clases detectadas ({len(class_names)}): {class_to_id}")
    print(f"Generando dataset YOLO en '{args.output_dir}' (split estratificado {args.train_ratio:.0%}/{1 - args.train_ratio:.0%}, seed={args.seed})...\n")

    for source in sources:
        samples, src_boxes, empty_frames = load_source_samples(source, class_to_id)
        rng.shuffle(samples)

        cut = int(len(samples) * args.train_ratio)
        train_items = samples[:cut]
        val_items = samples[cut:]

        write_split_files(source, train_items, val_items, args.output_dir)

        total_train += len(train_items)
        total_val += len(val_items)
        total_boxes.update(src_boxes)

        tipo = f"CVAT ({source.annotations_zip.name})" if source.using_cvat else "auto (sin CVAT)"
        print(
            f"[{source.name}] imágenes={source.images_zip.name} | etiquetas={tipo}\n"
            f"  -> total={len(samples)} (train={len(train_items)}, val={len(val_items)}, "
            f"sin_cajas={empty_frames}, cajas={sum(src_boxes.values())})"
        )

    yaml_path = write_data_yaml(args.output_dir, class_names)

    print("\n--- Resumen final ---")
    print(f"Total imágenes: {total_train + total_val} (train={total_train}, val={total_val})")
    print("Cajas por clase:")
    for name in class_names:
        print(f"  - {class_to_id[name]} ({name}): {total_boxes[name]}")
    print(f"Archivo de configuración generado: {yaml_path}")


if __name__ == "__main__":
    main()
