# TUIA-IET

Proyecto de Computer Vision para detección, seguimiento y aforo vehicular y peatonal (`car`, `motorcycle`, `bicycle`, `person`) sobre videos de tránsito.

## Flujo general del proyecto

1. **Auto-etiquetado (`auto_label_video.py`)**: extrae frames de un video y genera un `.zip` en formato **CVAT 1.1 for images** (`images/` + `annotations.xml`) para revisar y corregir en CVAT.
2. **Armado y split del dataset (`split_datasets.py`)**: combina múltiples pares `*_dataset.zip` (imágenes) + `.zip` corregidos de CVAT (`annotations.xml`), convierte a formato YOLO y genera `data/` con split estratificado 80/20 por fuente y `data.yaml`.
3. **Entrenamiento (`train.py`)**: entrena el modelo YOLO (`yolo26l.pt`) sobre `data/data.yaml`.
4. **Inferencia con zona de exclusión (`predict.py`)**: prueba visual rápida ignorando detecciones dentro de un polígono (por ejemplo, autos estacionados).
5. **Aforo y exportación (`conteo_trafico.py`)**: conteo por líneas virtuales con ROI válida, tracking `ByteTrack`, video anotado opcional y tabla consolidada en `.csv`.

---

## Estructura del repositorio

- `auto_label_video.py`: extracción de frames y pre-etiquetado automático para CVAT.
- `split_datasets.py`: armado de dataset YOLO multi-fuente desde ZIPs de `auto_label_video.py` + CVAT.
- `split_dataset.py`: script simple legado para dividir una carpeta suelta de imágenes y `.txt` ya extraídos.
- `train.py`: entrenamiento con Ultralytics YOLO.
- `predict.py`: inferencia interactiva con polígono de exclusión.
- `conteo_trafico.py`: conteo de cruces por clase sobre 4 líneas virtuales y exportación a CSV.
- `requirements.txt`: dependencias base (`ultralytics`, `opencv-python`).

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para ejecutar `conteo_trafico.py` también se requiere `supervision` y `trackers` (en `supervision >= 0.31.0`, `ByteTrack` pasó al paquete `trackers`):

```bash
python -m pip install supervision trackers
```

---

## 1. Auto-etiquetado para CVAT (`auto_label_video.py`)

Procesa un video frame cada `--frame-step` fotogramas, ejecuta detección sobre el frame completo con YOLO y genera un `.zip` listo para importar en CVAT.

```bash
python auto_label_video.py video.mp4 \
    --model yolo26x.pt \
    --conf 0.22 \
    --iou 0.7 \
    --imgsz 960 \
    --frame-step 30 \
    --classes car motorcycle bicycle person \
    --device 0 \
    --output Nombre_dataset.zip
```

Luego en CVAT se importan las imágenes/anotaciones, se corrigen manualmente las cajas y se exportan las anotaciones como `.zip` (`CVAT for images 1.1`).

---

## 2. Preparación del dataset YOLO (`split_datasets.py`)

Combina automáticamente los `*_dataset.zip` (fuente de imágenes) con sus `.zip` exportados de CVAT (fuente de etiquetas corregidas), normalizando mayúsculas y guiones bajos (por ejemplo `IMAE1730_dataset.zip` + `Imae_1730.zip`).

- Omite por defecto los `*_dataset.zip` que aún no tienen `.zip` de CVAT asociado (usar `--include-auto-only` para incluirlos con sus etiquetas automáticas).
- Prefija cada frame con el nombre de su fuente (`IMAE1730__frame_000000.jpg`/`.txt`) para evitar colisiones.
- Realiza un split estratificado `80% train / 20% val` dentro de cada fuente y genera `data/data.yaml` con ruta absoluta lista para Ultralytics.

```bash
python split_datasets.py
```

Opciones principales:

- `--input-dir`: carpeta con los `.zip` (por defecto `.`).
- `--output-dir`: carpeta destino (por defecto `data`).
- `--train-ratio`: proporción para `train` por fuente (por defecto `0.8`).
- `--seed`: semilla aleatoria (por defecto `42`).
- `--include-auto-only`: incluye `*_dataset.zip` sin ZIP de CVAT.
- `--keep-existing`: conserva `train/` y `val/` previos en lugar de regenerarlos limpios.
- `--pair img_dataset.zip=cvat.zip`: emparejamiento manual explícito (repetible).

---

## 3. Entrenamiento (`train.py`)

Entrena `yolo26l.pt` usando `data/data.yaml` con configuración ajustada para GPU de 8 GB VRAM (RTX 5060), 32 GB de RAM y Ryzen 7 5700 (`imgsz=640`, `batch=8`, `amp=True`, `workers=4`, `cache="ram"`, `epochs=150`, `patience=25`):

```bash
python train.py
```

Los resultados y pesos se guardan bajo `runs/detect/TUIA-IET_runs/`.

---

## 4. Inferencia rápida con zona de exclusión (`predict.py`)

Permite trazar un polígono sobre el primer frame (clic izquierdo para agregar puntos, clic derecho para borrar el último, `ENTER` para confirmar) e ignora toda detección cuyo centro caiga dentro de esa zona:

```bash
python predict.py
```

*(La ruta del video y de los pesos se configuran al final de `predict.py`; presiona `q` para cerrar la ventana).*

---

## 5. Aforo vehicular y peatonal (`conteo_trafico.py`)

Realiza detección, filtrado espacial por ROI (`BOTTOM_CENTER`), seguimiento con `ByteTrack(lost_track_buffer=60)` y conteo sin duplicados por par `(tracker_id, línea)` sobre 4 líneas virtuales.

```bash
python conteo_trafico.py prueba_1.mp4 \
    --weights runs/detect/TUIA-IET_runs/yolo26l_trafico_v1-3/weights/best.pt \
    --conf 0.35 \
    --imgsz 640 \
    --device 0 \
    --save-video
```

### Flujo interactivo inicial (sobre frame alrededor de 1.5 s)
1. **Paso A — Polígono ROI (zona válida)**: clic izquierdo agrega vértices, clic derecho deshace, `ENTER` o `ESPACIO` confirma (menos de 3 puntos toma todo el frame).
2. **Paso B — 4 Líneas virtuales**: se trazan secuencialmente `Ingreso`, `Egreso`, `Ingreso/Egreso 1` y `Ingreso/Egreso 2` (2 clics izquierdos por línea, clic derecho reinicia la línea actual, `ENTER` o `ESPACIO` confirma).

### Flags opcionales y salidas
- `--no-show`: procesa sin abrir la ventana de visualización durante la inferencia.
- `--save-video`: guarda `<nombre_video>_aforo.mp4` junto al video original.
- Al terminar (o al salir con `q`), exporta automáticamente `<nombre_video>.csv` (UTF-8) en la misma carpeta del video con la tabla consolidada por clase (`car`, `motorcycle`, `bicycle`, `person`) y fila `TOTAL`:
  `Clase,Ingreso,Egreso,Ingreso/Egreso 1,Ingreso/Egreso 2,Total Ingreso/Egreso,Total General`
