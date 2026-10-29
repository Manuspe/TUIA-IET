# TUIA-IET

Proyecto de investigación de Computer Vision.  
Etiquetado automático de videos para datasets de imágenes.

`auto_label_video.py` recorre un video, permite seleccionar una región de interés (ROI), ejecuta detección de objetos con Ultralytics YOLO aplicando un paso de fotogramas (`--frame-step`), extrae las imágenes relevantes y genera un paquete comprimido (`.zip`) listo para importar directamente en CVAT como dataset de imágenes (`CVAT 1.1 for images`).

---

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

El modelo se descarga automaticamente la primera vez. Para usar YOLO26 XL, el
nombre de pesos esperado es `yolo26x.pt`; tambien se puede pasar una ruta local
con `--model`.

### Uso interactivo

```powershell
python auto_label_video.py .\video.mp4 --model yolo26x.pt --output .\cvat_dataset.zip
```

Se abre una ventana de OpenCV: dibuja la ROI y pulsa Enter. Para evitar la
ventana, pasa la ROI directamente en pixeles:

```powershell
python auto_label_video.py .\video.mp4 `
    --roi 120 80 1400 900 `
    --classes person car `
    --conf 0.35 `
    --frame-step 30 `
    --output .\cvat_dataset.zip
```

En CVAT, importa `cvat_annotations.xml` como anotaciones de una tarea de video.
Revisa especialmente las cajas con baja confianza y los cambios de identidad
del tracking. `--roi-mode intersection` acepta cualquier caja que toque la ROI;
el modo predeterminado `center` suele producir menos falsos positivos en los
bordes.

### Notas de rendimiento

YOLO26 XL requiere una GPU con memoria suficiente. Si aparece un error de
memoria, baja `--imgsz`, usa `--device cpu` solo para pruebas, o procesa cada
dos/cinco frames con `--frame-step`. Por ejemplo, `--frame-step 5` procesa los
frames 0, 5, 10, etc. y conserva el numero real de frame en el XML, por lo que
CVAT los ubica correctamente en el video original. El salto reduce el costo y
la redundancia entre frames, pero para evitar overfitting tambien conviene
seleccionar escenas variadas y separar algunos videos para validacion. La
deteccion solo conserva objetos cuyo centro o interseccion, segun el modo
elegido, cae dentro de la ROI.
