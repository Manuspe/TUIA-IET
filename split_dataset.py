import os
import random
import shutil
from pathlib import Path

def crear_split_yolo(ruta_origen, ruta_destino_base, porcentaje_train=0.8):
    origen = Path(ruta_origen)
    destino = Path(ruta_destino_base)
    
    # Extensiones de imagen que buscará el script
    extensiones_validas = ['.jpg', '.jpeg', '.png']
    
    # Listar todas las imágenes en la carpeta de origen
    imagenes = [f for f in origen.iterdir() if f.is_file() and f.suffix.lower() in extensiones_validas]
    
    if not imagenes:
        print(f"No se encontraron imágenes en {origen}. Revisa la ruta.")
        return

    # Mezclar las imágenes aleatoriamente
    random.seed(42) # Fijamos una semilla para que el split sea reproducible si lo corres de nuevo
    random.shuffle(imagenes)
    
    # Calcular cantidad de imágenes para train y val
    corte = int(len(imagenes) * porcentaje_train)
    imagenes_train = imagenes[:corte]
    imagenes_val = imagenes[corte:]
    
    # Definir y crear la estructura de carpetas de YOLO en el destino
    carpetas_yolo = [
        destino / 'train' / 'images',
        destino / 'train' / 'labels',
        destino / 'val' / 'images',
        destino / 'val' / 'labels'
    ]
    
    for carpeta in carpetas_yolo:
        carpeta.mkdir(parents=True, exist_ok=True)
        
    def mover_archivos(lista_imagenes, subconjunto):
        for img_path in lista_imagenes:
            # Deducir la ruta del .txt correspondiente
            txt_path = img_path.with_suffix('.txt')
            
            # Rutas de destino final
            dest_img = destino / subconjunto / 'images' / img_path.name
            dest_txt = destino / subconjunto / 'labels' / txt_path.name
            
            # Mover la imagen
            shutil.move(str(img_path), str(dest_img))
            
            # Mover el .txt (se verifica que exista por si hay imágenes sin etiquetas)
            if txt_path.exists():
                shutil.move(str(txt_path), str(dest_txt))

    print(f"Total de imágenes encontradas: {len(imagenes)}")
    print(f"Moviendo {len(imagenes_train)} imágenes y sus etiquetas a 'train'...")
    mover_archivos(imagenes_train, 'train')
    
    print(f"Moviendo {len(imagenes_val)} imágenes y sus etiquetas a 'val'...")
    mover_archivos(imagenes_val, 'val')
    
    print("¡División completada con éxito! Ya puedes borrar la carpeta original si quedó vacía.")

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================

# Cambia esto por el nombre de la carpeta donde están todas tus imágenes y .txt sueltos 
# (por ejemplo, la carpeta "obj_train_data" que extrajiste del ZIP)
CARPETA_ORIGEN = "dataset/images"

# Carpeta base donde se crearán las subcarpetas train/ y val/
# (Generalmente es la raíz de tu "dataset", donde pusiste el data.yaml)
CARPETA_DESTINO = "data"

# Ejecutar la función
crear_split_yolo(CARPETA_ORIGEN, CARPETA_DESTINO, porcentaje_train=0.8)