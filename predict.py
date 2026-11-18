import cv2
import numpy as np
from ultralytics import YOLO

puntos_poligono = []

def capturar_puntos(event, x, y, flags, param):
    global puntos_poligono
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_poligono.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if puntos_poligono:
            puntos_poligono.pop()

def definir_zona_exclusion(frame):
    """Interfaz para trazar el polígono del estacionamiento que se ignorará."""
    global puntos_poligono
    puntos_poligono.clear()
    
    frame_copia = frame.copy()
    nombre_ventana = "Seleccionar Zona a IGNORAR (Click Izq: Agregar, Click Der: Borrar, ENTER: Confirmar)"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(nombre_ventana, capturar_puntos)

    while True:
        temp_frame = frame_copia.copy()
        
        if len(puntos_poligono) > 0:
            for pt in puntos_poligono:
                cv2.circle(temp_frame, pt, 4, (0, 0, 255), -1)
            for i in range(len(puntos_poligono) - 1):
                cv2.line(temp_frame, puntos_poligono[i], puntos_poligono[i+1], (0, 0, 255), 2)
            if len(puntos_poligono) > 2:
                cv2.line(temp_frame, puntos_poligono[-1], puntos_poligono[0], (0, 0, 255), 2)
                
                # Colorear la zona de exclusión de rojo semitransparente
                overlay = temp_frame.copy()
                pts = np.array(puntos_poligono, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                cv2.addWeighted(overlay, 0.3, temp_frame, 0.7, 0, temp_frame)
        
        cv2.imshow(nombre_ventana, temp_frame)
        if cv2.waitKey(1) & 0xFF == 13: # Tecla ENTER
            break

    cv2.destroyWindow(nombre_ventana)
    
    if len(puntos_poligono) > 2:
        return np.array(puntos_poligono, np.int32)
    return None

def predecir_video(ruta_video, ruta_pesos):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"Error al abrir el video: {ruta_video}")
        return
        
    ret, primer_frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame.")
        return
        
    # 1. El usuario traza la zona que no quiere predecir (estacionamiento)
    poligono_exclusion = definir_zona_exclusion(primer_frame)
    
    # 2. Cargar el modelo entrenado con los mejores pesos
    model = YOLO(ruta_pesos)
    
    # Reiniciar el video al primer frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inferencia a 1024p utilizando la RTX 5070 Ti
        resultados = model.predict(frame, conf=0.3, imgsz=1024, device=0, verbose=False)[0]
        
        # Filtrar y dibujar resultados
        for box_tensor, conf_tensor, cls_tensor in zip(resultados.boxes.xyxy, resultados.boxes.conf, resultados.boxes.cls):
            x1, y1, x2, y2 = map(int, box_tensor.tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            # Si el centro del bounding box está dentro del polígono, se ignora
            if poligono_exclusion is not None:
                if cv2.pointPolygonTest(poligono_exclusion, (cx, cy), False) >= 0:
                    continue 
            
            # Dibujar detecciones válidas
            etiqueta = f"{resultados.names[int(cls_tensor)]} {float(conf_tensor):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Dibujar la línea roja del polígono como referencia visual
        if poligono_exclusion is not None:
            cv2.polylines(frame, [poligono_exclusion], True, (0, 0, 255), 2)
            
        cv2.imshow("Inferencia - Presiona 'q' para salir", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ruta donde se guardaron los pesos de tu modelo finalizado
    ruta_mejor_modelo = "runs/detect/TUIA-IET_runs/yolo26l_trafico_v1/weights/best.pt"
    
    predecir_video("prueba_1.mp4", ruta_mejor_modelo)