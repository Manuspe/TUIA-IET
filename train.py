from ultralytics import YOLO

def entrenar():
    model = YOLO("yolo26l.pt")

    # Configuración ajustada para RTX 5060 (8 GB VRAM), 32 GB RAM y Ryzen 7 5700 (8C/16T):
    # - batch=8 + amp=True mantiene el uso de VRAM seguro dentro de 8 GB con yolo26l a 640px
    #   (Ultralytics compensa el batch con acumulación de gradiente via nbs=64).
    # - workers=4 aprovecha el Ryzen 7 5700 sin saturar memoria ni procesos en Windows/Linux.
    # - cache="ram" aprovecha los 32 GB de RAM (~1.6 GB para ~1300 imgs a 640px) para acelerar épocas.
    model.train(
        data="data/data.yaml",
        epochs=150,
        patience=25,
        imgsz=640,
        batch=16,    #8
        amp=True,
        device=0,
        workers=4,
        cache="ram",
        optimizer="auto",
        project="TUIA-IET_runs",
        name="yolo26l_trafico_v1",
    )

if __name__ == "__main__":
    entrenar()