from ultralytics import YOLO

def entrenar():
    model = YOLO("yolo26l.pt")
    
    model.train(
        data="data/data.yaml",
        epochs=150,
        patience=25,
        imgsz=1024,
        batch=8, 
        device=0,
        workers=0,
        #cache=False,              # Fuerza a que lea del disco en cada iteración, salvando tu RAM
        optimizer="auto",
        project="TUIA-IET_runs",
        name="yolo26l_trafico_v1"
    )

if __name__ == "__main__":
    entrenar()