from ultralytics import YOLO

def treinar_modelo():
    model = YOLO('yolo11n.pt') 

    model.train(
        data='cafe_data.yaml',
        epochs=100,            # Aumentamos para 100 para ela "olhar" mais vezes
        imgsz=800,
        batch=2,               # Número baixo de fotos por vez (bom para datasets pequenos)
        augment=True,          # Ativa a criação de novas versões das fotos
        device='cpu'           
    )

if __name__ == '__main__':
    treinar_modelo()