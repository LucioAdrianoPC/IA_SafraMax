from ultralytics import YOLO

# 1. Carrega o novo peso gerado após as 300 épocas
# Verifique se a pasta será 'train-6' ou similar
model = YOLO(r'D:\projetos\SafraMax_IA\runs\detect\train-14\weights\best.pt')

# 2. Roda a predição
results = model.predict(source=r'D:\projetos\SafraMax_IA\dataset\images\val\cafe_02.png', save=True, conf=0.25)

# 3. Lógica de Contagem para o SafraMax
for result in results:
    count_verde = (result.boxes.cls == 0).sum().item()
    count_devez = (result.boxes.cls == 1).sum().item()
    count_maduro = (result.boxes.cls == 2).sum().item()
    
    print(f"--- RELATÓRIO SAFRAMAX ---")
    print(f"Verdes: {count_verde}")
    print(f"De Vez: {count_devez}")
    print(f"Maduros: {count_maduro}")
    print(f"Total de Frutos: {len(result.boxes)}")