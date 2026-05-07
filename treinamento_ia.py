import streamlit as st
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np

# Configurações de Caminho - Ajuste se necessário
PATH_TRAIN_IMAGES = 'dataset/images/train'
PATH_TRAIN_LABELS = 'dataset/labels/train'

# Carregamento do Modelo
@st.cache_resource
def load_model():
    # Usando o seu melhor peso após o refino
    return YOLO('weights/best.pt')

model = load_model()

st.set_page_config(page_title="SafraMax - Coleta de Dados Real", layout="wide")
st.title("🌱 SafraMax: Validação de Campo")
st.write("Suba a foto enviada pelo produtor para validar o cérebro da IA.")

uploaded_file = st.file_uploader("Escolha uma imagem de café...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converter para formato OpenCV
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    # Executar Predição
    # Usamos imgsz=800 conforme seu último teste de sucesso
    results = model.predict(source=img_array, conf=0.25, imgsz=800)
    
    # Criar colunas para visualização e dados
    col_img, col_data = st.columns([2, 1])
    
    with col_img:
        res_plotted = results[0].plot(line_width=1, font_size=8)
        st.image(res_plotted, caption="Detecção do SafraMax", use_container_width=True)
        
    with col_data:
        st.header("Relatório de Campo")
        
        # Contagem de classes
        counts = {0: 0, 1: 0, 2: 0} # Verde, De Vez, Maduro
        for box in results[0].boxes:
            cls = int(box.cls[0])
            counts[cls] += 1
            
        st.metric("Verdes (Classe 0)", counts[0])
        st.metric("De Vez (Classe 1)", counts[1])
        st.metric("Maduros (Classe 2)", counts[2])
        st.subheader(f"Total: {sum(counts.values())} frutos")
        
        st.divider()
        st.write("### O resultado está correto?")
        
        c1, c2 = st.columns(2)
        if c1.button("✅ SIM", use_container_width=True):
            st.success("Excelente! O SafraMax está calibrado para esta propriedade.")
            
        if c2.button("❌ NÃO", use_container_width=True):
            # SALVAR PARA RE-TREINO
            fname = uploaded_file.name
            img_path = os.path.join(PATH_TRAIN_IMAGES, fname)
            label_path = os.path.join(PATH_TRAIN_LABELS, fname.replace(fname.split('.')[-1], 'txt'))
            
            # Salva a imagem original
            image.save(img_path)
            
            # Salva o rótulo atual (para você apenas ajustar depois no LabelImg)
            results[0].save_txt(label_path)
            
            st.warning("⚠️ Enviado para a fila de Refino! Imagem e rótulo salvos na pasta de treino.")
            st.info(f"Local: {PATH_TRAIN_IMAGES}")