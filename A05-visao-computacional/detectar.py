#importando as libs
import cv2 
from ultralytics import YOLO

#1 carregra o modelo YOLO v.8 
#na primeira execução, o arquivo dessa versão vai ser baixado
print("Carregando modelo...")
modelo = YOLO('yolov8n.pt')

#2. conectar uma câmera ao código
endereco_ip = "http://"

#verificar se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()
    
print("Iniciando a detecção. Pressione 'q' para sair.")

#3. loop de captura continua
while True:
    sucesso, frame = cap.read()
    
    #se o frame for capturado com sucesso
    if sucesso:
        results = model(frame, conf=0.5) #rodando o modelo de detecção com 50% de confiança
        annotated_frame = results[0].plot()