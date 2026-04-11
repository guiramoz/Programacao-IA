#importando as libs
import sys
import cv2 #openCV é uma biblioteca de visão computacional que permite processar imagens e vídeos
from ultralytics import YOLO #lib responsável pelo reconhecimento facial/objetos usando o modelo YOLO (You Only Look Once)

#1 carregar o modelo YOLO v.8 
#na primeira execução, o arquivo dessa versão vai ser baixado
print("Carregando modelo...")
modelo = YOLO('yolov8n.pt')#carregando o modelo. O modelo é leve e rápido, adequado para aplicações em tempo real.

#2. Abrir uma conexão com a câmera
# Use 0 para webcam do PC, 1 ou 2 para outra webcam/iVCam, ou passe uma URL completa de stream HTTP/RTSP
# Exemplo iVCam como webcam virtual: python detectar.py 1
# Exemplo stream URL: python detectar.py http://192.168.15.5:8080/video
camera_source = sys.argv[1] if len(sys.argv) > 1 else "0"
if isinstance(camera_source, str) and camera_source.isdigit():
    camera_source = int(camera_source)
print(f"Usando fonte de vídeo: {camera_source}")
cap = cv2.VideoCapture(camera_source) # aceita índice local ou URL de stream
#numero 0 é a webcam integrada do computador. 
#numero 1 seria uma câmera externa conectada ao computador, e assim por diante.
#caso a via seja remota, o endereço de ip deve ser informado por uma variavel, por exemplo: cap = cv2.VideoCapture("http://



#verificar se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()
    
print("Iniciando a detecção. Pressione 'q' para sair.")

#3. loop de captura continua
while True:
    sucesso, frame = cap.read()
    
    #se o frame for capturado com sucesso
    if sucesso: #realizar a detecção (inference)
        frame = cv2.flip(frame, 1)  # inverte horizontalmente o vídeo
        results = modelo(frame, conf=0.5) #rodando o modelo de detecção com 50% de confiança
        annotated_frame = results[0].plot() #criar caixa visual na imagem para cada objeto detectado
        cv2.imshow("Visão computacional", annotated_frame) #exibir o frame na tela

        if cv2.waitKey(1) & 0xFF == ord('q'): #esperar por 1ms e verificar se a tecla 'q' foi pressionada para sair do loop
            break

    else:
        print("Erro ao capturar o frame.")
        break

#limpeza
cap.release() #liberar a câmera
cv2.destroyAllWindows() #fechar todas as janelas abertas pelo OpenCV



        