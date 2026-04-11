#importando das libs
#Comando único de instalação: pip install deepface opencv-python
import sys
import cv2  # openCV é uma biblioteca de visão computacional
from deepface import DeepFace  # reconhecimento facial
import time  # medir tempo

# Fonte de vídeo: use 0 para webcam do PC, 1 ou 2 para outra câmera/iVCam, ou passe uma URL completa de stream HTTP/RTSP
# Exemplo iVCam como webcam virtual: python porteiro.py 1
# Exemplo stream URL: python porteiro.py http://192.168.15.5:8080/video
camera_source = sys.argv[1] if len(sys.argv) > 1 else "0"
if isinstance(camera_source, str) and camera_source.isdigit():
    camera_source = int(camera_source)
print(f"Usando fonte de vídeo: {camera_source}")

# Passo 1: carregar a identidade
imagem_referencia = "face_id.jpg"
print("Carregando a identidade do morador...")

# pré análise da imagem
try:
    DeepFace.represent(img_path=imagem_referencia, model_name="VGG-Face")
    print("Identidade carregada com sucesso!")
except:
    print("Erro! Não encontrei ou não há rosto nele.")
    exit()

# Passo 2: abrir a câmera
cap = cv2.VideoCapture(camera_source)
print("Sistema de portaria ativo.")

status = None
status_color = (255, 255, 255)
last_check = 0
check_interval = 2.0  # segundos entre verificações automáticas

# Passo 3: loop contínuo
while True:
    ret, frame = cap.read()

    if not ret:
        print("Erro ao acessar a câmera.")
        break

    # reduzir tamanho (melhor performance)
    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # desenhar área
    height, width, _ = frame.shape
    cv2.rectangle(frame, (100, 100), (width - 100, height - 100), (255, 0, 0), 2)

    # verificação automática a cada intervalo
    if time.time() - last_check > check_interval:
        last_check = time.time()
        print("Verificando identidade automaticamente...")
        try:
            roi = frame[100:height-100, 100:width-100]
            resultado = DeepFace.verify(
                img1_path=roi,
                img2_path=imagem_referencia,
                model_name="VGG-Face",
                enforce_detection=False
            )

            if resultado["verified"]:
                status = "ACESSO LIBERADO"
                status_color = (0, 255, 0)
                print(">>>> ACESSO LIBERADO! >>>>")
            else:
                status = "NAO RECONHECIDO"
                status_color = (0, 0, 255)
                print("Acesso negado.")

        except Exception as e:
            status = "NENHUM ROSTO"
            status_color = (255, 255, 255)
            print("Erro na verificação:", e)

    # texto na tela
    cv2.putText(
        frame,
        "Olhe para a camera para identificar",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),2
    )

    if status:
        cv2.putText(
            frame,
            status,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )
        if status == "ACESSO LIBERADO":
            cv2.rectangle(frame, (100, 100), (width - 100, height - 100), status_color, 2)

    cv2.imshow("Portaria", frame)

    key = cv2.waitKey(1) & 0xFF

    # sair do sistema
    if key == ord('q'):
        print("Encerrando o sistema.")
        break

# corrigido (era 'camp')
cap.release()
cv2.destroyAllWindows()