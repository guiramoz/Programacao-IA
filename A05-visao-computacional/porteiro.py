#importando das libs
import cv2  # openCV é uma biblioteca de visão computacional
from deepface import DeepFace  # reconhecimento facial
import time  # medir tempo

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
cap = cv2.VideoCapture(0)
print("Sistema de portaria ativo.")

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

    # texto na tela
    cv2.putText(
        frame,
        "Pressione V para verificar o acesso",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Portaria", frame)

    key = cv2.waitKey(1) & 0xFF

    # verificar identidade
    if key == ord('v'):
        print("Verificando identidade...")
        start_time = time.time()

        try:
            resultado = DeepFace.verify(
                img1_path=frame,
                img2_path=imagem_referencia,
                model_name="VGG-Face",
                enforce_detection=False
            )

            if resultado["verified"]:
                print(">>>> ACESSO LIBERADO! >>>>")
                cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
            else:
                print("Acesso negado.")
                cv2.rectangle(frame, (0, 0), (width, height), (255, 0, 0), 2)

            cv2.imshow("Portaria", frame)
            cv2.waitKey(2000)

        except Exception as e:
            print("Erro na verificação:", e)

    # sair do sistema
    if key == ord('q'):
        print("Encerrando o sistema.")
        break

# corrigido (era 'camp')
cap.release()
cv2.destroyAllWindows()