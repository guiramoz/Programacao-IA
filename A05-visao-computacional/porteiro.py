#importando das libs
import cv2 #openCV é uma biblioteca de visão computacional que permite processar imagens e vídeos
from deepface import DeepFace #lib responsável pelo reconhecimento facial usando o modelo DeepFace
import time #lib para medir o tempo de execução do código

#Passo 1: carregar a identidade 
imagem_referencia = "face_id.jpg" #caminho para a imagem de referência
print("Carregando a identidade do morador...")

#pré análise da imagem, pra garantir que a foto de referência é válida
try:
    DeepFace.represent(img_path=imagem_referencia, model_name="VGG-Face") #tenta processar a imagem de referência para verificar se é válida
    print("Identidade carregada com sucesso!")
except:
    print("Erro! Não encontrei ou não há rosto nele.")
    exit() #encerra o programa se a imagem de referência for inválida

#Passo 2: abrir a câmera e iniciar a detecção
cap = cv2.VideoCapture(0) #0 é o índice da câmera padrão (geralmente a webcam integrada).
print("Siatema de portaria ativo.")

#Passo 3: loop de captura contínua
while True:
    ret, frame = cap.read() #ret retorna true se a foto foi tirada, frame recebe a imagem
    if not ret:
        print("Erro ao acessar a câmera.")
        break

        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) #redimensiona o frame na altura e largura para 50% do tamanho original, para acelerar o processamento   

        #desenhar um retângulo no frame para indicar a area de leitura
        height, width, _ = frame.shape
        cv2.rectangle(frame, (100,100), (width-100, height-100), (255,0,0), 2) #desenha um retângulo verde no frame para indicar a área de leitura do rosto
        #tamanho, cor e espessura da linha 

        #verificação da imagem com o rosto detectado
        cv2.putText(frame, "Pressione V para verificar o acesso", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 
        #texto, posição, fonte, tamanho, cor e espessura da linha
        key = cv2.waitKey(1) & 0xFF #espera por 1ms e verifica se a tecla 'v' foi pressionada para iniciar a verificação de acesso

        if key & 0xFF == ord('v'):
            print("Verificando identidade...")
            start_time = time.time() #inicia a contagem do tempo de execução
            try:
                resultado = DeepFace.verify(
                    img1_path = frame, #quem está na câmera
                    img2_path = imagem_referencia, #foto capturada 
                    model_name="VGG-Face", #compara a imagem de referência com o frame capturado
                    enforce_detection = False #tente detectar o rosto, mas se não conseguir não lance um erro
                ) 
                
                #se resultado é verdadeiro (acesso liberado)
                if resultado["verified"]:
                    print(">>>>ACESSO LIBERADO!>>>>>")
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
                    cv2.imshow("Portaria", frame) #exibe o frame com a borda verde indicando acesso liberado
                    cv2.waitKey(2000) #espera por 2 segundos para mostrar a borda verde 
                
                else:
                    print("Acesso negado. Identidade não reconhecida.")
                    cv2.rectangle(frame, (0, 0), (width, height), (255, 0, 0), 2)
                    cv2.imshow("Portaria", frame) #exibe o frame com a borda vermelha indicando acesso negado
                    cv2.waitKey(2000) #espera por 2 segundos para mostrar a borda vermelha

            except Exception as e:
                print("Portaria", frame)

            if key & 0xFF == ord('q'): #espera por 1ms e verifica se a tecla 'q' foi pressionada para sair do loop
                print("Encerrando o sistema de portaria.")
                break

        camp.release() #libera a câmera
        cv2.destroyAllWindows() #fecha todas as janelas abertas pelo OpenCV
               

