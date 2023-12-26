# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    # Carrega nomes das classes
    with open("yolo/coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    print(class_names)

    # Carrega o modelo YOLO (Arquivo .weights = https://pjreddie.com/media/files/yolov3.weights)
    rede_dsa = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")


    # Função para obter as saídas dos nomes das camadas
    def get_output_layers(net):
        # Obtém os nomes de todas as camadas na rede neural
        layer_names = net.getLayerNames()

        # Cria uma lista dos nomes das camadas de saída. 'net.getUnconnectedOutLayers()' retorna os índices
        # das camadas de saída e esses índices são usados para acessar os nomes das camadas correspondentes
        # em 'layer_names'
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Retorna a lista dos nomes das camadas de saída
        return output_layers


    # Função para desenhar as caixas de detecção em uma imagem
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        # Converte o identificador da classe em uma string usando um array de nomes de classes
        label = str(class_names[class_id])

        # Gera uma cor aleatória para a caixa de detecção e o texto
        color = [float(c) for c in np.random.uniform(0, 255, size=(3,))]

        # Desenha uma caixa retangular na imagem 'img' com as coordenadas fornecidas e a cor gerada
        cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), color, 2)

        # Coloca um texto (nome da classe e confiança) na imagem, um pouco acima e à esquerda da caixa de detecção
        cv2.putText(img, label, (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Carregar uma imagem
    image = cv2.imread('dados/imagem1.jpg')
    # image = cv2.imread('dados/imagem2.png')
    # image = cv2.imread('dados/imagem3.jpg')
    # image = cv2.imread('dados/imagem4.png')
    # image = cv2.imread('dados/imagem5.png')

    # Extrai largura e altura
    width, height = image.shape[1], image.shape[0]

    # Escala de ajuste da imagem
    scale = 0.00392

    # Cria blob da imagem
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    rede_dsa.setInput(blob)

    # Realiza a detecção e obtém as saídas
    outs = rede_dsa.forward(get_output_layers(rede_dsa))

    # Parâmetros de controle
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Processa as saídas da rede neural
    for out in outs:

        # Itera sobre cada detecção na saída atual
        for detection in out:

            # Extrai as pontuações de confiança das classes a partir da detecção
            scores = detection[5:]

            # Encontra o índice da classe com a pontuação mais alta
            class_id = np.argmax(scores)

            # Obtém a pontuação de confiança para a classe com a pontuação mais alta
            confidence = scores[class_id]

            # Verifica se a pontuação de confiança está acima de um limiar definido
            if confidence > conf_threshold:
                # Calcula as coordenadas x e y do centro da caixa de detecção
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                # Calcula a largura e altura da caixa de detecção
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calcula as coordenadas x e y do canto superior esquerdo da caixa de detecção
                x = center_x - w / 2
                y = center_y - h / 2

                # Adiciona o índice da classe à lista de class_ids
                class_ids.append(class_id)

                # Adiciona a pontuação de confiança à lista de confidences
                confidences.append(float(confidence))

                # Adiciona a caixa de detecção à lista de boxes
                boxes.append([x, y, w, h])

    # Aplica non-max suppression para remover caixas redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Desenha as caixas finais na imagem
    for i in indices:
        box = boxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    # Converte a imagem BGR para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Exibe a imagem usando matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()