import cv2
from cv2 import THRESH_OTSU

# LENDO A IMAGEM ORIGINAL
image = cv2.imread("img/remedio8.jpeg")

# ALTERANDO O TAMANHO DA IMAGEM ORIGINAL
img_resized = cv2.resize(image, None, fx= 0.4, fy=0.4)

# ALTERANDO A COR DA IMAGEM PARA CINZA
image_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# ALTERANDO O FILTRO BILATERAL NA IMAGEM
image_filtered = cv2.bilateralFilter(image_gray,20,255,255)
#cv2.imshow("imagem binarizada", image_filtered)

# BINARIZANDO A IMAGEM
method = cv2.THRESH_BINARY_INV + THRESH_OTSU
limiar, image_binary = cv2.threshold(image_filtered, 0, 255, method)

# FAZENDO O PROCESSO DE EROSÃO NA IMAGEM
ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
image_erodida = cv2.erode(image_binary, ee, iterations=10)
#cv2.imshow("imagem erodida", image_erodida)

# DILATANDO A IMAGEM
image_dilatada = cv2.dilate(image_erodida, ee, iterations=10)
cv2.imshow("imagem dilatada", image_dilatada)

# APLICANDO BORDAS FINAS NOS PONTOS DE INTERESSE
image_contornos = cv2.Canny(image_dilatada,150,150)
#cv2.imshow("imagem com contornos", image_contornos)

contornos, h = cv2.findContours(image_contornos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'teste: {len(contornos)} {len(h)}')

phrase = f'Remedios faltantes: {len(contornos)}'
cv2.putText(image_contornos, phrase, (40,500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)

cv2.imshow("teste",image_contornos)




cv2.waitKey()