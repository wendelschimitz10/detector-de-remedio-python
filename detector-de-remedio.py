import cv2
from cv2 import THRESH_OTSU

# LENDO A IMAGEM ORIGINAL
image = cv2.imread("img/remedio8.jpeg")
#cv2.imshow("1 - Imagem original", image)

# ALTERANDO O TAMANHO DA IMAGEM ORIGINAL
img_resized = cv2.resize(image, None, fx=0.4, fy=0.4)
cv2.imshow("2 - Imagem com tamanho alterado", img_resized)

# ALTERANDO A COR DA IMAGEM PARA CINZA
image_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
#cv2.imshow("3 - Imagem cinza", image_gray)

# ALTERANDO O FILTRO BILATERAL NA IMAGEM
image_filtered = cv2.bilateralFilter(image_gray,20,255,255) # tentativa e erro
#cv2.imshow("4 - Imagem com filtro Bilateral", image_filtered)

# BINARIZANDO A IMAGEM
method = cv2.THRESH_BINARY_INV + THRESH_OTSU
limiar, image_binary = cv2.threshold(image_filtered, 0, 255, method) # tentativa e erro
#cv2.imshow("5 - Imagem Binarizada", image_binary)

# FAZENDO O PROCESSO DE EROSÃO NA IMAGEM
ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
image_erodida = cv2.erode(image_binary, ee, iterations=10) # tentativa e erro
#cv2.imshow("6 - imagem Erodida", image_erodida)

# DILATANDO A IMAGEM
image_dilatada = cv2.dilate(image_erodida, ee, iterations=10) # tentativa e erro
cv2.imshow("7 - Imagem Dilatada", image_dilatada)

# PROCURANDO CONTORNOS NA IMAGEM
contornos, h = cv2.findContours(image_dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f'teste: {len(contornos)} {len(h)}')

# COLOCAR O TEXTO NA IMAGEM ORIGINAL (RESIZED)
phrase = f'Remedios faltantes: {len(contornos)-2}'
cv2.putText(img_resized, phrase, (40,500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1)

# APLICANDO UMA LÓGICA PARA COMEÇAR A PEGAR O CONTORNO DE DENTRO DA IMAGEM
for i in range(2, len(contornos)):
    image_final = cv2.drawContours(img_resized, contornos, i, (0,0,255),2)
    cv2.imshow('Final contornos', image_final)

cv2.waitKey()
