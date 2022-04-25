import cv2
from cv2 import THRESH_OTSU

# LENDO A IMAGEM ORIGINAL
image = cv2.imread("img/remedio4.jpg")

# ALTERANDO O TAMANHO DA IMAGEM ORIGINAL
img_resized = cv2.resize(image, None, fx= 0.1, fy=0.1)
cv2.imshow("imagem alterada", img_resized)

# ALTERANDO O FILTRO BILATERAL NA IMAGEM
image_filtered = cv2.bilateralFilter(img_resized,30, 255,255)
cv2.imshow("imagem gaussian", image_filtered)

# BINARIZANDO A IMAGEM
method = cv2.THRESH_BINARY_INV
limiar, image_binary = cv2.threshold(image_filtered, 100, 255, method)

img_final = cv2.Canny(image_binary, 255,255)
cv2.imshow("img final", img_final)


cv2.waitKey()