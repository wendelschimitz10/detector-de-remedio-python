import cv2

# LENDO A IMAGEM ORIGINAL
image = cv2.imread("img/remedio4.jpg")
#cv2.imshow("imagem original",image)

# ALTERANDO O TAMANHO DA IMAGEM ORIGINAL
img_resized = cv2.resize(image, None, fx= 0.17, fy=0.17)
#cv2.imshow("imagem com tamanho menor", img_resized)

cv2.waitKey()