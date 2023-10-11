import cv2
import numpy as np
import face_recognition


imgelon_bgr = face_recognition.load_image_file('Elon-Musk1.jpg') #matriz com cada píxel gbr (azul)
imgelon_rgb = cv2.cvtColor(imgelon_bgr,cv2.COLOR_BGR2RGB) #matriz com cada píxel rgb (normal)
# cv2.imshow('bgr', imgelon_bgr) #imagem com elementos em azul
# cv2.imshow('rgb', imgelon_rgb)
# cv2.waitKey(0)

imgelon = face_recognition.load_image_file('Elon-Musk1.jpg') #matriz com cada píxel gbr (azul)
imgelon = cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB) #matriz com cada píxel rgb (normal)

#----------Encontrando a localização da face para desenhar caixas delimitadoras-------
face = face_recognition.face_locations(imgelon_rgb)[0] #[(134, 669, 455, 348)] --> (134, 669, 455, 348)
copy = imgelon.copy() #copiando a imagem e salvando na variável copy

#-------------------Desenhando o retângulo-------------------------
cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2) #atribuindo à imagem copy o desenho de um retângulo (com as devidas dimensões)
cv2.imshow('copy', copy)
# cv2.imshow('elon',imgelon)
cv2.waitKey(0)
# print(imgelon)