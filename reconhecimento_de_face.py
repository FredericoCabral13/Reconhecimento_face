import cv2
import numpy as np
import face_recognition

'''-------------Transformando a imagem em matrizes bgr e rgb-------------------------------'''
imgelon_bgr = face_recognition.load_image_file('Elon-Musk1.jpg') #matriz com cada píxel gbr (azul)
imgelon_rgb = cv2.cvtColor(imgelon_bgr,cv2.COLOR_BGR2RGB) #matriz com cada píxel rgb (normal)
# cv2.imshow('bgr', imgelon_bgr) #imagem com elementos em azul
# cv2.imshow('rgb', imgelon_rgb)
# cv2.waitKey(0)

'''-------------Encontrando a localização da face para desenhar caixas delimitadoras------------'''
face = face_recognition.face_locations(imgelon_rgb)[0] #[(134, 669, 455, 348)] --> (134, 669, 455, 348)
copy = imgelon_rgb.copy() #copiando a imagem e salvando na variável copy

'''-------------Desenhando o retângulo----------------------------------------------------------'''
cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2) #atribuindo à imagem copy o desenho de um retângulo (com as devidas dimensões)
# cv2.imshow('copy', copy)
# cv2.imshow('elon',imgelon_rgb)
# cv2.waitKey(0)
# print(imgelon)

'''-------------Treinando a imagem para reconhecimento facial---------------------------------'''
#codificando a imagem para essa codificação ser comparada mais tarde com a de outra imagem
treino_img = face_recognition.face_encodings(imgelon_rgb)[0] #array([-8.60052928e-02, .... ] --> [-8.60052928e-02, .... ]


'''-------------Treinando com uma nova imagem--------------------------------------------------'''
#realizar novamente os processos anteriores (menos retirando os da face até o retângulo) para a nova imagem
img_teste = face_recognition.load_image_file('Elon-Musk2.jpg')
img_teste = cv2.cvtColor(img_teste, cv2.COLOR_BGR2RGB)
teste_img = face_recognition.face_encodings(img_teste)[0]
#Agora sim, comparando com a codificação da imagem de treino
comparacao = face_recognition.compare_faces([treino_img],teste_img)
print(comparacao) #True (significa que é a mesma pessoa)