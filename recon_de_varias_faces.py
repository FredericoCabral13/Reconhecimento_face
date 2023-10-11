import cv2
import numpy as np
import face_recognition

img1_path = 'Elon-Bezos1.jpeg'
img2_path = 'Elon-Bezos-Mark.jpg'

img1_matiz = face_recognition.load_image_file(img1_path)
img1_matiz = cv2.cvtColor(img1_matiz,cv2.COLOR_BGR2RGB)

img2_matiz = face_recognition.load_image_file(img2_path)
img2_matiz = cv2.cvtColor(img2_matiz, cv2.COLOR_BGR2RGB)

cod1 = face_recognition.face_encodings(img1_matiz)
cod2 = face_recognition.face_encodings(img2_matiz)

pessoas = {}
for i in range(0,len(cod2)):
    # cod1[i]
    pessoas[f'pessoa {i+1}'] = cod2[i]
    for j in range(0,len(cod1)):
        comp = face_recognition.compare_faces([cod2[i]],cod1[j])
        if (comp == [True]):
            print(list(pessoas.keys())[i])
        # print(comp)
    
# print(pessoas[list(pessoas.keys())[1]])
print(pessoas.keys())