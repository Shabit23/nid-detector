import cv2
import numpy as np
import face_recognition
import easyocr
import matplotlib.pyplot as plt
import os
from datetime import datetime
# import pytesseract

path = 'ScannedImages'
imgPath = []
imageList = os.listdir(path)
pathTest = './ScannedImages/b1.JPG'
#
for cl in imageList:
    imgPath.append('./' + path + '/' + cl)
print(imgPath)
#
#
# def recognize_text(images):
#     #text_list = []
#     reader = easyocr.Reader(['en'])
#    # text_list.append(reader.readtext(images[0]))
#     return reader.readtext(images)
#
#
# textKnown = recognize_text(pathTest)
# print(textKnown)

# C:\Program Files (x86)\Tesseract-OCR

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
roi = [[(380, 282), (970, 353), 'Name'], [(438, 538), (973, 602), 'Nid No.']]

for j, y in enumerate(imgPath):
    imgScan = cv2.imread(y)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []
    print(f'################ Extracting Data from Card {j+1} ################')
    # cv2.imshow("output", imgMask)
    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # cv2.imshow(str(x), imgCrop)

        reader = easyocr.Reader(['en'])
        print(f'{r[2]}: {reader.readtext(imgCrop)[0][1]}')
        myData.append(reader.readtext(imgCrop)[0][1])


    with open('Record.csv', 'a+') as file:
        for data in myData:
            file.write((str(data)+','))
        file.write('\n')


    print(myData)
    cv2.imshow(y, imgShow)
    cv2.waitKey(0)
