import cv2
import numpy as np
import face_recognition
import easyocr
import matplotlib.pyplot as plt
import os
from datetime import datetime


# Path and necessary variables declaration
path = 'ScannedImages'
images = []
classNames = []
imageList = os.listdir(path)
imgPath = []

# Individual path to /ScannedImages/`image.JPG` folder listed in imageList
for cl in imageList:
    imgPath.append('./' + path + '/' + cl)

print(imgPath)
print(imageList)


# Function to create a list of all the encodings of the faces found in images
def encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


# Function to recognize required text from images
def text_recognition(imgPath, imageList):
    # Region of interest to extract text from images
    roi = [[(380, 282), (970, 353), 'Name'], [(438, 538), (973, 602), 'Nid No.']]
    myData = []

    # for s in imageList:
    #     print(s[0])
    #     if s[0] != '$':
    for j, y in enumerate(imgPath):
        if y[16] != '$':
            imgScan = cv2.imread(y)
            imgShow = imgScan.copy()
            imgMask = np.zeros_like(imgShow)
            tempData = []

            print(f'################ Extracting Data from Card {j + 1} ################')
            # cv2.imshow("output", imgMask)
            for x, r in enumerate(roi):
                cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
                imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

                imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                # cv2.imshow(str(x), imgCrop)

                reader = easyocr.Reader(['en'])
                print(f'{r[2]}: {reader.readtext(imgCrop)[0][1]}')
                tempData.append(reader.readtext(imgCrop)[0][1])

            myData.append(tempData)
    return myData                       # myData holds the texts extracted from images i.e. Name and NID No.
    # return []


# Function to record the detected face through webcam
def record(name, nid_number):
    with open('Detected_Record.csv', 'r+') as file:
        my_data_list = file.readlines()
        name_list = []
        nid_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
            nid_list.append(entry[0])

        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S')
            file.writelines(f'\n{name},{nid_number},{date_string}')


# Function to record data extracted from image i.e. Name and NID no.
def record_data(textKnown):
    with open('Record.csv', 'a+') as file:
        for data in textKnown:
            for row in data:
                file.write(row + ',')
            file.write('\n')


# Function text_recognition called
textKnown = text_recognition(imgPath, imageList)
# test = [['MD. KHALEQUZZAMAN', '2690243815978'], ['M. S ZAMAN SHABIT', '6907626193'], ['UMME ZAKIA SAJAL', '32072191']]
print(textKnown)

record_data(textKnown)

# Creation of new image path to rename the images in `$ Name_NID.JPG` format
newImgPath = []

for s in imageList:
    if s[0] != '$':
        for horizontal in textKnown:
            new_name = './' + path + '/$ '
            for x, row in enumerate(horizontal):
                if x == 0:
                    new_name = new_name + row
                elif x == 1:
                    new_name = new_name + '_' + row + '.JPG'
            newImgPath.append(new_name)


old_path = []
old_name = []
# newImageList = os.listdir(path)

# print(imageList)
# for cl in imageList:
    # currentImage = cv2.imread(f'{path}/{cl}')
    # images.append(currentImage)
    # classNames.append(os.path.splitext(cl)[0])

for s in imageList:
    if s[0] != '$':
        old_name.append(s)

for item in old_name:
    name = './' + path + '/' + item
    old_path.append(name)


print("new")
print(newImgPath)

# Rename files in the `$ Name_NID.JPG` format
for x, item in enumerate(old_path):
    prev_name = old_path[x]
    newName = newImgPath[x]
    if not os.path.isfile(new_name):
        os.rename(prev_name, newName)

newImageList = os.listdir(path)
for cl in newImageList:
    currentImage = cv2.imread(f'{path}/{cl}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# Call encodings function
encodeListKnown = encodings(images)
# print(encodeListKnown)

# Open Webcam
capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()
    # imgSmall = cv2.resize(img, (0, 0), None, 2.20, 2.20)
    imgSmall = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesInCurrFrame = face_recognition.face_locations(imgSmall)
    encodesInCurrFrame = face_recognition.face_encodings(imgSmall, facesInCurrFrame)

    for encodeFace, faceLoc in zip(encodesInCurrFrame, facesInCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)         # Comparing the encodings of faces in Scanned Images and web cam
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)        # Finding the euclidean distance between the face in Scanned Images and webcam
        # print(faceDist)
        matchIndex = np.argmin(faceDist)                                              # Finding the index of the face with the least distance in the faceDist array
        # print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper().split('_')[0][2:]                   # Extracting name from the image file name
            nid = classNames[matchIndex].split('_')[1]                                # Extracting NID NO. from the image file name
            # print(name)
            # print(nid)
            y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1 / 2, x2 / 2, y2 / 2, x1 / 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)                    # Creating a rectangle using the location of the face
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            record(name, nid)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyWindow('Webcam')
